"""Unified COCOeval / Tiny-COCO validator for RGB-T aerial detection benchmarks."""
import copy
import importlib.util
import math
import os
from types import SimpleNamespace

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics.utils.ops import non_max_suppression

from engine.checkpoint import json_write

TINY_AREA_RANGES = [[1, 1e10], [1, 400], [1, 64], [64, 144], [144, 400], [400, 1024], [1024, 1e10]]
TINY_AREA_LABELS = ["all", "tiny", "tiny1", "tiny2", "tiny3", "small", "reasonable"]


def _cap_predictions(predictions, nc, topk=100):
    by_image = {}
    for pred in predictions:
        if pred["category_id"] not in range(nc):
            raise ValueError(f"Prediction category {pred['category_id']} outside range(0, {nc})")
        by_image.setdefault(pred["image_id"], []).append(pred)
    return [p for group in by_image.values() for p in sorted(group, key=lambda p: -p["score"])[:topk]]


def _load_optional_tiny_cocoeval():
    """Load an external tiny-object COCOeval implementation if RGBT_TINY_COCOEVAL is configured."""
    plugin_path = os.environ.get("RGBT_TINY_COCOEVAL")
    if not plugin_path:
        return None
    spec = importlib.util.spec_from_file_location("rgbt_tiny_cocoeval", plugin_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _eval_tiny_coco(annotation_file, predictions, image_ids):
    gt = COCO(str(annotation_file))
    capped = _cap_predictions(predictions, nc=3, topk=100)
    if capped:
        dt = gt.loadRes(capped)
    else:
        dt = COCO()
        dt.dataset = {
            "images": copy.deepcopy(gt.dataset["images"]),
            "categories": copy.deepcopy(gt.dataset["categories"]),
            "annotations": [],
        }
        dt.createIndex()

    custom_tiny = _load_optional_tiny_cocoeval()
    if custom_tiny is not None:
        custom_tiny.Params.EVAL_STRANDARD = "tiny"
        ev = custom_tiny.COCOeval(gt, dt, "bbox", ignore_uncertain=True, use_iod_for_ignore=True)
    else:
        ev = COCOeval(gt, dt, "bbox")
        ev.params.areaRng = list(TINY_AREA_RANGES)
        ev.params.areaRngLbl = list(TINY_AREA_LABELS)

    ev.params.catIds = [0, 1, 2]
    ev.params.imgIds = sorted(image_ids)
    ev.params.maxDets = [100, 300, 1000]
    ev.params.iouThrs = np.array([0.25, 0.5, 0.75])
    ev.evaluate()
    ev.accumulate()

    precision = ev.eval["precision"]
    metrics = {}
    for i, thr in enumerate((25, 50, 75)):
        for j, area in enumerate(ev.params.areaRngLbl):
            vals = precision[i, :, :, j, -1]
            metrics[f"mAP{thr}_{area}"] = float(vals[vals >= 0].mean()) if (vals >= 0).any() else 0.0
    for k, name in enumerate(("person", "rider", "crowd")):
        vals = precision[1, :, k, 0, -1]
        metrics[f"AP50_{name}"] = float(vals[vals >= 0].mean()) if (vals >= 0).any() else 0.0
    return metrics


def _eval_standard_coco(annotation_file, predictions, image_ids):
    gt = COCO(str(annotation_file))
    if not predictions:
        return dict(mAP50_all=0.0, AP50_95=0.0, AP75=0.0)
    ev = COCOeval(gt, gt.loadRes(predictions), "bbox")
    ev.params.catIds = [0]
    ev.params.imgIds = sorted(image_ids)
    ev.params.maxDets = [100, 300, 1000]
    ev.evaluate()
    ev.accumulate()
    p = ev.eval["precision"][:, :, :, 0, -1]
    avg = lambda a: float(a[a >= 0].mean()) if (a >= 0).any() else 0.0
    return dict(mAP50_all=avg(p[0]), AP50_95=avg(p), AP75=avg(p[5]))


class Validator:
    """Dataset-aware COCOeval validator supporting Tiny-COCO (R) and COCO AP (U, N)."""

    def __init__(self, spec):
        self.spec = spec
        self.selection_metric = spec.get("selection_metric", "mAP50_all")
        keys = ["mAP25_all", "mAP50_all", "mAP75_all"] if spec["evaluator"] == "tiny" else ["mAP50_all", "AP50_95", "AP75"]
        self.metrics = SimpleNamespace(keys=keys)

    @torch.inference_mode()
    def __call__(self, trainer):
        s = self.spec
        nc = int(s["nc"])
        model = trainer.ema.ema.float().eval()
        predictions, ids = [], []
        for batch in trainer.test_loader:
            images = batch["img"].to(trainer.device, non_blocking=True).float() / 255.0
            with torch.autocast("cuda", enabled=False):
                out = model(images)
            out = out[0] if isinstance(out, tuple) else out
            if s["evaluator"] != "tiny":
                scores = out[:, 4:].amax(1)
                order = scores.argsort(1, descending=True)[:, :1000]
                out = out.gather(2, order[:, None].expand(-1, out.shape[1], -1))
            detections = non_max_suppression(
                out.float(),
                conf_thres=s["conf"],
                iou_thres=s["iou"],
                nc=nc,
                agnostic=False,
                max_det=s["max_det"],
                max_time_img=30.0,
            )
            for det, iid, shape, ratio_pad in zip(detections, batch["image_id"], batch["orig_shape"], batch["ratio_pad"]):
                ids.append(iid)
                boxes = det[:, :4].clone()
                (rx, ry), (left, top) = ratio_pad
                boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - left) / rx).clamp(0, shape[1])
                boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - top) / ry).clamp(0, shape[0])
                for b, row in zip(boxes.cpu().tolist(), det.cpu().tolist()):
                    x1, y1, x2, y2 = b
                    if not all(math.isfinite(v) for v in (*b, row[4])):
                        raise ValueError(f"Non-finite detection box/score on image {iid}")
                    cat = int(row[5]) if nc > 1 else 0
                    predictions.append(
                        dict(image_id=iid, category_id=cat, bbox=[x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)], score=row[4])
                    )

        if len(ids) != len(trainer.test_loader.dataset) or len(set(ids)) != len(ids):
            raise AssertionError("Validation loop image count or uniqueness mismatch")

        ann_file = trainer.test_loader.dataset.annotation_file
        if s["evaluator"] == "tiny":
            metrics = _eval_tiny_coco(ann_file, predictions, ids)
        else:
            metrics = _eval_standard_coco(ann_file, predictions, ids)

        if not all(v is None or np.isfinite(v) for v in metrics.values()):
            raise AssertionError(f"Non-finite evaluation metrics: {metrics}")

        tag = getattr(trainer, "evaluation_tag", f"epoch_{trainer.epoch + 1:03d}")
        result = dict(metrics=metrics, images=len(ids), protocol=s, precision="FP32")
        json_write(trainer.save_dir / tag / "metrics.json", result)
        json_write(trainer.save_dir / "latest_metrics.json", dict(epoch=trainer.epoch + 1, **result))
        if tag in ("final_test", "best_validation"):
            json_write(trainer.save_dir / tag / "predictions.json", predictions)
        return dict(metrics, fitness=metrics[self.selection_metric])
