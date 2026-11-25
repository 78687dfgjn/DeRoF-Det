"""Unified paired RGB-Thermal dataset supporting centered/origin letterbox, mosaic, HSV, and modality degradation."""
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from ultralytics.data.augment import RandomHSV, RandomPerspective
from ultralytics.data.build import seed_worker
from ultralytics.utils.instance import Instances


class PairedPerspective(RandomPerspective):
    """Apply identical geometric affine/perspective warping to concatenated 6-channel RGB+TIR images."""

    def affine_transform(self, img, border):
        rgb, matrix, scale = super().affine_transform(img[:, :, :3], border)
        warp = cv2.warpPerspective if self.perspective else cv2.warpAffine
        ir = warp(
            img[:, :, 3:],
            matrix if self.perspective else matrix[:2],
            dsize=self.size,
            borderValue=(114, 114, 114),
        )
        return np.concatenate((rgb, ir), axis=2), matrix, scale


def letterbox(image, boxes, hw, centered=True):
    """Resize and pad an RGB+TIR image to `hw` while transforming `boxes` consistently."""
    th, tw = hw
    h, w = image.shape[:2]
    out = np.full((th, tw, image.shape[2]), 114, dtype=np.uint8)
    if not centered and h <= th and w <= tw and (w == tw or h == th):
        out[:h, :w] = image
        return out, boxes.copy(), ((1.0, 1.0), (0, 0))
    ratio = min(th / h, tw / w)
    rh, rw = round(h * ratio), round(w * ratio)
    resized = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR)
    if centered:
        left, top = round((tw - rw) / 2 - 0.1), round((th - rh) / 2 - 0.1)
    else:
        left, top = 0, 0
    out[top : top + rh, left : left + rw] = resized
    out_boxes = boxes.copy()
    out_boxes[:, [0, 2]] = out_boxes[:, [0, 2]] * (rw / w) + left
    out_boxes[:, [1, 3]] = out_boxes[:, [1, 3]] * (rh / h) + top
    return out, out_boxes, ((rw / w, rh / h), (left, top))


def _apply_random_degradation(img):
    """Randomly simulate single-modality dropout, blur, sensor noise, or block occlusion."""
    mode = random.randrange(5)
    if mode == 0:
        return img
    offset = 0 if random.random() < 0.5 else 3
    out = img.copy()
    patch = out[offset : offset + 3].transpose(1, 2, 0).copy()
    h, w = patch.shape[:2]
    if mode == 1:
        patch[:] = 0
    elif mode == 2:
        patch = cv2.GaussianBlur(patch, (5, 5), 1.5)
    elif mode == 3:
        patch = np.clip(patch.astype(np.float32) + np.random.normal(0, 20, patch.shape), 0, 255).astype(np.uint8)
    else:
        y = random.randrange(max(1, h - h // 4))
        x = random.randrange(max(1, w - w // 4))
        patch[y : y + h // 4, x : x + w // 4] = 114
    out[offset : offset + 3] = patch.transpose(2, 0, 1)
    return out


class PairedRGBTDataset(Dataset):
    """Paired RGB-Thermal dataset loader for RGBTDronePerson (R), VTUAV-det (U), and NII-CU (N) splits."""

    def __init__(self, spec, split, hyp, data_root=None, degrade=False, fixture=False):
        self.spec = spec
        self.split = split
        self.hyp = hyp
        self.train = split == "train"
        self.mosaic = self.train
        self.rect = False
        self.hw = tuple(spec["hw"])
        self.centered_pad = spec["dataset"] == "R"
        self.degrade = bool(degrade and self.train)
        self.root = Path(data_root or os.environ.get("RGBT_DATA_ROOT", "datasets"))

        prefix = spec.get("prefix")
        if prefix:
            manifest = self.root / f"{prefix}_{split}.json"
            self.pre_resized = True
        elif spec["dataset"] == "U" and (self.root / f"U_640x360_{split}.json").exists():
            manifest = self.root / f"U_640x360_{split}.json"
            self.pre_resized = True
        else:
            manifest = self.root / f"{spec['dataset']}_{split}.json"
            self.pre_resized = False

        self.images = json.loads(manifest.read_text(encoding="utf-8"))
        if fixture:
            self.images = self.images[:32] if self.train else self.images[:8]

        proto_file = spec.get("proto") or f"{spec['dataset']}_protocol.json"
        meta = json.loads((self.root / proto_file).read_text(encoding="utf-8"))
        self.annotation_file = meta["annotation"]

        self.perspective = PairedPerspective(
            degrees=hyp.degrees,
            translate=hyp.translate,
            scale=hyp.scale,
            shear=hyp.shear,
            perspective=hyp.perspective,
        )
        self.hsv = RandomHSV(hyp.hsv_h, hyp.hsv_s, hyp.hsv_v)

    def __len__(self):
        return len(self.images)

    def close_mosaic(self, hyp=None):
        self.mosaic = False

    def raw(self, index):
        row = self.images[index]
        rgb = cv2.imread(row["rgb"])
        ir = cv2.imread(row["ir"])
        if rgb is None or ir is None or rgb.shape != ir.shape:
            raise ValueError(f"Invalid RGB/TIR pair at index {index}: {row.get('rgb')} / {row.get('ir')}")
        boxes = np.array(row["boxes"], np.float32).reshape(-1, 4)
        if self.pre_resized:
            ow = row.get("orig_width", rgb.shape[1])
            oh = row.get("orig_height", rgb.shape[0])
            boxes[:, [0, 2]] *= row.get("width", rgb.shape[1]) / ow
            boxes[:, [1, 3]] *= row.get("height", rgb.shape[0]) / oh
        classes = np.array(row["classes"], np.float32).reshape(-1, 1)
        return np.concatenate((rgb, ir), axis=2), boxes, classes

    def _mosaic_sample(self, index):
        h, w = self.hw
        yc, xc = int(random.uniform(h / 2, h * 1.5)), int(random.uniform(w / 2, w * 1.5))
        canvas = np.full((h * 2, w * 2, 6), 114, np.uint8)
        boxes_out, classes_out = [], []
        indices = [index] + random.choices(range(len(self)), k=3)
        for tile, j in enumerate(indices):
            image, boxes, classes = self.raw(j)
            image, boxes, _ = letterbox(image, boxes, self.hw, centered=self.centered_pad)
            if tile == 0:
                x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
                sx, sy = w - (x2 - x1), h - (y2 - y1)
            elif tile == 1:
                x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, 2 * w), yc
                sx, sy = 0, h - (y2 - y1)
            elif tile == 2:
                x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(yc + h, 2 * h)
                sx, sy = w - (x2 - x1), 0
            else:
                x1, y1, x2, y2 = xc, yc, min(xc + w, 2 * w), min(yc + h, 2 * h)
                sx, sy = 0, 0
            canvas[y1:y2, x1:x2] = image[sy : sy + y2 - y1, sx : sx + x2 - x1]
            boxes += np.array([x1 - sx, y1 - sy, x1 - sx, y1 - sy], np.float32)
            boxes_out.append(boxes)
            classes_out.append(classes)

        boxes = np.concatenate(boxes_out, axis=0)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, 2 * w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, 2 * h)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        return dict(
            img=canvas,
            cls=np.concatenate(classes_out, axis=0)[valid],
            instances=Instances(boxes[valid], segments=np.zeros((0, 1000, 2), np.float32), bbox_format="xyxy", normalized=False),
            mosaic_border=(-h // 2, -w // 2),
        )

    def __getitem__(self, index):
        image, boxes, classes = self.raw(index)
        orig_shape = image.shape[:2]
        ratio_pad = None
        if self.train and self.mosaic and random.random() < self.hyp.mosaic:
            labels = self._mosaic_sample(index)
        else:
            image, boxes, ratio_pad = letterbox(image, boxes, self.hw, centered=self.centered_pad)
            labels = dict(
                img=image,
                cls=classes,
                instances=Instances(boxes, segments=np.zeros((0, 1000, 2), np.float32), bbox_format="xyxy", normalized=False),
            )

        if self.train:
            labels = self.perspective(labels)
            visible = dict(img=np.ascontiguousarray(labels["img"][:, :, :3]))
            self.hsv(visible)
            labels["img"][:, :, :3] = visible["img"]
            if random.random() < self.hyp.fliplr:
                labels["img"] = np.fliplr(labels["img"])
                labels["instances"].fliplr(self.hw[1])

        image, boxes = labels["img"], labels["instances"].bboxes.copy()
        if image.shape != (*self.hw, 6):
            raise AssertionError(f"Expected image shape {(*self.hw, 6)}, got {image.shape}")

        rgb_ir = np.concatenate((image[:, :, :3][:, :, ::-1], image[:, :, 3:][:, :, ::-1]), axis=2)
        tensor_img = np.ascontiguousarray(rgb_ir.transpose(2, 0, 1))
        if self.degrade:
            tensor_img = _apply_random_degradation(tensor_img)

        xywh = boxes.copy()
        xywh[:, :2] = (boxes[:, :2] + boxes[:, 2:]) / 2.0
        xywh[:, 2:] = boxes[:, 2:] - boxes[:, :2]
        xywh /= np.array([self.hw[1], self.hw[0], self.hw[1], self.hw[0]], np.float32)

        if self.pre_resized and not self.train:
            row = self.images[index]
            ow = row.get("orig_width", orig_shape[1])
            oh = row.get("orig_height", orig_shape[0])
            orig_shape = (oh, ow)
            ratio_pad = ((row.get("width", self.hw[1]) / ow, row.get("height", self.hw[0]) / oh), (0, 0))

        return dict(
            img=torch.from_numpy(tensor_img),
            cls=torch.from_numpy(labels["cls"]),
            bboxes=torch.from_numpy(xywh),
            image_id=self.images[index]["id"],
            orig_shape=orig_shape,
            ratio_pad=ratio_pad,
        )

    @staticmethod
    def collate_fn(batch):
        return dict(
            img=torch.stack([x["img"] for x in batch]),
            cls=torch.cat([x["cls"] for x in batch]),
            bboxes=torch.cat([x["bboxes"] for x in batch]),
            batch_idx=torch.cat([torch.full((len(x["cls"]),), i) for i, x in enumerate(batch)]),
            image_id=[x["image_id"] for x in batch],
            orig_shape=[x["orig_shape"] for x in batch],
            ratio_pad=[x["ratio_pad"] for x in batch],
        )


class _EpochLoader(torch.utils.data.DataLoader):
    def reset(self):
        pass


def make_loader(dataset, batch, workers, training):
    """Build deterministic shuffled train/eval DataLoader with persistent workers when workers > 0."""
    kw = dict(
        batch_size=batch,
        num_workers=workers,
        shuffle=training,
        drop_last=training,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )
    if training:
        kw["generator"] = torch.Generator().manual_seed(6148914691236517205)
    if workers > 0:
        kw["worker_init_fn"] = seed_worker
        kw["persistent_workers"] = training
        kw["prefetch_factor"] = 4
    return _EpochLoader(dataset, **kw)
