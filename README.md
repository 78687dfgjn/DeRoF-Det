# DeRoF-Det: Decomposition, Routing and Frequency Refinement Detector

Dual-stream YOLOv8m-P2 (`P2-P5` detection heads, <50M parameters) RGB-Thermal object detector built around three cohesive modules:

- **Module A · `SPMR` (`models/spmr.py`)** — *Shared-Private Multiscale Recalibration*: cross-modal shared-private decomposition with spatial reconstruction constraint (`0.005 * aux_scale * MSE`) coupled with channel prior recalibration and multi-scale orthogonal strip attention (`MultiscaleStripAttention`, supporting configurable residual `gamma_init`, `0.05` in `slim1-tuned`).
- **Module B · `MEAF` (`models/meaf.py`)** — *Modality-Expert Atrous Fusion*: pixel-wise 3-expert modality routing (`R'`, `T'`, `R'+T'`), spatial foreground prior estimation, pre-projected foreground/background 4-branch dilated ASPP (`d = 1, 3, 5, 7`), and 4-channel uncertainty map emission (`cond_k`).
- **Module C · `CGSR` (`models/cgsr.py`)** — *Condition-Guided Spectral Refinement*: FP32 2D-DCT orthogonal spectral separation, pooled frequency channel modulation, and zero-initialized cross-modal condition gating (`α⊙H + β⊙L`) applied to PAN-FPN outputs `{18: P2, 21: P3, 24: P4, 27: P5}` (dense `3x3` refinement on P2, depthwise-separable refinement on P3-P5).

## Repository Layout

```text
DeRoF-Det/
├── configs/
│   ├── yolov8m-p2.yaml      # Dual-stream YOLOv8m P2-P5 backbone & PAN-FPN topology
│   └── dataset_rgbt.yaml    # Dataset manifest specification for R / U / N tiers
├── models/
│   ├── spmr.py              # Module A: Shared-Private Multiscale Recalibration (SPMR)
│   ├── meaf.py              # Module B: Modality-Expert Atrous Fusion (MEAF)
│   ├── cgsr.py              # Module C: Condition-Guided Spectral Refinement (CGSR)
│   ├── baselines.py         # Baselines: P2AddModel, P2ConcatC2fModel, P2LateModel, P2BaseV2Model
│   └── detector.py          # DeRoFDet unified model (slim1 / slimB + ablation masks)
├── data/
│   └── paired_dataset.py    # Paired RGB-T mosaic, HSV, perspective, letterbox & splits
├── engine/
│   ├── checkpoint.py        # Atomic state serialization, resume & RNG restoration
│   ├── protocols.py         # Dataset protocols, strict APStopper & experiment registry
│   ├── evaluator.py         # COCOeval & Tiny-COCO evaluators across R, U, and N benchmarks
│   └── trainer.py           # Mixed-precision training loop, tail FP32 guard & logging
├── train.py                 # Unified CLI entrypoint
└── requirements.txt
```

## Supported Benchmarks & Configurations

| Dataset | Experiment IDs | Input HxW | Classes | Metric / Selection |
| :--- | :--- | :--- | :--- | :--- |
| RGBTDronePerson | `R-{ABC,A,B,C,AB,AC,BC}-{slim1,slimB,slim1-tuned}-in1k-fast` | `512 x 640` | 3 (`person`, `rider`, `crowd`) | Tiny-COCO `mAP50_all` |
| VTUAV-det | `U-ABC-{slim1,slimB,slim1-tuned}-in1k-fast` | `384 x 640` | 1 (`person`) | COCO `mAP50_all` |
| NII-CU MAPD | `N-{ABC-slim1,ABC-slimB,P2-Add,P2-ConcatC2f}-in1k-fast` | `384 x 640` | 1 (`person`) | COCO `mAP50_all` |
| NII-CU 4ch / HD | `{N4ch,N960,N1280}-{P2-Base-v2,ABC-slim1}-in1k-fast`, `N4ch-ABC-slim1-tuned-in1k-fast` | `480x640` / `544x960` / `736x1280` | 1 (`person`) | COCO `AP50_95` |

## Usage

```bash
pip install -r requirements.txt

# Place ImageNet-1K pretrained YOLOv8m-cls weights at ./weights/yolov8m-cls.pt
# Place dataset JSON manifests & *_protocol.json under ./datasets (or pass --data-root)

python train.py R-ABC-slim1-tuned-in1k-fast --data-root ./datasets --weights ./weights/yolov8m-cls.pt
python train.py U-ABC-slim1-tuned-in1k-fast --data-root ./datasets --weights ./weights/yolov8m-cls.pt
python train.py N4ch-ABC-slim1-tuned-in1k-fast --data-root ./datasets --weights ./weights/yolov8m-cls.pt
python train.py N960-ABC-slim1-in1k-fast --batch 8 --data-root ./datasets --weights ./weights/yolov8m-cls.pt
```
