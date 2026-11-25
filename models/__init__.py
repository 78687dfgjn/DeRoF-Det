"""DeRoF-Det dual-stream RGB-T P2-P5 detector and SPMR / MEAF / CGSR modules."""
from models.detector import (
    CHANNELS,
    CONFIG_YAML,
    DeRoFDet,
    LEVELS,
    NECK_TO_LEVEL,
    TAIL_BATCH_THRESHOLD,
    VARIANT_SPECS,
    transfer_coco_p2,
    transfer_imagenet_backbone,
)
from models.baselines import P2AddModel, P2BaseV2Model, P2ConcatC2fModel, P2LateModel
from models.cgsr import CGSR
from models.meaf import ASPPBlock, MEAF, PriorHead
from models.spmr import ChannelRecalibration, MultiscaleStripAttention, Residual, SPMR, isolated_seed

__all__ = [
    "ASPPBlock",
    "CGSR",
    "CHANNELS",
    "CONFIG_YAML",
    "ChannelRecalibration",
    "DeRoFDet",
    "LEVELS",
    "MEAF",
    "MultiscaleStripAttention",
    "NECK_TO_LEVEL",
    "P2AddModel",
    "P2BaseV2Model",
    "P2ConcatC2fModel",
    "P2LateModel",
    "PriorHead",
    "Residual",
    "SPMR",
    "TAIL_BATCH_THRESHOLD",
    "VARIANT_SPECS",
    "isolated_seed",
    "transfer_coco_p2",
    "transfer_imagenet_backbone",
]
