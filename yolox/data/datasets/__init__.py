#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

from .coco import COCODataset
from .coco_classes import COCO_CLASSES
from .datasets_wrapper import ConcatDataset, Dataset, MixConcatDataset
from .mosaicdetection import MosaicDetection
from .voc import VOCDetection
from .tt100k import TT100kDetection
from .tt100k_classes import TT100K_CLASSES
from .tsr_zo import TSR_ZO_Detection
from .tsr_zo_classes import TSR_ZO_CLASSES
from .tsr_2nd import TSR_2ND_Detection
from .tsr_2nd_classes import TSR_2ND_CLASSES
