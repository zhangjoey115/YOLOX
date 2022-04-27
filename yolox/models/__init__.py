#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

# from .darknet import CSPDarknet, Darknet
from .losses import IOUloss
from .yolo_fpn import YOLOFPN
from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN
from .yolox import YOLOX
from .tsr_dense import DenseNetTSR
from .yolo_head_od import YOLOXHead_OD
from .yolo_pafpn_od import YOLOPAFPN_OD
from .yolox_od import YOLOX_OD
from .bisnetv2 import SegNet
