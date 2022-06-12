#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .darknet_od import CSPDarknet
from .network_blocks_od import BaseConv, CSPLayer, DWConv


class YOLOPAFPN_OD(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        # in_features=("dark3", "dark4", "dark5"),
        in_features=("dark3", "dark4", "dark5", "dark6"),
        in_channels=[256, 512, 1024],
        # in_channels=[256, 512],
        depthwise=False,
        act="relu",
        task='od',
    ):
        super().__init__()
        self.task = task
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act, task=task)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        # self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.deconv00 = nn.ConvTranspose2d(int(in_channels[3] * width), int(in_channels[2] * width), 2, 2)
        self.deconv1 = nn.ConvTranspose2d(int(in_channels[1] * width), int(in_channels[1] * width), 2, 2)
        self.deconv2 = nn.ConvTranspose2d(int(in_channels[0] * width), int(in_channels[0] * width), 2, 2)

        self.lateral_conv00 = BaseConv(
            int(in_channels[3] * width), int(in_channels[2] * width), 1, 1, act=act
        )
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )

        self.C4_p4 = CSPLayer(
            int(2 * in_channels[2] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        self.bu_conv4_0 = Conv(
            int(in_channels[4] * width), int(in_channels[4] * width), 3, 2, act=act
        )
        self.bu_conv4_1 = Conv(
            int(in_channels[4] * width), int(in_channels[4] * width), 3, 2, act=act
        )

        if self.task == 'od':
            self.bu_conv0 = Conv(
                int(in_channels[2] * width), int(in_channels[2] * width), 3, 2, act=act
            )
            self.C3_n5 = CSPLayer(
                int(2 * in_channels[2] * width),
                int(in_channels[3] * width),
                round(3 * depth),
                False,
                depthwise=depthwise,
                act=act,
            )

            # self.bu_conv00 = Conv(
            #     int(in_channels[3] * width), int(in_channels[3] * width), 3, 2, act=act
            # )
            self.C4_n5 = CSPLayer(
                int(2 * in_channels[3] * width),
                int(in_channels[4] * width),
                round(3 * depth),
                False,
                depthwise=depthwise,
                act=act,
            )

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        out_features = self.backbone(input)
        # return out_features
        features = [out_features[f] for f in self.in_features]
        # [x2, x1, x0] = features
        [x2, x1, x0, x] = features

        x00 = self.deconv00(x)
        fpn_out0 = torch.cat([x00, x0], 1)
        fpn_out0 = self.C4_p4(fpn_out0)

        fpn_out0 = self.lateral_conv0(fpn_out0)  # 1024->512/32
        # # f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = self.deconv1(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        # f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = self.deconv2(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        if self.task == 'od':
            p_out00 = self.bu_conv0(pan_out0)  # 512->512/32
            x0_0 = self.lateral_conv00(x)
            p_out00 = torch.cat([p_out00, x0_0], 1)  # 256->512/16
            pan_out0_1 = self.C3_n5(p_out00)  # 1024->1024/32
            pan_out0_2 = self.bu_conv4_0(pan_out0_1)
            x0_2 = self.bu_conv4_1(x)

            pan_out000 = torch.cat([pan_out0_2, x0_2], 1)
            pan_out000 = self.C4_n5(pan_out000)  # 1024->1024/64

            outputs = (pan_out2, pan_out1, pan_out0, pan_out0_1, pan_out000)
        else:
            outputs = (pan_out2, pan_out1, pan_out0)
        return outputs
