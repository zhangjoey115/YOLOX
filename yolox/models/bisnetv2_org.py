#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F


np_dict = {}

"""
    Block Modules
"""
class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat1 = self.conv(x)
        feat2 = self.bn(feat1)
        feat3 = self.relu(feat2.clone())

        # x_np = x.cpu().detach().numpy()
        # feat1_np = feat1.cpu().detach().numpy()
        # feat2_np = feat2.cpu().detach().numpy()
        # feat3_np = feat3.cpu().detach().numpy()
        return feat3


class CEBlock(nn.Module):
    def __init__(self, channel) -> None:
        super(CEBlock, self).__init__()
        self.bn = nn.BatchNorm2d(channel)
        self.conv_gap = ConvBNReLU(channel, channel, 1, stride=1, padding=0)
        self.conv_last = ConvBNReLU(channel, channel, 3, stride=1)

    def forward(self, x):
        feat = torch.mean(x, dim=(2,3), keepdim=True)
        feat = self.bn(feat)
        feat = self.conv_gap(feat)
        feat = feat * x
        feat = feat + x
        feat = self.conv_last(feat)
        return feat

#坐标注意力机制
class CA_Block(nn.Module):
    def __init__(self, channel, h, w, reduction=16) -> None:
        super(CA_Block, self).__init__()
        self.h = h
        self.w = w
        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel//reduction, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel//reduction)
        self.F_h = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        x_h = self.avg_pool_x(x).permute(0,1,3,2)
        x_w = self.avg_pool_y(x)
        x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 3)))
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], 3)
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out


class SpatialAttention(nn.Module):
    def __init__(self) -> None:
        super(SpatialAttention, self).__init__()
        self.conv1 = ConvBNReLU(2, 1, 3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)
        output = x + x*out
        output = self.relu(output)
        return output


class NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None) -> None:
        super(NonLocalBlockND, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
        
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels)
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        # np_dict['np_seg.nonlocal1'] = z.cpu().detach().numpy()
        return z


"""
    FFM - BGALayer
"""
class BGALayer(nn.Module):
    def __init__(self, channel_config) -> None:
        super(BGALayer, self).__init__()
        self.left2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.right1 = nn.Sequential(
            nn.Conv2d(channel_config[1], channel_config[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channel_config[0]),
        )
        self.right2 = nn.Sequential(
            nn.Conv2d(channel_config[0], channel_config[0], kernel_size=3, stride=1, padding=1, groups=channel_config[0], bias=False),
            nn.BatchNorm2d(channel_config[0]),
            nn.Conv2d(channel_config[0], channel_config[0], kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(channel_config[0], channel_config[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channel_config[0]),
            nn.ReLU(inplace=True),
        )
        self.right_conv = ConvBNReLU(channel_config[0], channel_config[0], 3, stride=1, padding=1)
        
        # from pytorch_quantization import quant_modules
        # quant_modules.deactivate()
        self.right1_deconv = nn.ConvTranspose2d(channel_config[0], channel_config[0], 4, 4, padding=0, output_padding=0, bias=True)
        self.right_deconv_1 = nn.ConvTranspose2d(channel_config[0], channel_config[0], 4, 4, padding=0, output_padding=0, bias=True)
        # quant_modules.initialize()

    def forward(self, feat_d, feat_r):
        dsize = feat_d.size()[2:]
        feat_s, feat_4 = feat_r
        left2 = self.left2(feat_d)
        # np_dict['np_ffm.maxpool2'] = left2.cpu().detach().numpy()
        right1 = self.right1(feat_4)
        right2 = self.right2(feat_s)
        right1 = self.right1_deconv(right1)

        left = feat_d * torch.sigmoid(right1)
        right = left2 * torch.sigmoid(right2)
        # np_dict['np_ffm.sigmul'] = right.cpu().detach().numpy()
        right = self.right_conv(right)
        # np_dict['np_ffm.cbr'] = right.cpu().detach().numpy()
        right = self.right_deconv_1(right)
        out = left + right
        # np_dict['np_ffm.left'] = left.cpu().detach().numpy()
        # np_dict['np_ffm.right'] = right.cpu().detach().numpy()
        # np_dict['np_ffm.out'] = out.cpu().detach().numpy()
        return out


"""
    SegmentBranch
"""
class StemBlock(nn.Module):
    def __init__(self, channel_config):
        super(StemBlock, self).__init__()
        self.conv_1 = ConvBNReLU(3, channel_config[0], 3, stride=2)
        self.conv_2 = nn.Sequential(
            ConvBNReLU(channel_config[0], channel_config[1], 3, stride=2),
            ConvBNReLU(channel_config[1], channel_config[1], 3, stride=1,
                padding=0, groups=channel_config[1], bias=False),
            ConvBNReLU(channel_config[1], channel_config[1], 1, stride=1,
                padding=0, bias=False),
        )
        self.left = nn.Sequential(
            ConvBNReLU(channel_config[1], channel_config[0], 1, stride=1, padding=0),
            ConvBNReLU(channel_config[0], channel_config[1], 3, stride=2),
        )
        self.right = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.fuse = ConvBNReLU(channel_config[2], channel_config[3], 3, stride=1)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        feat = self.conv_1(x)
        feat = self.conv_2(feat)
        feat_left = self.left(feat)
        feat_right = self.right(feat)
        feat = torch.cat([feat_left, feat_right], 1)
        feat = self.fuse(feat)
        feat = self.spatial_att(feat)

        return feat


class GELayerS1(nn.Module):
    def __init__(self, in_chan, out_chan, exp_ratio=2):
        super(GELayerS1, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, mid_chan, 3, stride=1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(mid_chan, mid_chan, kernel_size=3, stride=1, padding=1, groups=mid_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_chan, out_chan, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.conv2[1].last_bn = True
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv(feat)
        feat = self.conv2(feat)
        feat = feat + x
        feat = self.relu(feat)
        return feat


class GELayerS2(nn.Module):
    def __init__(self, in_chan, out_chan, exp_ratio=2):
        super(GELayerS2, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, mid_chan, 3, stride=1)
        self.dwconv1 = nn.Sequential(
            nn.Conv2d(mid_chan, mid_chan, kernel_size=3, stride=2, padding=1, groups=mid_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
        )
        self.dwconv2 = nn.Sequential(
            nn.Conv2d(mid_chan, mid_chan, kernel_size=3, stride=1, padding=1, groups=mid_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_chan, out_chan, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.conv2[1].last_bn = True
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_chan, in_chan, kernel_size=3, stride=2, padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(in_chan),
            nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv1(feat)
        feat = self.dwconv2(feat)
        feat = self.conv2(feat)
        shortcut = self.shortcut(x)
        feat = feat + shortcut
        feat = self.relu(feat)
        return feat


class SegmentBranch(nn.Module):
    def __init__(self, channel_config):
        super(SegmentBranch, self).__init__()
        self.S1S2 = StemBlock(channel_config)
        self.S3_1 = GELayerS2(channel_config[3], channel_config[4])
        self.S3_2 = GELayerS1(channel_config[4], channel_config[4])
        self.S4_1 = GELayerS2(channel_config[4], channel_config[5])
        self.S4_2 = GELayerS1(channel_config[5], channel_config[5])
        self.S5_1 = NonLocalBlockND(channel_config[5])
        self.S5_2 = CEBlock(channel_config[5])
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        feat2 = self.S1S2(x)            # 1/8
        feat3_1 = self.S3_1(feat2)      # 1/16
        feat3_2 = self.S3_2(feat3_1)
        feat4_1 = self.S4_1(feat3_2)    # 1/32
        feat4_2 = self.S4_2(feat4_1)
        feat5_1 = self.S5_1(feat4_2)    # 1/32
        # np_dict['np_seg.np_nonlocal'] = feat5_1.cpu().detach().numpy()
        feat5_2 = self.S5_2(feat5_1)
        feat3_1_pool = self.pool(feat3_1)
        feat3_2_pool = self.pool(feat3_2)
        feat4 = torch.cat([feat3_1_pool, feat3_2_pool, feat4_2], dim=1)
        # np_dict['np_seg.np_concat'] = feat4.cpu().detach().numpy()
        # np_dict['np_seg.np_ce'] = feat5_2.cpu().detach().numpy()
        return feat4, feat5_2


"""
    Detail Branch
"""
class DetailBranch(nn.Module):
    def __init__(self, channel_config) -> None:
        super(DetailBranch, self).__init__()

        self.conv1 = ConvBNReLU(3, channel_config[0], 3, stride=2)      # 1/2
        self.conv3 = ConvBNReLU(channel_config[0], channel_config[1], 3, stride=2)  # 1/4
        self.conv4 = ConvBNReLU(channel_config[1], channel_config[2], 3, stride=2)  # 1/8
        self.conv5 = ConvBNReLU(channel_config[3], channel_config[3], 1, stride=1, padding=0)  # 1/8
        self.conv6 = ConvBNReLU(channel_config[3], channel_config[4], 3, stride=1)  # 1/8
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        feat_1 = self.conv1(x)
        feat_3 = self.conv3(feat_1)
        feat_4 = self.conv4(feat_3)

        feat_pool_1 = self.pool(feat_1)
        feat_pool_1 = self.pool(feat_pool_1)
        feat_pool_3 = self.pool(feat_3)
        
        feat = torch.cat([feat_pool_1, feat_pool_3, feat_4], 1)
        feat = self.conv5(feat)
        feat = self.conv6(feat)
        # np_dict['np_detail.out'] = feat.cpu().detach().numpy()
        return feat


"""
    Head
"""
class SegmentHead(nn.Module):
    def __init__(self, channel_config, n_classes, scale) -> None:
        super(SegmentHead, self).__init__()
        self.in_chan = channel_config[0]
        self.mid_chan = channel_config[1]
        self.last_chan = self.mid_chan//2
        self.conv_1 = ConvBNReLU(self.in_chan, self.mid_chan, 3, stride=1)
        self.conv_2 = ConvBNReLU(self.mid_chan, self.last_chan, 3, stride=1)
        self.drop = nn.Dropout(0.1)
        self.conv_out = nn.Conv2d(self.last_chan, n_classes, kernel_size=1, stride=1, padding=0, bias=True)

        # from pytorch_quantization import quant_modules
        # quant_modules.deactivate()
        self.deconv = nn.ConvTranspose2d(n_classes, n_classes, scale, scale, padding=0, output_padding=0, bias=True)
        # quant_modules.initialize()

    def forward(self, x):
        feat = self.conv_1(x)
        feat = self.conv_2(feat)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        feat = self.deconv(feat)
        return feat


"""
    Seg Model
"""
class SegNet(nn.Module):
    def __init__(self, DetailBranch_config, SegmentBranch_config, BGALayer_config,
                 SegmentHead_config, Aux_SegmentHead_config, lane_class=3, roadway_class=3, train_flag=False):
        super(SegNet, self).__init__()
        self.train_flag = train_flag
        self.Detail_branch = DetailBranch(DetailBranch_config)
        self.Seg_branch = SegmentBranch(SegmentBranch_config)
        self.ffm = BGALayer(BGALayer_config)
        
        self.lane_head = SegmentHead(SegmentHead_config, lane_class, scale=2)
        self.lane_head_aux_seg_1 = SegmentHead(SegmentHead_config, lane_class, scale=8)
        self.lane_head_aux_seg_2 = SegmentHead(Aux_SegmentHead_config, lane_class, scale=8)

        self.roadway_head = SegmentHead(SegmentHead_config, roadway_class, scale=2)
        self.roadway_head_aux_seg_1 = SegmentHead(SegmentHead_config, roadway_class, scale=8)
        self.roadway_head_aux_seg_2 = SegmentHead(Aux_SegmentHead_config, roadway_class, scale=8)

    def forward(self, x):
        detail_feat = self.Detail_branch(x)
        seg_feat = self.Seg_branch(x)
        ffm_feat = self.ffm(detail_feat, seg_feat)
        lane_head_feat = self.lane_head(ffm_feat)
        roadway_head_feat = self.roadway_head(ffm_feat)
        if self.train_flag is True:
            lane_seg_aux_head_layer1 = self.lane_head_aux_seg_1(seg_feat[0])
            lane_seg_aux_head_layer2 = self.lane_head_aux_seg_2(seg_feat[1])
            roadway_seg_aux_head_layer1 = self.roadway_head_aux_seg_1(seg_feat[0])
            roadway_seg_aux_head_layer2 = self.roadway_head_aux_seg_2(seg_feat[1])
            return lane_head_feat, lane_seg_aux_head_layer1, lane_seg_aux_head_layer2, \
                roadway_head_feat, roadway_seg_aux_head_layer1, roadway_seg_aux_head_layer2
        else:
            return lane_head_feat, roadway_head_feat
