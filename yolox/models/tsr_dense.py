import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple
from torchvision.models import DenseNet


class DenseNetTSR(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """

    def __init__(
        self,
        growth_rate: int = 32,
        block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0,
        num_classes: int = 100,
        memory_efficient: bool = False
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.model = DenseNet(block_config=(4,), num_init_features=32, num_classes=self.num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: Tensor, targets=None) -> Tensor:
        cls_preds = self.model(x)
        if self.training:
            assert targets is not None
            gt_cls = targets.squeeze()[:, 0].to(torch.int64)     # target = tensor([[[cls, x0, y0, x1, y1]]])
            # print(cls_preds)
            # print(targets)
            # print(gt_cls)
            # print(cls_preds.shape, gt_cls.shape)
            # gt_cls = F.one_hot(targets.to(torch.int64), self.num_classes)
            loss = self.loss_fn(cls_preds, gt_cls)
            outputs = {
                "total_loss": loss,
                "iou_loss": 0.0,
                "l1_loss": 0.0,
                "conf_loss": 0.0,
                "cls_loss": 0.0,
            }
            # gt_cls_per_image = F.one_hot(gt_classes.to(torch.int64), self.num_classes)
            # pair_wise_cls_loss = F.binary_cross_entropy()
        else:
            outputs = self.decoding(cls_preds)
        return outputs

    def pre_decoding(self, preds):
        """
        preds = [batch_size, class_score]
        output = [batch_size, det_num, [box, obj, score]]
        """
        score = F.softmax(preds, dim=1)
        batch_size = score.shape[0]
        obj = torch.ones(batch_size, 1, device=score.device)
        box = torch.tensor([50, 50, 100, 100], device=score.device) # [cent_x, cent_y, wid, height]
        boxes = box.repeat(batch_size, 1)
        output = torch.cat([boxes, obj, score], 1).unsqueeze(1)     # output should like [batch_size, det_num, det_detail]
        return output

    def decoding(self, outputs, conf_thrsh=0.0):
        score = F.softmax(outputs, dim=1)
        class_conf, class_pred = torch.max(score[:, :self.num_classes], 1, keepdim=True)
        mask = (class_conf > conf_thrsh).squeeze()
        cls_out = torch.cat((class_conf, class_pred.float()), 1)
        cls_out = cls_out[mask]
        return cls_out

