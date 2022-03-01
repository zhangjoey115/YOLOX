# encoding: utf-8
import os

import torch
import torch.nn as nn
import torch.distributed as dist

from yolox.data import get_yolox_datadir
from yolox.exp import Exp as MyExp
from yolox.data.datasets.tt100k_classes import TT100K_CLASSES


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = len(TT100K_CLASSES)      # 3    # 45
        self.depth = 0.33
        self.width = 0.25
        self.warmup_epochs = 1
        self.max_epoch = 30
        self.no_aug_epochs = 10
        self.no_aug_eval_epochs = 5
        self.eval_interval = 10
        self.basic_lr_per_img = 1.0e-3 / 8.0      # devide batch_size
        self.min_lr_ratio = 0.01
        # self.input_size = (416, 416)
        # self.test_size = (416, 416)
        self.input_size = (640, 640)
        self.test_size = (640, 640)
        # self.multiscale_range = 0
        self.mixup_prob = 0.0       # 1.0
        self.mosaic_scale = (0.5, 2)
        # self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        # self.exp_name = "test_nano640_210906/yolox_tt100k_nano11_640Pre_newCs_mulR0_0_0.5_20_1e-3"
        self.exp_name = "tsr_tt100k_3_20220222/yolox_tt100k_nano3_640_200_1e-3_0p01_rm_unused"

    def get_model(self, sublinear=False):

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        if "model" not in self.__dict__:
            from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
            # in_channels = [256, 512, 1024]
            in_channels = [256, 512, 1024]
            # strides = [8, 16],
            # NANO model use depthwise = True, which is main difference.
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, depthwise=False, act='relu')
            head = YOLOXHead(self.num_classes, self.width, strides=[8, 16], in_channels=[256, 512], depthwise=False, act='relu')
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        from yolox.data import (
            TSR_ZO_Detection,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            worker_init_reset_seed,
        )
        from yolox.utils import (
            wait_for_the_master,
            get_local_rank,
        )
        local_rank = get_local_rank()

        with wait_for_the_master(local_rank):
            dataset = TSR_ZO_Detection(
                data_dir=os.path.join(get_yolox_datadir(), "tsr_voc"),
                image_sets=[('tt100k_voc_dataset', 'train')],
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=50,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob),
                    # bgr_means=(0.406, 0.456, 0.485),
                    # std=(0.225, 0.224, 0.229)),
                cache=cache_img,
            )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob),
                # bgr_means=(0.406, 0.456, 0.485),
                # std=(0.225, 0.224, 0.229)),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            perspective=self.perspective,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(
            len(self.dataset), seed=self.seed if self.seed else 0
        )

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.data import TSR_ZO_Detection, ValTransform

        valdataset = TSR_ZO_Detection(
            data_dir=os.path.join(get_yolox_datadir(), "tsr_voc"),
            image_sets=[('tt100k_voc_dataset', 'test')],
            img_size=self.test_size,
            preproc=ValTransform(),
            # preproc=ValTransform(legacy=legacy),
                # bgr_means=(0.406, 0.456, 0.485),
                # std=(0.225, 0.224, 0.229)),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import TSR_ZO_Evaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        evaluator = TSR_ZO_Evaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
        )
        return evaluator
