# encoding: utf-8
import os

import torch
import torch.nn as nn
import torch.distributed as dist

from yolox.data import get_yolox_datadir
from yolox.exp import Exp as MyExp
from yolox.data.datasets import TSR_ZO_CLASSES, TSR_ZO_CLASSES_45


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = len(TSR_ZO_CLASSES)      # 3    # 45
        self.depth = 0.33
        self.width = 0.25
        self.warmup_epochs = 1
        self.max_epoch = 200
        self.no_aug_epochs = 50
        self.no_aug_eval_epochs = 5
        self.eval_interval = 10
        self.basic_lr_per_img = 1.0e-4 / 12.0      # devide batch_size
        self.min_lr_ratio = 0.05
        # self.input_size = (448, 768)
        # self.input_size = (256, 768)
        self.input_size = (540, 960)            # (320, 960)
        self.test_size = (540, 960)     # (256, 768)
        # self.multiscale_range = 0
        self.mixup_prob = 0.0       # 1.0
        self.mosaic_scale = (0.5, 2)
        self.flip_prob = 0.0
        # self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        # self.exp_name = "train_tsr_zo_768_211019/yolox_tsr_zo_nano5_320norm_p_om_300_1e-4"
        # self.exp_name = "train_tsr_zo_960_211116_v01_03/yolox_tsr_zo_head2_320norm_300_1e-3_0p005"
        self.exp_name = "train_tsr_2nd_128_211117/tsr_v3_20k_dense32_46_500p_1e-3_0p001"

    def get_model(self, sublinear=False):

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        if "model" not in self.__dict__:
            from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
            in_channels = [256, 512, 1024]
            # NANO model use depthwise = True, which is main difference.
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, depthwise=True)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, depthwise=True)
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
                data_dir=os.path.join(get_yolox_datadir(), "zone_tsr"),
                # image_sets=[('zone_tsr_generate01', 'train')],
                image_sets=[('zone_tsr_v3_20211111_20k', 'train')],
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=50,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob,
                    bgr_means=(0.406, 0.456, 0.485),
                    std=(0.225, 0.224, 0.229)),
                cache=cache_img,
            )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,
                bgr_means=(0.406, 0.456, 0.485),
                std=(0.225, 0.224, 0.229)),
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
        from yolox.data import TSR_ZO_Detection_Two

        # valdataset = TSR_ZO_Detection(
        valdataset = TSR_ZO_Detection_Two(
            data_dir=os.path.join(get_yolox_datadir(), "zone_tsr"),
            # image_sets=[('zone_tsr_generate01', 'test')],
            image_sets=[('zone_tsr_v3_20211111_20k', 'test')],
            img_size=self.test_size,
            preproc=ValTransform(
                bgr_means=(0.406, 0.456, 0.485),
                std=(0.225, 0.224, 0.229)),
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
        from yolox.evaluators import TSR_ZO_Evaluator_Two

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        # evaluator = TSR_ZO_Evaluator(
        evaluator = TSR_ZO_Evaluator_Two(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            # num_classes=self.num_classes,
            num_classes=len(TSR_ZO_CLASSES_45)
        )
        return evaluator
