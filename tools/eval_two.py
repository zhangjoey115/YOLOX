#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import random
import warnings
from loguru import logger

import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

from yolox.core import launch
from yolox.exp import get_exp
from yolox.utils import configure_nccl, fuse_model, get_local_rank, get_model_info, setup_logger


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--test",
        dest="test",
        default=False,
        action="store_true",
        help="Evaluating on test-dev set.",
    )
    parser.add_argument(
        "--speed",
        dest="speed",
        default=False,
        action="store_true",
        help="speed test only.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def model_fetcher(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    # if args.trt:
    #     args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    # if args.device == "gpu":
    model.cuda()
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)
    return model


@logger.catch
def main(exp, exp_2, model_list, args, num_gpu):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed testing. This will turn on the CUDNN deterministic setting, "
        )

    is_distributed = num_gpu > 1

    # set environment variables for distributed training
    configure_nccl()
    cudnn.benchmark = True

    rank = get_local_rank()

    file_name = os.path.join(exp.output_dir, args.experiment_name)

    if rank == 0:
        os.makedirs(file_name, exist_ok=True)

    setup_logger(file_name, distributed_rank=rank, filename="val_log.txt", mode="a")
    logger.info("Args: {}".format(args))

    evaluator = exp.get_evaluator(args.batch_size, is_distributed, args.test, args.legacy)

    trt_file = None
    decoder = None
    # start evaluate
    *_, summary = evaluator.evaluate(
        model_list, is_distributed, args.fp16, trt_file, decoder, exp.test_size
    )
    logger.info("\n" + summary)


if __name__ == "__main__":
    """ cmd = python tools/eval_two.py -d 1 -b 12 """
    
    args = make_parser().parse_args()
    # exp = get_exp(args.exp_file, args.name)
    # exp.merge(args.opts)

    exp_file_1 = "/home/zjw/workspace/DL_Vision/TSR/YOLOX/exps/example/tsr/yolox_tsr_zo_nano_eval_two.py"
    exp_file_2 = "/home/zjw/workspace/DL_Vision/TSR/YOLOX/exps/example/tsr/tsr_2nd_exp_dense.py"
    # checkpoint_1 = "/home/zjw/workspace/DL_Vision/TSR/YOLOX/YOLOX_outputs/train_tsr_zo_960_211116_v01_03/yolox_tsr_zo_head2_320norm_300_1e-3_0p005/best_ckpt.pth"
    checkpoint_1 = "/home/zjw/workspace/DL_Vision/TSR/YOLOX/YOLOX_outputs/train_tsr_zo_960_211117_v3/yolox_tsr_zo_h2_3_960p_200_1e-4_0p05/best_ckpt.pth"
    checkpoint_2 = "/home/zjw/workspace/DL_Vision/TSR/YOLOX/YOLOX_outputs/train_tsr_2nd_128_211117/tsr_v3_20k_dense32_46_500p_1e-3_0p001/best_ckpt.pth"
    exp_1 = get_exp(exp_file_1, args.name)
    exp_2 = get_exp(exp_file_2, args.name)

    model_list = []
    args.ckpt = checkpoint_1
    model_list.append(model_fetcher(exp_1, args))
    args.ckpt = checkpoint_2
    model_list.append(model_fetcher(exp_2, args))

    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()

    dist_url = "auto" if args.dist_url is None else args.dist_url
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp_1, exp_2, model_list, args, num_gpu),
    )
