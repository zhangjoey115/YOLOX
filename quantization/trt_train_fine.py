import datetime
import os
import sys
import time

import torch
import torch.utils.data
from torch import nn

import torchvision
from torchvision import transforms

from pytorch_quantization import calib
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules


import argparse
from yolox.core import TrainerQuant
from yolox.exp import get_exp
from loguru import logger
from tqdm import tqdm


def prepare_model(exp, args, data_loader, num_pics=5):
    quant_modules.initialize()
    model = exp.get_model()

    if args.ckpt is None:
        file_name = os.path.join(exp.output_dir, args.experiment_name)
        ckpt_file = os.path.join(file_name, "best_ckpt.pth")
    else:
        ckpt_file = args.ckpt

    # load the model state dict
    ckpt = torch.load(ckpt_file, map_location="cuda:0")
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    model.eval()
    model.cuda()

    with torch.no_grad():
        # Enable calibrators
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.disable_quant()
                    module.enable_calib()
                else:
                    module.disable()

        for i, (image, _, _, _) in tqdm(enumerate(data_loader), total=num_pics):
            model(image.cuda())
            if i >= num_pics:
                break

        # Disable calibrators
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.enable_quant()
                    module.disable_calib()
                else:
                    module.enable()
        
            # Load calib result
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.load_calib_amax(strict=False)

    print("Finish prepare model")
    print("ckpt state_dict = ")
    print(ckpt)

    print("model init state_dict = ")
    state_dict = model.state_dict()
    print(model.state_dict())
    return model


def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
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
    parser.add_argument("-b", "--batch-size", type=int, default=2, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="plz input your expriment description file",
    )
    parser.add_argument(
        "--resume", default=False, action="store_true", help="resume training"
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument(
        "-e",
        "--start_epoch",
        default=None,
        type=int,
        help="resume training start epoch",
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision training.",
    )
    parser.add_argument(
        "--cache",
        dest="cache",
        default=False,
        action="store_true",
        help="Caching imgs to RAM for fast training.",
    )
    parser.add_argument(
        "--output-name", type=str, default="yolox.onnx", help="output name of models"
    )
    parser.add_argument(
        "--input", default="images", type=str, help="input node name of onnx model"
    )
    parser.add_argument(
        "--output", default="output", type=str, help="output node name of onnx model"
    )
    parser.add_argument(
        "-o", "--opset", default=13, type=int, help="onnx opset version"
    )
    parser.add_argument(
        "--dynamic", action="store_true", help="whether the input shape should be dynamic or not"
    )
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    batch_size = 1
    is_distributed = False
    data_loader = exp.get_data_loader(
            batch_size=batch_size,
            is_distributed=is_distributed,
            no_aug=True
        )
    model = prepare_model(exp, args, data_loader)

    # trainer = TrainerQuant(exp, args)
    # trainer.train()

    if "head" in model.__dict__['_modules']:
        model.head.decode_in_inference = False
    model.cuda()

    logger.info("loading checkpoint done.")
    dummy_input = torch.rand(args.batch_size, 3, exp.test_size[0], exp.test_size[1])

    dummy_input = dummy_input.to(torch.float32)
    quant_nn.TensorQuantizer.use_fb_fake_quant = True
    torch.onnx.export(
        model,
        dummy_input.cuda(),
        args.output_name,
        input_names=[args.input],
        output_names=[args.output],
        dynamic_axes={args.input: {0: 'batch'},
                      args.output: {0: 'batch'}} if args.dynamic else None,
        # verbose=True, 
        opset_version=args.opset,
    )


    # file_name = os.path.join(file_dir, 'test_fine_t1.pth')
    # torch.save(model.state_dict(), file_name)
