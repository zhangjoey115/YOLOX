#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
from loguru import logger

import torch
from torch import nn

from yolox.exp import get_exp
from yolox.models.network_blocks import SiLU
from yolox.utils import replace_module


def make_parser():
    parser = argparse.ArgumentParser("YOLOX onnx deploy")
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
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument(
        "--dynamic", action="store_true", help="whether the input shape should be dynamic or not"
    )
    parser.add_argument("--no-onnxsim", action="store_true", help="use onnxsim or not")
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="expriment description file",
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    return parser


def read_img_input(exp):
    import cv2
    from yolox.data.data_augment import ValTransform

    img_path = '/home/zjw/workspace/AI/tools/TensorRT_test/img_test/2_640.jpg'
    preprocess = ValTransform()
    img = cv2.imread(img_path)
    img, _ = preprocess(img, None, exp.test_size)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.cuda()

    return img


def read_img_input_dummy(exp):
    img = torch.full((1, 3, exp.test_size[0], exp.test_size[1]), 255)
    return img


@logger.catch
def main():
    args = make_parser().parse_args()
    logger.info("args value: {}".format(args))
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

   # ------------- Quantization -------------
    from pytorch_quantization import nn as quant_nn
    # from pytorch_quantization import calib
    # from pytorch_quantization.tensor_quant import QuantDescriptor
    from pytorch_quantization import quant_modules

    quant_modules.initialize()
    #
    # quant_desc_input = QuantDescriptor(calib_method='histogram')
    # quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    # quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
    # quant_nn.QuantConvTranspose2d.set_default_quant_desc_input(quant_desc_input)
   # ------------- Quantization -------------

    from quantization.tensorrt_examples.torchvision.models.classification import resnet18
    model = resnet18(pretrained=False, quantize=True)
    model.eval()
    model.cuda()

    logger.info("loading checkpoint done.")
    # dummy_input = torch.randn(args.batch_size, 3, exp.test_size[0], exp.test_size[1])
    dummy_input = torch.rand(args.batch_size, 3, 256, 256) * 255
    # dummy_input = read_img_input(exp)

   # ------------- Quantization -------------
    dummy_input = dummy_input.to(torch.float32)

    quant_nn.TensorQuantizer.use_fb_fake_quant = True
   # ------------- Quantization -------------

    # torch.onnx.export(
    #     model, dummy_input, "test.onnx", verbose=True, opset_version=13, enable_onnx_checker=False)

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
    logger.info("generated onnx model named {}".format(args.output_name))

    if not args.no_onnxsim:
        import onnx

        from onnxsim import simplify

        input_shapes = {args.input: list(dummy_input.shape)} if args.dynamic else None
        
        # use onnxsimplify to reduce reduent model.
        onnx_model = onnx.load(args.output_name)
        model_simp, check = simplify(onnx_model,
                                     dynamic_input_shape=args.dynamic,
                                     input_shapes=input_shapes)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, args.output_name)
        logger.info("generated simplified onnx model named {}".format(args.output_name))


if __name__ == "__main__":
    main()
