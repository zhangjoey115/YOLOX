import datetime
import os
import sys
import time

import torch
import torch.utils.data
from torch import nn

from tqdm import tqdm

import torchvision
from torchvision import transforms

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor

import numpy as np
import argparse
from yolox.exp import get_exp
from loguru import logger


def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument(
        "-f",
        "--exp_file",
        default="./exps/example/tsr/yolox_tt100k_nano_new.py",
        type=str,
        help="plz input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    return parser


def input_process(input_tensor):
    mean = np.array([[[0.3257, 0.3690, 0.3223]]])
    std = np.array([[[0.2112, 0.2148, 0.2115]]])

    image = input_tensor.cpu().numpy().squeeze(0)
    image = image.transpose((1, 2, 0))
    image = image.astype(np.float32) / 255.0
    image = (image - mean) / std
    
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :, :, :]
    image = np.array(image, dtype=np.float32)
    image = torch.from_numpy(image)

    return image


def collect_stats(model, data_loader, num_batches):
    """Feed data to the network and collect statistic"""

    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    dev = next(model.parameters()).device
    for i, (image, _, _, _) in tqdm(enumerate(data_loader), total=num_batches):
        # image_g = image.cuda()
        # if image_g is None:
        #     image_g = image.to(dev)
        image = input_process(image)
        model(image.cuda())
        if i >= num_batches:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()


def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax(strict=False)
                else:
                    module.load_calib_amax(strict=False, **kwargs)
                print("Calibrated {}".format(name))
    model.cuda()


def quantize_model(exp, ckpt_file, data_loader, num_pics=2):
    quant_desc_input = QuantDescriptor(calib_method='histogram')
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

    from pytorch_quantization import quant_modules
    quant_modules.initialize()

    model = exp.get_model()
    ckpt = torch.load(ckpt_file)
    if "model" in ckpt:
        ckpt = ckpt["model"]
    elif 'model_state_dict' in ckpt:
        ckpt = ckpt['model_state_dict']
    model.load_state_dict(ckpt)
    if "head" in model.__dict__['_modules']:
        model.head.decode_in_inference = False

    # model = torchvision.models.resnet50(pretrained=True, progress=False)
    model.eval()
    model.cuda()
    with torch.no_grad():
        collect_stats(model, data_loader, num_batches=num_pics)
        compute_amax(model, method="percentile", percentile=99.99)
        # compute_amax(model, method="percentile", percentile=99.9)
        # compute_amax(model, method="entropy")

    return model


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    batch_size = 1
    is_distributed = False
    data_loader = exp.get_data_loader(
            batch_size=batch_size,
            is_distributed=is_distributed,
            no_aug=True
        )
    # data_loader = exp.get_eval_loader(
    #         batch_size=batch_size,
    #         is_distributed=is_distributed
    #     )

    file_dir = os.path.join(exp.output_dir, exp.exp_name)
    if args.ckpt is None:
        ckpt_file = os.path.join(file_dir, "best_ckpt.pth")
    else:
        ckpt_file = args.ckpt

    model = quantize_model(exp, ckpt_file, data_loader, num_pics=200)

    # model.head.decode_in_inference = True
    # batch_size = 8
    # evaluator = exp.get_evaluator(
    #     batch_size=batch_size, is_distributed=is_distributed
    #     )
    # ap50_95, ap50, summary = exp.eval(
    #     model, evaluator, is_distributed
    # )
    # logger.info("val/COCOAP50: {:.4f}, val/COCOAP50_95: {:.4f}".format(ap50, ap50_95))
    # logger.info("\n" + summary)

    file_name = os.path.join(file_dir, 'lane_ptq_t200_p9999.pth')
    torch.save(model.state_dict(), file_name)


# # from absl import logging
# # logging.set_verbosity(logging.FATAL)  # Disable logging as they are too noisy in notebook

# # For simplicity, import train and eval functions from the train script from torchvision instead of copything them here
# # Download torchvision from https://github.com/pytorch/vision
# # sys.path.append("/raid/skyw/models/torchvision/references/classification/")
# from train import evaluate, train_one_epoch, load_data

# quant_desc_input = QuantDescriptor(calib_method='histogram')
# quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
# quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)


# from pytorch_quantization import quant_modules
# quant_modules.initialize()

# model = torchvision.models.resnet50(pretrained=True, progress=False)
# model.cuda()


# data_path = "/raid/data/imagenet/imagenet_pytorch"
# batch_size = 512

# traindir = os.path.join(data_path, 'train')
# valdir = os.path.join(data_path, 'val')
# dataset, dataset_test, train_sampler, test_sampler = load_data(traindir, valdir, False, False)


# data_loader = torch.utils.data.DataLoader(
#     dataset, batch_size=batch_size,
#     sampler=train_sampler, num_workers=4, pin_memory=True)

# data_loader_test = torch.utils.data.DataLoader(
#     dataset_test, batch_size=batch_size,
#     sampler=test_sampler, num_workers=4, pin_memory=True)




# # It is a bit slow since we collect histograms on CPU
# with torch.no_grad():
#     collect_stats(model, data_loader, num_batches=2)
#     compute_amax(model, method="percentile", percentile=99.99)


# criterion = nn.CrossEntropyLoss()
# with torch.no_grad():
#     evaluate(model, criterion, data_loader_test, device="cuda", print_freq=20)
    
# # Save the model
# torch.save(model.state_dict(), "/tmp/quant_resnet50-calibrated.pth")

# # other calibration method
# with torch.no_grad():
#     compute_amax(model, method="percentile", percentile=99.9)
#     evaluate(model, criterion, data_loader_test, device="cuda", print_freq=20)


# with torch.no_grad():
#     for method in ["mse", "entropy"]:
#         print(F"{method} calibration")
#         compute_amax(model, method=method)
#         evaluate(model, criterion, data_loader_test, device="cuda", print_freq=20)