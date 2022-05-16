#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
from re import I
from loguru import logger

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
# import logging
from PIL import Image
import cv2
import tqdm
import copy
from yolox.exp import get_exp
from yolox.models.network_blocks import SiLU
from yolox.utils import replace_module, setup_logger
# from yolox.utils.boxes import postprocess


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
        default='./exps/example/seg/seg_exp_quant.py',
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


# ------------------------------- datasets -------------------------------

def resize_label(label, size, num_class, interp_type):
    seg_labels = np.eye(num_class)[label.reshape([-1])]
    one_hot_labels = seg_labels.reshape(int(label.shape[0]), int(label.shape[1]), num_class)
    de_onehot = []

    for i in range(one_hot_labels.shape[2]):
        one_label = cv2.resize(one_hot_labels[:, :, i], size, interpolation=interp_type)
        one_label[one_label != 0] = i
        de_onehot.append(one_label)
    
    de_onehot = np.array(de_onehot)
    area_label = np.argmax(de_onehot, axis=0)
    return area_label


class Resize_smap(object):
    def __init__(self, input_size, out_size, num_class=3) -> None:
        self.in_size = input_size
        self.out_size = out_size
        self.num_class = num_class

    def __call__(self, im_lb):
        image, label = im_lb['im'], im_lb['lb']
        if self.in_size != self.out_size:
            label = resize_label(label, self.out_size, self.num_class, interp_type=cv2.INTER_AREA)
        if self.in_size != image.shape[:-1][::-1]:
            image = cv2.resize(image, self.in_size)
        seg_labels = np.eye(self.num_class)[label.reshape([-1])]
        one_hot_labels = seg_labels.reshape(int(label.shape[0]), int(label.shape[1]), self.num_class)
        one_hot_labels = one_hot_labels.transpose(2, 0, 1)
        return dict(im=image, lb=label, oh=one_hot_labels)


class ToTensor(object):
    def __init__(self, mean=(0,0,0), std=(1.,1.,1.)) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, im_lb):
        im, lb, oh = im_lb['im'], im_lb['lb'], im_lb['oh']
        im = im.transpose(2, 0, 1).astype(np.float32)
        im = torch.from_numpy(im).div_(255)
        dtype, device = im.dtype, im.device
        mean = torch.as_tensor(self.mean, dtype=dtype, device=device)[:, None, None]
        std = torch.as_tensor(self.std, dtype=dtype, device=device)[:, None, None]
        im = im.sub_(mean).div_(std).clone()
        lb = torch.from_numpy(lb.astype(np.int64).copy()).clone()
        if oh is not None:
            oh = torch.from_numpy(oh.astype(np.int64).copy()).clone()
        else:
            oh = lb.clone()
        return dict(im=im, lb=lb, oh=oh)

class Lane_Dataset(Dataset):
    def __init__(self, data_path, img_lbl_txt, num_class, input_size, crop_size, trans_func=None, mode='train') -> None:
        super(Lane_Dataset, self).__init__()
        # logger = logging.getLogger("data_parser")
        self.mode = mode
        self.trans_func = trans_func
        self.n_classes = num_class
        self.root = os.getcwd()
        self._image_path, self._anno_path = self._load_items(data_path, img_lbl_txt)
        self.resize_ = input_size
        self.crop_size = crop_size
        self.to_tensor = ToTensor(
            mean=(0.3257, 0.3690, 0.3223),
            std=(0.2112, 0.2148, 0.2115),
        )

    def __len__(self):
        return len(self._image_path)

    def _load_items(self, data_dir, img_lbl_txt):
        anno_paths = []
        image_paths = []
        for (i, img_lbl) in enumerate(img_lbl_txt):
            with open(img_lbl, 'r') as f:
                for line in f.readlines():
                    line = line.strip('\n').split(' ')
                    image_path = os.path.join(data_dir[i], line[0])
                    anno_path = os.path.join(data_dir[i], line[1])

                    if os.path.exists(image_path):
                        image_paths.append(image_path)
                    else:
                        logger.info('Lost: %s ' % image_path)

                    if os.path.exists(anno_path):
                        anno_paths.append(anno_path)
                    else:
                        logger.info('Lost: %s ' % anno_path)
        return image_paths, anno_paths

    def __getitem__(self, index):
        image_path = self._image_path[index]
        label_path = self._anno_path[index]
        basename = image_path.split('/')[-1]
        img_type = basename.split('/')[-1]
        
        try:
            img = Image.open(image_path)
        except:
            logger.info('path error: %s' % image_path)
            exit()
        try:
            label = Image.open(label_path).convert('L')
        except:
            logger.info('path error: %s' % label_path)
            exit()
        
        img = np.array(img)[:, :, ::-1]
        label = np.array(label)
        label[label == 255] = self.n_classes
        h, w, _ = img.shape

        img = img[self.crop_size[1]:self.crop_size[3], self.crop_size[0]:self.crop_size[2]]
        label = label[self.crop_size[1]:self.crop_size[3], self.crop_size[0]:self.crop_size[2]]
        # resize img & label

        im_lb = dict(im=img, lb=label, oh=None)
        if self.trans_func is not None:
            im_lb = self.trans_func(im_lb)
        im_lb = self.to_tensor(im_lb)
        img, label, one_hot = im_lb['im'], im_lb['lb'], im_lb['oh']
        return img.detach(), label.unsqueeze(0).detach(), one_hot.detach()


def get_data_loader(data_path, list_txt, ims_per_gpu, num_works, num_class, input_size, output_size, crop_size, padding_path, mode='train'):
    if mode == 'train':
        trans_func = Resize_smap(input_size, output_size, num_class)
        batch_size = ims_per_gpu
        shuffle = True
        drop_last = True
    elif mode == 'test':
        trans_func = Resize_smap(input_size, output_size, num_class)
        batch_size = ims_per_gpu
        shuffle = False
        drop_last = False
    
    ds = Lane_Dataset(data_path, list_txt, num_class, input_size, crop_size, trans_func, mode)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_works, pin_memory=True,)
    return dl

# ------------------------------- datasets -------------------------------


# ------------------------------- loss -------------------------------

class OhemCELoss(nn.Module):
    def __init__(self, thresh, num_classes=3) -> None:
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).cuda()
        self.ignore_lb = num_classes
        self.criteria = nn.CrossEntropyLoss(ignore_index=num_classes, reduction='none')
    
    def forward(self, logits, labels):
        n_min = labels[labels != self.ignore_lb].numel() // 5
        loss = self.criteria(logits, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)


def loss_calculate(loss_type, class_num, pred, aux_1, aux_2, truth_lb, truth_oh):
    if loss_type == 'ohem':
        criteria_pre = OhemCELoss(0.7, class_num)
        criteria_aux = OhemCELoss(0.7, class_num)
        loss_pre = criteria_pre(pred, truth_lb)
        loss_aux_1 = criteria_aux(aux_1, truth_lb)
        loss_aux_2 = criteria_aux(aux_2, truth_lb)
    return loss_pre, loss_aux_1, loss_aux_2

# ------------------------------- loss -------------------------------

def post_process(output_raw, output_save_path='test.jpg', index=0):
    input_tensor = output_raw[index]
    input_np = input_tensor.cpu().detach().numpy()
    # output_data = np.array(input_np)
    output_data = np.argmax(input_np, 0)
    # print(output_data.shape)
    image_output = output_data*100
    cv2.imwrite(output_save_path, image_output)
    return


def poly_lr_scheduler(optimizer, init_lr, iter, max_iter=200, power=0.9):
    lr = init_lr * (1 - iter/max_iter) ** power
    optimizer.param_groups[0]['lr'] = lr
    return lr


def quantize_model_process(model, data_loader, num_pic=2):
    from pytorch_quantization import nn as quant_nn
    from pytorch_quantization import calib

    model.cuda()
    with torch.no_grad():
        # Enable calibrators
        for _, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.disable_quant()
                    module.enable_calib()
                else:
                    module.disable()

        data_tmp = tqdm.tqdm(data_loader, total=num_pic)
        for i, (image, _, _) in enumerate(tqdm.tqdm(data_loader, total=num_pic)):
            model(image.cuda())
            if i >= num_pic:
                break

        # Disable calibrators
        for _, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.enable_quant()
                    module.disable_calib()
                else:
                    module.enable()

        # Load calib result
        for _, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    if isinstance(module._calibrator, calib.MaxCalibrator):
                        module.load_calib_amax(strict=False)
                    else:
                        module.load_calib_amax(strict=False, method="percentile", percentile=99.99)

        # collect_stats(model, data_loader, num_batches=num_pics)
        # compute_amax(model, method="percentile", percentile=99.99)
    torch.save({'epoch': 1, 'model_state_dict': model.state_dict(), },
               '/home/zjw/workspace/AI/perception/YOLOX/models/lane/debug_finetune/lane_ep148_finetune_ptq.pth')
    return model


def save_onnx(model, onnx_name, batch_size=1, size=[1024, 1920]):
    from pytorch_quantization import nn as quant_nn
    dummy_input = torch.rand(batch_size, 3, size[0], size[1])
    dummy_input = dummy_input.to(torch.float32)
    quant_nn.TensorQuantizer.use_fb_fake_quant = True
    torch.onnx.export(
        model,
        dummy_input.cuda(),
        onnx_name,
        input_names=['images'],
        output_names=['output'],
        dynamic_axes=None,
        opset_version=13,
    )


def lane_train(cfgs, model, optimizer, lane_train_data, lane_test_data, epoch, lr):
    max_miou = 0.5
    step = 0

    tq = tqdm.tqdm(lane_train_data)
    tq.set_description('epoch%d/lr=%3f' % (epoch+1, lr))
    loss_record = []
    hist = torch.zeros(cfgs['num_class'], cfgs['num_class']).cuda().detach()
    iters = len(tq)

    for i, (im, lb, oh) in enumerate(tq):
        if torch.cuda.is_available() and cfgs['use_gpu']:
            im = im.cuda()
            lb = lb.cuda()
            oh = oh.cuda()
        lb = torch.squeeze(lb, 1)
        optimizer.zero_grad()
        lane_head_feat, lane_seg_aux_head_layer1, lane_seg_aux_head_layer2, \
                roadway_head_feat, roadway_seg_aux_head_layer1, roadway_seg_aux_head_layer2 = model(im)
        
        train_preds = torch.argmax(lane_head_feat, dim=1)
        keep = lb < cfgs['num_class']
        hist += torch.bincount(
            lb[keep] * cfgs['num_class'] + train_preds[keep],
            minlength=cfgs['num_class']**2
        ).view(cfgs['num_class'], cfgs['num_class'])

        loss_pre, loss_aux_1, loss_aux_2 = loss_calculate(
            cfgs['loss_type'], cfgs['num_class'], pred=roadway_head_feat, 
            aux_1=lane_seg_aux_head_layer1, aux_2=lane_seg_aux_head_layer2, truth_lb=lb, truth_oh=oh)
        if cfgs['model_frozen'] is False:
            # loss = loss_pre + 0.5*loss_aux_2 + 0.3*loss_aux_1
            loss = loss_pre
        else:
            loss = loss_pre
        
        loss.backward()
        optimizer.step()
        step += 1
        loss_record.append(loss.item())
        if step > iters:
            break

        post_process(lane_head_feat)
    
    tq.close()
    loss_train_mean = np.mean(loss_record)
    train_ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
    train_ious = train_ious.cpu().numpy().tolist()
    iou_list = [train_ious[i] for i in range(cfgs['num_class'])]
    dash_line = '-'*60
    logger.info(dash_line)
    logger.info('epoch%d/loss for train : mean_loss=%4f' % (epoch, loss_train_mean))
    logger.info('lane iou for train :')
    logger.info(iou_list)

    if (epoch + 1) % cfgs['save_interval'] == 0:
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),},
                   os.path.join(cfgs['out_dir'], 'epoch%03d.pth' % (epoch+1)))
                   

@logger.catch
def main():
    args = make_parser().parse_args()
    logger.info("args value: {}".format(args))
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    setup_logger(
        os.path.join(exp.output_dir, args.experiment_name),
        distributed_rank=0,
        filename="train_log.txt",
        mode="a",
    )

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

    model = exp.get_model(train_flag=True)
    if args.ckpt is not None:
        # file_name = os.path.join(exp.output_dir, args.experiment_name)
        # ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        ckpt_file = args.ckpt

        # load the model state dict
        ckpt = torch.load(ckpt_file)

        model.eval()
        if "model" in ckpt:
            ckpt = ckpt["model"]
        elif 'model_state_dict' in ckpt:
            ckpt = ckpt['model_state_dict']
        model.load_state_dict(ckpt)
    logger.info("\n{}\n".format(model))
    model.cuda()

    # -------- params --------
    init_lr = 1e-3   # TODO
    start_epoch = 0
    max_epoch = 1
    optimizer = torch.optim.SGD(model.parameters(), init_lr, momentum=0.9, weight_decay=1e-4)

    data_path = ['/home/zjw/workspace/AI/perception/dataset/lane_dataset_org/']
    list_txt = ['/home/zjw/workspace/AI/perception/dataset/lane_dataset_org/train.txt']
    ims_per_gpu = 2     # batch_size, must > 1
    num_works = 1
    num_class = 3
    input_size = (1920, 1024)
    output_size = (480, 256)
    crop_size = (0, 56, 1920, 1080)
    padding_path = ''
    mode = 'train'
    cfgs = {'num_class': num_class, 'use_gpu': True, 'loss_type': 'ohem', 
            'model_frozen': False, 'save_interval': 10, 'out_dir': os.path.join(exp.output_dir, args.experiment_name)}
    lane_train_data = get_data_loader(data_path, list_txt, ims_per_gpu, num_works, num_class, input_size, output_size, crop_size, padding_path, mode)
    lane_test_data = get_data_loader(data_path, list_txt, ims_per_gpu, num_works, num_class, input_size, output_size, crop_size, padding_path, mode)

    model = quantize_model_process(model, lane_train_data, num_pic=500)

    for epoch in range(start_epoch, max_epoch):
        model.train()
        lr = poly_lr_scheduler(optimizer, init_lr, iter=epoch, max_iter=max_epoch)
        if epoch % 2 == 0:
            lane_train(cfgs, model, optimizer, lane_train_data, lane_test_data, epoch, lr)
        else:
            # seg_train(cfgs, model, optimizer, lane_train_data, lane_test_data, epoch, lr)
            lane_train(cfgs, model, optimizer, lane_train_data, lane_test_data, epoch, lr)

    # onnx_path = os.path.join('/home/zjw/workspace/AI/perception/YOLOX/YOLOX_outputs/seg_test_20220505/seg_test_20220505_finetune',
    #                          'lane_ep148_finetune.onnx')
    # save_onnx(model, onnx_path, batch_size=1, size=[input_size[1], input_size[0]])


if __name__ == "__main__":
    params = {}
    main()
