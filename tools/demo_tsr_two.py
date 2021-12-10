#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger

import cv2
import numpy as np

import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
# from yolox.data.datasets import TT100K_CLASSES as COCO_CLASSES      # zjw
from yolox.data.datasets import TSR_2ND_CLASSES as COCO_CLASSES      # zjw
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="./assets/dog.jpg", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
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
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
        self,
        model_list,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        legacy=False,
    ):
        self.model_list = model_list
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.preproc = ValTransform(bgr_means=(0.406, 0.456, 0.485),
                                    std=(0.225, 0.224, 0.229))
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None
        logger.info("Processing image: {}".format(img_info["file_name"]))

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        if self.device == "gpu":
            img = img.cuda()

        with torch.no_grad():
            # 1st stage tsr model
            t0 = time.time()
            output_1st = self.model_list[0](img)
            logger.info("Infer time1: {:.4f}s".format(time.time() - t0))
            if self.decoder is not None:
                output_1st = self.decoder(output_1st, dtype=outputs.type())
            output_1st = postprocess(
                output_1st, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            # logger.info("output_1st len = {}, type={}".format(len(output_1st), output_1st[0]))
            if output_1st[0] is None:
                return output_1st, img_info

            # 2nd stage tsr model
            input_2nd = process_1st_output(output_1st, img_info["raw_img"], img_info["ratio"], self.preproc)
            t0 = time.time()
            outputs_2nd = self.model_list[1](input_2nd)
            logger.info("Infer time2: {:.4f}s".format(time.time() - t0))
            outputs_2nd = self.model_list[1].decoding(outputs_2nd)      # zjw
            if self.decoder is not None:
                outputs_2nd = self.decoder(outputs_2nd, dtype=outputs_2nd.type())
            # outputs_2nd = postprocess(
            #     outputs_2nd, self.num_classes, self.confthre,
            #     self.nmsthre, class_agnostic=True
            # )
            outputs = process_result_combination(output_1st, outputs_2nd)

        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res


def image_demo(predictor, vis_folder, path, current_time, save_result):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    for image_name in files:
        outputs, img_info = predictor.inference(image_name)
        result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if save_result:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)
        # ch = cv2.waitKey(0)
        # if ch == 27 or ch == ord("q") or ch == ord("Q"):
        #     break


def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    save_folder = os.path.join(
        vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = os.path.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = os.path.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame)
            result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
            if args.save_result:
                vid_writer.write(result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break


def process_1st_output(output_1st, img, ratio, preproc=None):
    """ 
    output_1st: [tensor(n,7)] : bbox, obj_conf, class_conf, class_pred
    input_2nd:  (10, 3, 128, 128)
    """
    max_batch_num = 100
    num = output_1st[0].shape[0]
    boxes = output_1st[0][:, :4].cpu().numpy() / ratio
    boxes = boxes.astype(np.int32)
    input_2nd = []
    for i in range(max_batch_num):
        if i < num:
            box = [max(x,0) for x in boxes[i, :]]
            img_crop = img[box[1]:box[3], box[0]:box[2]]
            # img_resize = cv2.resize(img_crop, (128, 128))
            img_resize = cv2.resize(img_crop, (64, 64))
            # cv2.imshow('demo', img_resize)
            # input_2nd.append(img_resize.transpose((2,0,1)))
            # img_resize, _ = preproc(img_resize, None, (128, 128))
            img_resize, _ = preproc(img_resize, None, (64, 64))
            input_2nd.append(img_resize)
        else:
            input_2nd.append(input_2nd[-1])
    input_2nd = [torch.from_numpy(x) for x in input_2nd]
    input_2nd = torch.stack(input_2nd, 0)
    input_2nd = input_2nd.to(device=output_1st[0].device, dtype=output_1st[0].dtype)
    return input_2nd


def process_result_combination(output_1st, outputs_2nd):
    """ 
    output_1st: [tensor(n,7)] : bbox, obj_conf, class_conf, class_pred
    input_2nd:  tensor(10, 2): score, class
    output: 
    """
    output = output_1st
    num = output_1st[0].shape[0]
    for i in range(num):
        output[0][i, 5] = outputs_2nd[i, 0]
        output[0][i, 6] = outputs_2nd[i, 1]
    return output


def model_fetcher(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
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


def main(model_list, exp_1, args):
    # if args.trt:
    #     assert not args.fuse, "TensorRT model is not support model fusing!"
    #     trt_file = os.path.join(file_name, "model_trt.pth")
    #     assert os.path.exists(
    #         trt_file
    #     ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
    #     model.head.decode_in_inference = False
    #     decoder = model.head.decode_outputs
    #     logger.info("Using TensorRT to inference")
    # else:
    trt_file = None
    decoder = None

    predictor = Predictor(model_list, exp_1, COCO_CLASSES, trt_file, decoder, args.device, args.legacy)
    current_time = time.localtime()
    vis_folder = os.path.join(os.path.join(exp_1.output_dir, exp_1.exp_name), 'vis_res')
    if args.demo == "image":
        image_demo(predictor, vis_folder, args.path, current_time, args.save_result)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    """ cmd = python tools/demo_tsr_two.py image --conf 0.3 --nms 0.45 --save_result --device gpu --path xx.jpg """
    
    args = make_parser().parse_args()
    exp_file_1 = "/home/zjw/workspace/DL_Vision/TSR/YOLOX/exps/example/tsr/yolox_tsr_zo_nano.py"
    exp_file_2 = "/home/zjw/workspace/DL_Vision/TSR/YOLOX/exps/example/tsr/tsr_2nd_exp_dense.py"
    # checkpoint_1 = "/home/zjw/workspace/DL_Vision/TSR/YOLOX/YOLOX_outputs/train_tsr_zo_768_211019/yolox_tsr_zo_nano5_320norm_p_om_300_1e-4/best_ckpt.pth"
    # checkpoint_1 = "/home/zjw/workspace/DL_Vision/TSR/YOLOX/YOLOX_outputs/train_tsr_zo_960_211111/yolox_tsr_zo_head2_320norm_p_om_300_1e-3_0p005/best_ckpt.pth"
    # checkpoint_1 = "/home/zjw/workspace/DL_Vision/TSR/YOLOX/YOLOX_outputs/train_tsr_zo_960_211116_v01_03/yolox_tsr_zo_head2_320norm_300_1e-3_0p005/best_ckpt.pth"
    # checkpoint_2 = "/home/zjw/workspace/DL_Vision/TSR/YOLOX/YOLOX_outputs/train_tsr_2nd_128_211111/tsr_v01v02_dense16L1_67_400_1e-3_0p005/best_ckpt.pth"
    # checkpoint_2 = "/home/zjw/workspace/DL_Vision/TSR/YOLOX/YOLOX_outputs/train_tsr_2nd_128_211111/tsr_v01v02_dense32_67_600_1e-3_0p005/best_ckpt.pth"
    # checkpoint_1 = "/home/zjw/workspace/DL_Vision/TSR/YOLOX/YOLOX_outputs/train_tsr_zo_960_211117_v3/yolox_tsr_zo_head2_960p_200_1e-4_0p05/best_ckpt.pth"
    checkpoint_1 = "/home/zjw/workspace/DL_Vision/TSR/YOLOX/YOLOX_outputs/train_tsr_zo_960_211117_v3/yolox_tsr_zo_h2_3_960p_200_1e-4_0p05/best_ckpt.pth"
    # checkpoint_2 = "/home/zjw/workspace/DL_Vision/TSR/YOLOX/YOLOX_outputs/train_tsr_2nd_128_211117/tsr_v3_20k_dense32_46_300pwttk_1e-3_0p001/best_ckpt.pth"
    checkpoint_2 = "/home/zjw/workspace/DL_Vision/TSR/YOLOX/YOLOX_outputs/train_tsr_2nd_128_211206/tsr_v3_20k_aug_dense32_46_i64_600_1e-3_0p001/best_ckpt.pth"
    exp_1 = get_exp(exp_file_1, args.name)
    exp_2 = get_exp(exp_file_2, args.name)

    model_list = []
    args.ckpt = checkpoint_1
    model_list.append(model_fetcher(exp_1, args))
    args.ckpt = checkpoint_2
    model_list.append(model_fetcher(exp_2, args))

    main(model_list, exp_1, args)
