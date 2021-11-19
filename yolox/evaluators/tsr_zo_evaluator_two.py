#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import sys
import tempfile
import time
from collections import ChainMap
from loguru import logger
from tqdm import tqdm

import numpy as np
import cv2

import torch

from yolox.utils import gather, is_main_process, postprocess, synchronize, time_synchronized
from yolox.data.data_augment import ValTransform


class TSR_ZO_Evaluator_Two:
    """
    VOC AP Evaluation class.
    """

    def __init__(
        self,
        dataloader,
        img_size,
        confthre,
        nmsthre,
        num_classes,
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.num_images = len(dataloader.dataset)
        self.preproc = ValTransform(bgr_means=(0.406, 0.456, 0.485),
                                    std=(0.225, 0.224, 0.229))

    def evaluate(
        self,
        model_list,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
    ):
        """
        VOC average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO style AP of IoU=50:95
            ap50 (float) : VOC 2007 metric AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model_list[0] = model_list[0].eval()
        model_list[1] = model_list[1].eval()
        if half:
            model_list[0] = model_list[0].half()
            model_list[1] = model_list[1].half()
        ids = []
        data_list = {}
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        preproc2_time = 0
        infer2_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model_list[0](x)
            model_list[0] = model_trt

        for cur_iter, (imgs, _, info_imgs, ids, img_info_org) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model_list[0](imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

                outputs = postprocess(
                    outputs, self.num_classes, self.confthre, self.nmsthre
                )
                if is_time_record:
                    nms_end = time_synchronized()
                    nms_time += nms_end - infer_end
                
                # --- zjw: for 2 stage ---
                for i in range(len(outputs)):
                    if outputs[i] is None:
                        continue
                    output_1st = outputs[i]
                    img_info = {"raw_img": img_info_org["raw_img"][i], "ratio": img_info_org["ratio"][i]}
                    # 2nd stage tsr model
                    input_2nd = process_1st_output(output_1st, img_info["raw_img"], img_info["ratio"], self.preproc)
                    if is_time_record:
                        preproc2_end = time_synchronized()
                        preproc2_time += preproc2_end - nms_end
                    outputs_2nd = model_list[1](input_2nd)
                    outputs_2nd = model_list[1].decoding(outputs_2nd)      # zjw
                    outputs[i] = process_result_combination(output_1st, outputs_2nd)
                if is_time_record:
                    infer2_end = time_synchronized()
                    infer2_time += infer2_end - preproc2_end
                    # --- zjw: for 2 stage ---

            data_list.update(self.convert_to_voc_format(outputs, info_imgs, ids))

        statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples, preproc2_time, infer2_time])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = ChainMap(*data_list)
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def convert_to_voc_format(self, outputs, info_imgs, ids):
        predictions = {}
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                predictions[int(img_id)] = (None, None, None)
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]

            predictions[int(img_id)] = (bboxes, cls, scores)
        return predictions

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()
        preproc2_time = statistics[3].item()
        infer2_time = statistics[4].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_size)
        a_preproc2_time = 1000 * preproc2_time / (n_samples * self.dataloader.batch_size)
        a_infer2_time = 1000 * infer2_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "NMS", "2nd_pre", "2nd_infer", "inference"],
                    [a_infer_time, a_nms_time, a_preproc2_time, a_infer2_time, 
                     (a_infer_time + a_nms_time + a_preproc2_time + a_infer2_time)],
                )
            ]
        )

        info = time_info + "\n"

        all_boxes = [
            [[] for _ in range(self.num_images)] for _ in range(self.num_classes)
        ]
        for img_num in range(self.num_images):
            bboxes, cls, scores = data_dict[img_num]
            if bboxes is None:
                for j in range(self.num_classes):
                    all_boxes[j][img_num] = np.empty([0, 5], dtype=np.float32)
                continue
            for j in range(self.num_classes):
                mask_c = cls == j
                if sum(mask_c) == 0:
                    all_boxes[j][img_num] = np.empty([0, 5], dtype=np.float32)
                    continue

                c_dets = torch.cat((bboxes, scores.unsqueeze(1)), dim=1)
                all_boxes[j][img_num] = c_dets[mask_c].numpy()

            sys.stdout.write(
                "im_eval: {:d}/{:d} \r".format(img_num + 1, self.num_images)
            )
            sys.stdout.flush()

        with tempfile.TemporaryDirectory() as tempdir:
            mAP50, mAP70 = self.dataloader.dataset.evaluate_detections(
                all_boxes, tempdir
            )
            logger.info("val/mAP50: {:.4f}, val/mAP50_95: {:.4f}".format(mAP70, mAP50))
            return mAP50, mAP70, info


def process_1st_output(output_1st, img, ratio, preproc=None):
    """ 
    output_1st: [tensor(n,7)] : bbox, obj_conf, class_conf, class_pred
    input_2nd:  (10, 3, 128, 128)
    """
    max_batch_num = 10
    num = output_1st.shape[0]
    boxes = output_1st[:, :4].cpu().numpy() / ratio.item()
    boxes = boxes.astype(np.int32)
    input_2nd = []
    # cv2.imshow('demo', img.cpu().numpy())
    for i in range(max_batch_num):
        if i < num:
            box = [max(x,0) for x in boxes[i, :]]
            img_crop = img[box[1]:box[3], box[0]:box[2]].cpu().numpy()
            img_resize = cv2.resize(img_crop, (128, 128))
            # cv2.imshow('demo_crop', img_resize)
            # input_2nd.append(img_resize.transpose((2,0,1)))
            img_resize, _ = preproc(img_resize, None, (128, 128))
            input_2nd.append(img_resize)
        else:
            input_2nd.append(input_2nd[-1])
    input_2nd = [torch.from_numpy(x) for x in input_2nd]
    input_2nd = torch.stack(input_2nd, 0)
    input_2nd = input_2nd.to(device=output_1st.device, dtype=output_1st.dtype)
    return input_2nd


def process_result_combination(output_1st, outputs_2nd):
    """ 
    output_1st: [tensor(n,7)] : bbox, obj_conf, class_conf, class_pred
    input_2nd:  tensor(10, 2): score, class
    output: 
    """
    # none_index = 66
    max_batch_num = 10
    output = output_1st
    num = min(output_1st.shape[0], max_batch_num)
    for i in range(num):
        output[i, 5] = outputs_2nd[i, 0]
        output[i, 6] = outputs_2nd[i, 1]
        # if outputs_2nd[i, 1] >= none_index:
        #     output[i, 5] = 0.0
        #     output[i, 6] = 0.0
        # else:
        #     output[i, 5] = outputs_2nd[i, 0]
        #     output[i, 6] = outputs_2nd[i, 1]
    return output