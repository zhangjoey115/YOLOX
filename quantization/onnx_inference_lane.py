
import argparse
import os
import time
from loguru import logger

import cv2
import numpy as np
import onnxruntime as rt


def image_process(image_path):
    mean = np.array([[[0.3257, 0.3690, 0.3223]]])
    std = np.array([[[0.2112, 0.2148, 0.2115]]])
    roi = [0, 56, 1920, 1080]
    size = [1920, 1024]

    image = cv2.imread(image_path)
    image = image[roi[1]:roi[3], roi[0]:roi[2]]
    image = cv2.resize(image, (size[0], size[1]))

    image = image.astype(np.float32) / 255.0
    image = (image - mean) / std
    
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :, :, :]
    image = np.array(image, dtype=np.float32)

    return image

def onnx_run():
    image_path = '/home/zjw/workspace/AI/perception/YOLOX/models/lane/image/000349.jpg'
    # onnx_path = "/home/zjw/workspace/AI/perception/YOLOX/models/lane/model_release/test_0514/lane_0514_q1_4.onnx"
    # onnx_path = "/home/zjw/workspace/AI/perception/YOLOX/models/lane/model_release/test_0514/lane_quant_0512_org.onnx.folded.onnx"
    onnx_path = "/home/zjw/Downloads/lane_latest_iou_double_head_0525.onnx"

    image_input = image_process(image_path)
    session = rt.InferenceSession(onnx_path)
    input = session.get_inputs()[0]
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    pred_onnx = session.run([], {input_name: image_input})

    output_data = np.array(pred_onnx[1])
    use_outside_argmax = False
    if use_outside_argmax is True:
        output_data = np.argmax(output_data, 1).squeeze(0)
        image_output = output_data*100
    else:
        output_data = output_data.squeeze(0).squeeze(0)
        max_v = np.max(output_data)
        scale = int(255/max_v)
        image_output = output_data*scale
        image_output = image_output.astype(int)
    print(output_data.shape)


    import re
    name_ex = re.split('[./]', onnx_path)[-2]
    image_path_out = image_path + '.onnx.{}.jpg'.format(name_ex)
    cv2.imwrite(image_path_out, image_output)


if __name__ == "__main__":
    onnx_run()
