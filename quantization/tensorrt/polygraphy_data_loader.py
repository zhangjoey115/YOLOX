#!/usr/bin/env python3
#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Defines a `load_data` function that returns a generator yielding
feed_dicts so that this script can be used as the argument for
the --data-loader-script command-line parameter.
"""
import cv2
import numpy as np


INPUT_SHAPE = (1, 3, 1024, 1920)


def get_image_processed(image_path="/home/zjw/workspace/AI/perception/YOLOX/models/lane/image/000349.jpg"):
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
    image = np.array(image, dtype=np.float32, order='C')

    return image


def load_data():
    for _ in range(1):
        # yield {"x": np.ones(shape=INPUT_SHAPE, dtype=np.float32)}  # Still totally real data
        yield {"images": get_image_processed()}  # Still totally real data
