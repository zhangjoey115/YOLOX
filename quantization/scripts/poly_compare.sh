#!/usr/bin/env bash
# set -e

echo "compare: $1";

source ~/tools/miniconda3/bin/activate;
conda activate torch;

polygraphy run --onnxrt --int8 --workspace=10000M \
  /home/zjw/workspace/AI/perception/YOLOX/models/lane/model_release/test_0514/lane_quant_0512_org.onnx \
  --data-loader-script quantization/tensorrt/polygraphy_data_loader.py \
  --onnx-outputs=$1 --save-outputs models/lane/model_release/test_0514/output_org.log

polygraphy run --onnxrt --int8 --workspace=10000M \
  /home/zjw/workspace/AI/perception/YOLOX/models/lane/model_release/test_0514/lane_0514_q1_5.onnx \
  --data-loader-script quantization/tensorrt/polygraphy_data_loader.py \
  --onnx-outputs=$1 --save-outputs models/lane/model_release/test_0514/output_5.log

polygraphy run --abs 1e-5\
  --load-outputs models/lane/model_release/test_0514/output_org.log models/lane/model_release/test_0514/output_5.log