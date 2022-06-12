

import sys, os
import argparse
import cv2
import numpy as np
import common
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def make_parser():
    parser = argparse.ArgumentParser("Onnx-Trt parser")
    parser.add_argument(
        "-i",
        "--input",
        # default="/home/zjw/workspace/AI/perception/YOLOX/models/lane/image/000349.jpg",
        # default="/home/zjw/workspace/AI/perception/YOLOX/models/lane/image/001799.jpg",
        # default="/home/zjw/workspace/AI/perception/YOLOX/models/lane/image/007799.jpg",
        default='/home/zjw/workspace/AI/perception/vision_learning/deployment/sample_int8/test_pic/test.jpeg',
        type=str,
        help="input path",
    )
    parser.add_argument("-o", "--onnx", type=str, 
        # default="/home/zjw/workspace/AI/perception/YOLOX/models/lane/debug_python/lane_ep148_n1.onnx",
        # default="/home/zjw/workspace/AI/perception/YOLOX/models/lane/model_release/test_0514/lane_quant_0512_org.onnx.folded.onnx",
        # default="/home/zjw/workspace/AI/perception/YOLOX/models/lane/model_release/test_0512/lane_ep02_q.onnx",
        # default="/home/zjw/workspace/AI/perception/YOLOX/models/od/debug_infer/od_max_q1.onnx",
        default="/home/zjw/workspace/AI/perception/vision_learning/deployment/sample_int8/models/od_max_q1.onnx",
        help="onnx path")
    parser.add_argument("-e", "--engine", type=str, 
        # default="/home/zjw/workspace/AI/perception/YOLOX/models/lane/debug_python/lane_ep148_n1.trt",
        # default="/home/zjw/workspace/AI/perception/YOLOX/models/lane/model_release/test_0514/lane_quant_0512_org.onnx.folded.onnx",
        # default="/home/zjw/workspace/AI/perception/YOLOX/models/lane/model_release/test_0512/lane_ep02_q.trt",
        # default="/home/zjw/workspace/AI/perception/YOLOX/models/od/debug_infer/od_max_q1.trt",
        default="/home/zjw/workspace/AI/perception/vision_learning/deployment/sample_int8/models/od_max_q1_py.trt",
        help="engine path")
    parser.add_argument("-p", "--precision", type=str,
        # default="FP16", help="INT8|FP16|FP32")
        default="INT8", help="INT8|FP16|FP32")
    return parser


# outputs_fp = ['537', '344', '372']
# outputs_int = ['490', '491', '492', '552', '608', '1032', '1064', '1067',
#                '1069', '1121', '1123', '1137', '1149', '1150', '1162', '1163', '1164', '1178', '1190']
# outputs_size_fp = {'537': (1, 18, 512, 960), '344': (1, 18, 512, 960), '372': (1, 20, 254, 478)}
# outputs_size_int = {'490': (1, 18, 512, 960), '491': (1, 18, 512, 960), '492': (1, 18, 512, 960),
#                     '552': (1, 128, 128, 240),  # detail.out
#                     '608': (1, 20, 254, 478), # segbranch.Stem
#                     '1032': (1, 64, 32, 60), # segbranch.Nonlocal
#                     '1064': (1, 64, 32, 60), # segbranch.CE
#                     '1067': (1, 128, 32, 60), # segbranch.concat
#                     '1069': (1, 128, 32, 60), # ffm.maxpool2
#                     '1121': (1, 128, 128, 240), # ffm.left(mul)
#                     '1123': (1, 128, 32, 60), # ffm.sigmul
#                     '1137': (1, 128, 32, 60), # ffm.cbr
#                     '1149': (1, 128, 128, 240), # ffm.right
#                     '1150': (1, 128, 128, 240), # ffm.out
#                     '1162': (1, 64, 128, 240), '1163': (1, 64, 128, 240), '1164': (1, 64, 128, 240),
#                     '1178': (1, 32, 128, 240), '1190': (1, 3, 128, 240)}
outputs_fp = []
outputs_size_fp = {}
# outputs_int = []
# outputs_size_int = {}
outputs_int = ['1240', '1268', '1381']
outputs_size_int = {'1240': (1,128,32,60), '1268': (1,512,32,60), '1381': (1,256,16,30)}
outputs_mark = {'FP16': {'out': outputs_fp, 'size': outputs_size_fp},
                'INT8': {'out': outputs_int, 'size': outputs_size_int}}

g_task = 'od'
g_task_info = {'od': {'is_norm': False, 'output_shape': [(1, 10200, 26)]},
               'seg': {'is_norm': True, 'output_shape': [(1, 3, 256, 480), (1, 5, 256, 480)]}}


def get_image_processed(image_path, is_norm=False):
    if is_norm is True:
        mean = np.array([[[0.3257, 0.3690, 0.3223]]])
        std = np.array([[[0.2112, 0.2148, 0.2115]]])
        roi = [0, 56, 1920, 1080]
        # roi = [0, 0, 1920, 1024]
        size = [1920, 1024]
        factor = 255.0
    else:
        mean = np.array([[[0.0, 0.0, 0.0]]])
        std = np.array([[[1.0, 1.0, 1.0]]])
        roi = [0, 0, 1920, 1024]
        size = [1920, 1024]
        factor = 1.0

    image = cv2.imread(image_path)
    image = image[roi[1]:roi[3], roi[0]:roi[2]]
    image = cv2.resize(image, (size[0], size[1]))
    img_info = {'raw_img': image, 'ratio': 1, 'image_path': image_path}

    image = image.astype(np.float32) / factor
    image = (image - mean) / std
    
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :, :, :]
    image = np.array(image, dtype=np.float32, order='C')
    return image, img_info


def post_process_od(output_raw, image_info, output_save_path, index=-1):
    import torch
    from yolox.utils import postprocess_od, vis
    from yolox.data.datasets import OD_CLASSES as CLASS_NAME
    confthre = 0.3
    nmsthre = 0.7
    clsnum = len(CLASS_NAME)

    def decode_outputs(outputs, dtype):
        model_strides = [16, 32, 64, 64]
        model_hw = [(64, 120), (32, 60), (16, 30), (8, 15)]
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(model_hw, model_strides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs
    
    def visual(output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img

        output_fullbox = output['full_box']        
        output_fullbox = output_fullbox.cpu()
        bboxes = output_fullbox[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio
        cls = output_fullbox[:, 6]
        scores = output_fullbox[:, 4] * output_fullbox[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, CLASS_NAME)
        return vis_res

    output_np = output_raw[index]
    output_onnx_tensor = torch.from_numpy(output_np)
    output_onnx_decoded = decode_outputs(output_onnx_tensor, dtype=output_onnx_tensor[0].type())
    num_classes = {'full_box': clsnum, 'front_box': 3, 'side_box': 3}
    outputs_onnx_post = postprocess_od(
        output_onnx_decoded, num_classes, confthre,
        nmsthre, class_agnostic=True
    )
    result_image = visual(outputs_onnx_post[0], image_info, confthre)
    cv2.imwrite(output_save_path, result_image)


def post_process_seg(output_raw, output_save_path, index=0, cls=3):
    output_data = np.array(output_raw[index])
    output_data = np.argmax(output_data, 1).squeeze(0)
    # print(output_data.shape)
    image_output = output_data*int(255/cls)
    cv2.imwrite(output_save_path, image_output)
    return


def post_process(outputs_np, image_info, onnx_file_path, precision):
    import re
    name_ex = re.split('[./]', onnx_file_path)[-2]
    image_path = image_info['image_path']
    if g_task == 'od':
        image_path_out = image_path + '.py.{}.{}.{}.jpg'.format('od', precision, name_ex)
        post_process_od(outputs_np, image_info, image_path_out, index=-1)
    elif g_task == 'seg':
        image_path_out = image_path + '.py.{}.{}.{}.jpg'.format('lane', precision, name_ex)
        post_process_seg(outputs_np, image_path_out, index=-2, cls=3)
        image_path_out = image_path + '.py.{}.{}.{}.jpg'.format('seg', precision, name_ex)
        post_process_seg(outputs_np, image_path_out, index=-1, cls=5)


# ---------------- TRT Debug Begin ----------------

def _get_network_outputs(network):
    return [network.get_output(index).name for index in range(network.num_outputs)]


def mark_outputs(network, outputs=[]):
    """
    Mark the specified outputs as network outputs.

    Args:
        network (trt.INetworkDefinition): The network in which to mark outputs.
        outputs (Sequence[str]): The names of tensors to mark as outputs.
    """
    outputs = set(outputs)
    all_outputs = []
    for layer in network:
        for index in range(layer.num_outputs):
            tensor = layer.get_output(index)
            all_outputs.append(tensor.name)
            # Clear all old outputs
            # if tensor.is_network_output:
            #     network.unmark_output(tensor)
            if tensor.name in outputs:
                if not tensor.is_network_output:
                    print("Marking {:} as an output".format(tensor.name))
                    network.mark_output(tensor)

    marked_outputs = set(_get_network_outputs(network))
    not_found = outputs - marked_outputs
    return

# ---------------- TRT Debug End ----------------

def get_engine(onnx_file_path, engine_file_path="", precision='FP16'):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser, trt.Runtime(TRT_LOGGER) as runtime:
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print (parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            shape = network.get_input(0).shape # = [1, 3, 608, 608]
            print('Completed parsing of ONNX file')

            config.max_workspace_size = 1 << 33  # 8G
            if precision == 'FP16':
                config.set_flag(trt.BuilderFlag.FP16)
                print("Build Engine with precision FP16")
                mark_outputs(network, outputs_mark[precision]['out'])
            elif precision == 'INT8':
                config.set_flag(trt.BuilderFlag.INT8)
                print("Build Engine with precision INT8")
                mark_outputs(network, outputs_mark[precision]['out'])
            else:
                print("Build Engine with precision FP32")
            # builder.max_batch_size = 1

            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(plan)
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


def trt_infer(image_path, onnx_file_path, engine_file_path, precision):
    # Do inference with TensorRT
    trt_outputs = []
    image, image_info = get_image_processed(image_path, is_norm=g_task_info[g_task]['is_norm'])
    with get_engine(onnx_file_path, engine_file_path, precision) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
        inputs[0].host = image
        # Do inference
        trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

    # Before doing post-processing, we need to reshape the outputs as the common.do_inference will give us flat arrays.
    output_shapes = [size for size in outputs_mark[precision]['size'].values()]
    output_shapes_org = g_task_info[g_task]['output_shape']
    output_shapes.extend(output_shapes_org)
    outputs_np = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]

    post_process(outputs_np, image_info, onnx_file_path, precision)


if __name__ == "__main__":
    args = make_parser().parse_args()
    trt_infer(args.input, args.onnx, args.engine, args.precision)
