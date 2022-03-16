
import onnx_graphsurgeon as gs
import numpy as np
import onnx


model_path_in = "../YOLOX_outputs/tsr_tt100k_3_20220222/yolox_tt100k_nano3_640_200_1e-3_0p01_rm_unused_qat/best_qat_2.onnx"
model_path_out = "/home/zjw/workspace/AI/tools/TensorRT_test/perception_linux_test_2112/bin/model/best_qat_modified.onnx"

graph = gs.import_onnx(onnx.load(model_path_in))

q1 = [node for node in graph.nodes if node.name == "QuantizeLinear_2"][0]
dq1 = [node for node in graph.nodes if node.name == "DequantizeLinear_5"][0]

dq1.inputs[1].values = np.array(2.0074586868286133, dtype=np.float32)
q1.inputs[1].values = np.array(2.0074586868286133, dtype=np.float32)

onnx.save(gs.export_onnx(graph), model_path_out)
