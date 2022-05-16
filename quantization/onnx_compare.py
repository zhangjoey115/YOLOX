
import onnx_graphsurgeon as gs
import numpy as np
import onnx


model_path_1 = "/home/zjw/workspace/AI/perception/YOLOX/models/lane/model_release/test_0514/lane_quant_0512_org.onnx"
model_path_2 = "/home/zjw/workspace/AI/perception/YOLOX/models/lane/model_release/test_0514/lane_0514_q1_4.onnx"

graph1 = gs.import_onnx(onnx.load(model_path_1))
graph2 = gs.import_onnx(onnx.load(model_path_2))

graph1.fold_constants().cleanup()
onnx.save(gs.export_onnx(graph1), model_path_1+".folded.onnx")
graph1 = gs.import_onnx(onnx.load(model_path_1+".folded.onnx"))


nodes1 = [node for node in graph1.nodes if not node.name.startswith('Constant')]
nodes1_c = [node for node in graph1.nodes if node.name.startswith('Constant')]
nodes2 = [node for node in graph2.nodes]

v1 = nodes1[2]
v2 = nodes2[0]
# dq1.inputs[1].values = np.array(2.0074586868286133, dtype=np.float32)
# q1.inputs[1].values = np.array(2.0074586868286133, dtype=np.float32)

# onnx.save(gs.export_onnx(graph), model_path_out)
pass
