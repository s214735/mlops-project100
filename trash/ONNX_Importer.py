import onnx
import numpy as np
import onnxruntime as rt

model = onnx.load("resnet18.onnx")
onnx.checker.check_model(model)
#print(onnx.helper.printable_graph(model.graph))

ort_session = rt.InferenceSession("../../src/p100/model.py")
input_names = [i.name for i in ort_session.get_inputs()]
output_names = [i.name for i in ort_session.get_outputs()]
batch = {input_names[0]: np.random.randn(1, 3, 224, 224).astype(np.float32)}
out = ort_session.run(output_names, batch)
