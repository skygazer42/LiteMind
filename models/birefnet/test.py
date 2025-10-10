import onnx
m = onnx.load("model.onnx")
t = m.graph.input[0].type.tensor_type
print([d.dim_value if d.dim_value else -1 for d in t.shape.dim])
