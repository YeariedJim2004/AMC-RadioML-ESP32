import torch
from model_v3 import AMCNet_v3  # তোমার actual class name যা আছে

# Model load
model = AMCNet_v3(num_classes=11)
model.load_state_dict(torch.load('models/best_model_v3.pth', map_location='cpu'))
model.eval()

# Dummy input: (batch=1, channels=5, length=128)
dummy = torch.randn(1, 5, 128)

# ONNX export
torch.onnx.export(
    model,
    dummy,
    'models/model_v3.onnx',
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
    opset_version=12
)

print("ONNX export done → models/model_v3.onnx")

# Verify
import onnx
m = onnx.load('models/model_v3.onnx')
onnx.checker.check_model(m)
print("ONNX model valid ✓")
