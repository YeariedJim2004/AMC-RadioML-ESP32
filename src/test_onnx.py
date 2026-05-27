import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('models/model_v3.onnx')

# Dummy input: (1, 5, 128)
dummy = np.random.randn(1, 5, 128).astype(np.float32)

output = session.run(['output'], {'input': dummy})
pred = np.argmax(output[0], axis=1)[0]

CLASSES = ['BPSK','QPSK','8PSK','QAM16','QAM64','PAM4','WBFM','AM-DSB','AM-SSB','GFSK','CPFSK']
print(f"ONNX inference OK ✓")
print(f"Output shape: {output[0].shape}")
print(f"Predicted class: {CLASSES[pred]}")