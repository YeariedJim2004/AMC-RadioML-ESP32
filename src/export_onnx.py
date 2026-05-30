import os
import torch
from model_v3 import AMCNet_v3

DEVICE = torch.device("cpu") # হার্ডওয়্যার ডিপ্লয়মেন্টের জন্য CPU তে এক্সপোর্ট করা নিরাপদ

def export_to_onnx(pth_path, onnx_name):
    if not os.path.exists(pth_path):
        print(f"Error: {pth_path} not found!")
        return
        
    model = AMCNet_v3(num_classes=11).to(DEVICE)
    model.load_state_dict(torch.load(pth_path, map_location=DEVICE, weights_only=False))
    model.eval()
    
    # আমাদের ইনপুট শেপ: (Batch_Size, Channels, Length) -> (1, 5, 128)
    dummy_input = torch.randn(1, 5, 128, dtype=torch.float32).to(DEVICE)
    onnx_path = os.path.join("models", onnx_name)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14, # Raspberry Pi compatibility এর জন্য ওপেসেট ১৪ বেস্ট
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Successfully exported: {onnx_path}")

print("Starting ONNX Export...")
export_to_onnx("models/best_model_v3.pth", "amc_model_high.onnx")
export_to_onnx("models/best_model_v3_low_snr.pth", "amc_model_low.onnx")
