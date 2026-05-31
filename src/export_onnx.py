import os
import torch
import torch.nn as nn

# Import architecture from model_v3
from model_v3 import AMCNet_v3

def export_zone_model(model_path, output_name):
    if not os.path.exists(model_path):
        print(f"Warning: {model_path} not found. Skipping {output_name}.")
        return
        
    print(f"Initializing AMCNet_v3 and loading weights for {output_name}...")
    model = AMCNet_v3(num_classes=11)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # Input Size for our 5-Channel Pipeline: [Batch_size=1, Channels=5, Signal_Length=128]
    dummy_input = torch.randn(1, 5, 128)
    
    # ONNX path definition
    onnx_path = model_path.replace(".pth", ".onnx")
    
    print(f"Exporting {output_name} to ONNX format...")
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input_5channel'],
        output_names=['output_modulation'],
        dynamic_axes={'input_5channel': {0: 'batch_size'}, 'output_modulation': {0: 'batch_size'}}
    )
    print(f"Successfully exported: {onnx_path}\n")

if __name__ == "__main__":
    # Exporting all 4 trained boundary models to ONNX
    export_zone_model("models/best_model_v4_extreme_low_snr.pth", "Extreme_Low_SNR")
    export_zone_model("models/best_model_v4_ultra_low_snr.pth", "Ultra_Low_SNR")
    export_zone_model("models/best_model_v4_low_snr.pth", "Standard_Low_SNR")
    export_zone_model("models/best_model_v4_mid_snr.pth", "Mid_SNR")