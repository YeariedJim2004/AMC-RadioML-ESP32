import os
import torch
import torch.nn as nn

# আপনার লকড model_v3 থেকে আর্কিটেকচার ইমপোর্ট করা হলো
from model_v3 import AMCNet_v3

def export_to_onnx(model_path, output_name):
    if not os.path.exists(model_path):
        print(f"❌ Error: {model_path} not found!")
        print("অনুগ্রহ করে নিশ্চিত করুন train_v3.py রান হয়ে মডেলটি তৈরি হয়েছে কি না।")
        return
        
    print(f"Initializing AMCNet_v3 and loading weights for {output_name}...")
    model = AMCNet_v3(num_classes=11, dropout=0.5)
    
    # সিপিইউ ম্যাপ লোকেশন দিয়ে ওয়েইট লোড (রাসবেরি পাই ফ্রেন্ডলি)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # আমাদের ৫-চ্যানেল পাইপলাইনের ইনপুট সাইজ: [Batch_size=1, Channels=5, Signal_Length=128]
    dummy_input = torch.randn(1, 5, 128)
    
    # এক্সপোর্ট পাথ ডাইনামিকালি সেট
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
        dynamic_axes={
            'input_5channel': {0: 'batch_size'}, 
            'output_modulation': {0: 'batch_size'}
        }
    )
    print(f"✅ Successfully exported: {onnx_path}\n")

if __name__ == "__main__":
    # আপনার train_v3.py দ্বারা তৈরি প্রজেক্টের আসল বেস্ট মডেল পাথ
    FINAL_MODEL_PATH = r"D:\Signal Project\AMC-6G-Projects\AMC-RadioML-ESP32\models\best_model_v3.pth"
    
    print("="*60)
    print("🚀 AMC ONNX EXPORT ENGINE V3")
    print("="*60)
    
    export_to_onnx(FINAL_MODEL_PATH, "AMCNet_v3_Universal_Model")