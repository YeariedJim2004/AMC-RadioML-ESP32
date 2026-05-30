import os
import torch
import numpy as np
from dataset import RadioMLDataset, SELECTED_MODS
from model_v3 import AMCNet_v3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/best_model_v3_low_snr.pth"

model = AMCNet_v3(num_classes=11).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
model.eval()

# snr_max ছাড়া নরমাল ডাটাসেট লোড (যা আপনার পিসিতে অলরেডি কাজ করছে)
ds = RadioMLDataset("data/RML2016.10a_dict.pkl", snr_min=-20, augment=False)

snr_correct = {}
snr_total = {}

print("Evaluating, please wait...")
with torch.no_grad():
    for iq, label, snr in ds:
        snr_val = int(snr.item())
        
        # আমরা স্ক্রিপ্টের ভেতরেই ম্যানুয়ালি লো-এসএনআর ফিল্টার করে নিচ্ছি (-20 dB থেকে -4 dB)
        if snr_val > -4:
            continue
            
        out = model(iq.unsqueeze(0).to(DEVICE))
        pred = out.argmax(dim=1).item()
        
        snr_correct[snr_val] = snr_correct.get(snr_val, 0) + (1 if pred == label.item() else 0)
        snr_total[snr_val] = snr_total.get(snr_val, 0) + 1

print("\n--- Low-SNR Model Per-SNR Accuracy ---")
for snr in sorted(snr_correct.keys()):
    acc = 100 * snr_correct[snr] / snr_total[snr]
    print(f"  {snr:4d} dB: {acc:.2f}%")
