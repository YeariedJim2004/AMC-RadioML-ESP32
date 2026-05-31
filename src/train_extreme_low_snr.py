import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np

# Path optimization
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from model_v3 import AMCNet_v3 
from dataset import RadioMLDataset

# --- Extreme Noise Suppression & Phase Tuning ---
class ExtremeSignalProcessing:
    @staticmethod
    def suppress_noise_peaks(iq_tensor, window_size=3):
        # Strict micro-smoothing to filter out massive noise spikes at -20dB
        padding = window_size // 2
        mean_filter = torch.ones(1, 1, window_size, device=iq_tensor.device) / window_size
        
        ch0 = iq_tensor[:, 0, :].unsqueeze(1) 
        ch1 = iq_tensor[:, 1, :].unsqueeze(1) 
        
        ch0_smooth = nn.functional.conv1d(ch0, mean_filter, padding=padding)
        ch1_smooth = nn.functional.conv1d(ch1, mean_filter, padding=padding)
        
        return torch.cat([ch0_smooth, ch1_smooth], dim=1)

    @staticmethod
    def apply_micro_phase(iq_tensor):
        # Ultra-tight phase lock loop simulation
        theta = np.radians(np.random.uniform(-1, 1))
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        
        I = iq_tensor[:, 0, :].clone()
        Q = iq_tensor[:, 1, :].clone()
        
        iq_tensor[:, 0, :] = I * cos_t - Q * sin_t
        iq_tensor[:, 1, :] = I * sin_t + Q * cos_t
        return iq_tensor

# Hardware setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device for Extreme Low-SNR Mission: {device}")

print("Loading dataset and strictly filtering Extreme Low-SNR Zone (-20dB to -15dB)...")
# Base load covering down to -20dB
base_dataset = RadioMLDataset("data/RML2016.10a_dict.pkl", snr_min=-20, augment=False)

# Filter strict boundaries: -20dB to -15dB only
targeted_indices = []
for idx in range(len(base_dataset)):
    _, _, snr_val = base_dataset[idx]
    if -20 <= snr_val <= -15:
        targeted_indices.append(idx)

extreme_low_subset = Subset(base_dataset, targeted_indices)
train_loader = DataLoader(extreme_low_subset, batch_size=64, shuffle=True, drop_last=True)
print(f"✅ Extreme Low-SNR Target Locked. Total Samples: {len(extreme_low_subset)}")

# Initialize Architecture
model = AMCNet_v3(num_classes=11).to(device)

# Extreme regularization to prevent overfitting to noise patterns
criterion = nn.CrossEntropyLoss(label_smoothing=0.08)
optimizer = optim.AdamW(model.parameters(), lr=0.0002, weight_decay=2e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-6) # Extended epochs for deep climb

num_epochs = 60
best_acc = 0.0

print("\n🚨 COMMENCING ULTIMATE MISSION: DESTINATION EXTREME LOW-SNR (80-90% TARGET) 🚨")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for iq_tensor, label, snr in train_loader:
        iq_tensor = iq_tensor.to(device)
        y_batch = label.to(device)
        
        # Dual-stage noise suppression
        X_batch = ExtremeSignalProcessing.suppress_noise_peaks(iq_tensor)
        X_batch = ExtremeSignalProcessing.apply_micro_phase(X_batch)
        
        I = X_batch[:, 0, :]
        Q = X_batch[:, 1, :]
        
        # 5-Channel Feature Extraction Pipeline
        phase = torch.atan2(Q, I)
        inst_freq = torch.diff(phase, dim=1, prepend=phase[:, :1])
        inst_freq = (inst_freq - inst_freq.mean(dim=1, keepdim=True)) / (inst_freq.std(dim=1, keepdim=True) + 1e-8)
        
        envelope = torch.sqrt(I**2 + Q**2)
        envelope = (envelope - envelope.mean(dim=1, keepdim=True)) / (envelope.std(dim=1, keepdim=True) + 1e-8)
        
        complex_sig = torch.complex(I, Q)
        fft_res = torch.fft.fft(complex_sig, dim=1)
        fft_mag = torch.abs(torch.fft.fftshift(fft_res, dim=1))
        fft_mag = (fft_mag - fft_mag.mean(dim=1, keepdim=True)) / (fft_mag.std(dim=1, keepdim=True) + 1e-8)
        
        final_features = torch.stack([I, Q, inst_freq, envelope, fft_mag], dim=1)
        
        optimizer.zero_grad()
        outputs = model(final_features)
        loss = criterion(outputs, y_batch)
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item() * X_batch.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
        
    scheduler.step()
    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100
    
    print(f"Epoch [{epoch+1:02d}/{num_epochs}] -> Loss: {epoch_loss:.4f} | Training Accuracy: {epoch_acc:.2f}%")
    
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        torch.save(model.state_dict(), "models/best_model_v4_extreme_low_snr.pth")

print(f"\n MISSION COMPLETE. HIGHEST EXTREME LOW-SNR ACCURACY SECURED: {best_acc:.2f}% ")