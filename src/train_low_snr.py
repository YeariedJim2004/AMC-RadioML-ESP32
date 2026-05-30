import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# --- Force Python to recognize the 'src' directory path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Now import safely with the EXACT validated class name
from model_v3 import AMCNet_v3 
from dataset import RadioMLDataset

# --- Balanced Noise Augmentation Class ---
class SignalAugmentation:
    @staticmethod
    def apply_awgn(iq_tensor, snr_db):
        # Apply controlled extra noise to prevent complete signal destruction
        if snr_db < -10:
            return iq_tensor # Leave extreme low-snr to focal loss exploration
        noise_factor = 10 ** (-snr_db / 20.0) * 0.5
        noise = torch.randn_like(iq_tensor) * noise_factor
        return iq_tensor + noise

    @staticmethod
    def apply_phase_offset(iq_tensor):
        theta = np.radians(np.random.uniform(-5, 5)) # Micro-rotation
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        
        I = iq_tensor[0, :].clone()
        Q = iq_tensor[1, :].clone()
        
        iq_tensor[0, :] = I * cos_t - Q * sin_t
        iq_tensor[1, :] = I * sin_t + Q * cos_t
        return iq_tensor

# --- Custom Focal Loss for Low-SNR Domination ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()

# --- Hardware & Config Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Super-Computing Device: {device}")

print("Loading dataset for the FINAL Low-SNR Training Run...")
train_dataset = RadioMLDataset("data/RML2016.10a_dict.pkl", snr_min=-20, augment=False)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)

# Initialize Model Architecture
model = AMCNet_v3(num_classes=11).to(device)

# Advanced Optimizer & Focal Loss Setup (LR Optimized to 1e-4 to break local minima)
criterion = FocalLoss(gamma=2.0)
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# --- Ultimate Training Loop ---
num_epochs = 35  
best_loss = float('inf')

print("\n🚀 Starting the Final Deep-Learning Execution. No more turns after this! 🚀")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for iq_tensor, label, snr in train_loader:
        augmented_tensors = []
        for b in range(iq_tensor.size(0)):
            img = iq_tensor[b].clone()
            img = SignalAugmentation.apply_phase_offset(img)
            img = SignalAugmentation.apply_awgn(img, snr[b].item())
            augmented_tensors.append(img)
        
        X_batch = torch.stack(augmented_tensors).to(device)
        y_batch = label.to(device)
        
        I = X_batch[:, 0, :]
        Q = X_batch[:, 1, :]
        
        # Instantaneous Frequency tracking
        phase = torch.atan2(Q, I)
        inst_freq = torch.diff(phase, dim=1, prepend=phase[:, :1])
        inst_freq = (inst_freq - inst_freq.mean(dim=1, keepdim=True)) / (inst_freq.std(dim=1, keepdim=True) + 1e-8)
        
        # Amplitude Envelope tracking
        envelope = torch.sqrt(I**2 + Q**2)
        envelope = (envelope - envelope.mean(dim=1, keepdim=True)) / (envelope.std(dim=1, keepdim=True) + 1e-8)
        
        # FFT Magnitude tracking
        complex_sig = torch.complex(I, Q)
        fft_res = torch.fft.fft(complex_sig, dim=1)
        fft_mag = torch.abs(torch.fft.fftshift(fft_res, dim=1))
        fft_mag = (fft_mag - fft_mag.mean(dim=1, keepdim=True)) / (fft_mag.std(dim=1, keepdim=True) + 1e-8)
        
        final_features = torch.stack([I, Q, inst_freq, envelope, fft_mag], dim=1).to(device)
        
        optimizer.zero_grad()
        outputs = model(final_features)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * X_batch.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100
    
    scheduler.step(epoch_loss)
    
    print(f"Epoch [{epoch+1:02d}/{num_epochs}] -> Loss: {epoch_loss:.4f} | Training Accuracy: {epoch_acc:.2f}%")
    
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), "models/best_model_v4_low_snr.pth")

print("\n✅ FINAL DEFINITIVE MODEL SAVED SECURELY AS 'models/best_model_v4_low_snr.pth' ✅")