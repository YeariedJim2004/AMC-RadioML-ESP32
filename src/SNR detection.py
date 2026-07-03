import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# ── ১. DEEP 1D-CNN ARCHITECTURE FOR RF SIGNALS ─────────────────────
class SNR1DCNNRouter(nn.Module):
    def __init__(self):
        super(SNR1DCNNRouter, self).__init__()
        # Input shape: [Batch, 5, 128]
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2),
            
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2),
            
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # [Batch, 256, 1]
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 4) # 4 SNR Zones
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.squeeze(x, dim=-1) # [Batch, 256]
        return self.classifier(x)

# ── ২. 5-CHANNEL SIGNAL PREPARATION ENGINE ──────────────────────────
def z_score_normalize(arr):
    std = arr.std()
    return (arr - arr.mean()) / (std + 1e-8)

def generate_5channel_tensor(iq_sample):
    I = iq_sample[0]
    Q = iq_sample[1]
    
    envelope = np.sqrt(I**2 + Q**2)
    phase = np.arctan2(Q, I)
    inst_freq = np.diff(np.unwrap(phase), prepend=phase[0])
    fft_mag = np.abs(np.fft.fftshift(np.fft.fft(I + 1j * Q)))
    
    iq_5ch = np.vstack([
        z_score_normalize(I)[np.newaxis, :],
        z_score_normalize(Q)[np.newaxis, :],
        z_score_normalize(inst_freq)[np.newaxis, :],
        z_score_normalize(envelope)[np.newaxis, :],
        z_score_normalize(fft_mag)[np.newaxis, :]
    ]).astype(np.float32)
    
    return iq_5ch # Shape: [5, 128]

# ── ৩. DATASET LOADING & PREPROCESSING ──────────────────────────────
class SNRCNNRecordDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'rb') as f:
            raw = pickle.load(f, encoding='latin1')
            
        self.features = []
        self.labels = []
        
        print("📦 ৫-চ্যানেল তরঙ্গবিন্যাস এক্সট্রাক্ট করা হচ্ছে (RML2016)...")
        for (mod, snr), data in raw.items():
            if -20 <= snr < -15: zone_label = 0
            elif -15 <= snr < -10: zone_label = 1
            elif -10 <= snr < -4: zone_label = 2
            elif -4 <= snr <= 20: zone_label = 3
            else: continue
                
            for i in range(data.shape[0]):
                ch5_data = generate_5channel_tensor(data[i])
                self.features.append(ch5_data)
                self.labels.append(zone_label)
                
        self.features = np.array(self.features)
        self.labels = np.array(self.labels)
        print(f"✅ Loaded {len(self.features)} samples successfully.")

    def __len__(self): return len(self.features)
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx], dtype=torch.long)

# ── ৪. BALANCED TRAINING PROCESS ────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("models", exist_ok=True)
    
    dataset = SNRCNNRecordDataset("data/RML2016.10a_dict.pkl")
    
    # ইন্ডিভিজুয়াল জোন ৯৫% লক করার জন্য ক্লাস ওয়েট ব্যালেন্সিং
    class_counts = np.bincount(dataset.labels)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)
    
    model = SNR1DCNNRouter().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    best_acc = 0.0
    print("\n🚀 1D-CNN ইন্ডিভিজুয়াল জোন-লকিং ট্রেইনিং শুরু হচ্ছে...")
    
    for epoch in range(15): # সিএনএন দ্রুত শিখে ফেলে, তাই ১৫ ইপোকই যথেষ্ট
        model.train()
        train_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            outputs = model(x_b)
            loss = criterion(outputs, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                outputs = model(x_b)
                _, preds = torch.max(outputs, 1)
                total += y_b.size(0)
                correct += (preds == y_b).sum().item()
                
        val_acc = (correct / total) * 100
        print(f"🔥 CNN Epoch [{epoch+1:02d}/15] | Loss: {train_loss/len(train_loader):.4f} | Validation Accuracy: {val_acc:.2f}%")
        
        scheduler.step(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "models/best_snr_router.pth")

    print(f"\n🎯 1D-CNN Training Finished. Best Model Locked: {best_acc:.2f}%")

    # ── ৫. ONNX EXPORTATION (Shape: [1, 5, 128]) ────────────────────
    model.load_state_dict(torch.load("models/best_snr_router.pth"))
    model.eval()
    dummy_input = torch.randn(1, 5, 128).to(device)
    torch.onnx.export(
        model, dummy_input, "models/best_snr_router.onnx",
        export_params=True, opset_version=12,
        input_names=['router_input_5channels'], output_names=['router_output_zone']
    )
    print("✅ 1D-CNN ONNX Model Exported: models/best_snr_router.onnx")