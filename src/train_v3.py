import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import time
# dataset.py থেকে আমাদের ফাইনাল লো-এসএনআর লোডার ইমপোর্ট করা হলো
from dataset import get_loaders_low_snr
# model_v3.py থেকে আপনার ফাইনাল AMCNet_v3 আর্কিটেকচার ইমপোর্ট করা হলো
from model_v3 import AMCNet_v3

# ── Config ───────────────────────────────────────────────
DATASET_PATH    = r'D:\Signal Project\AMC-6G-Projects\AMC-RadioML-ESP32\data\RML2016.10a_dict.pkl'
MODEL_SAVE_PATH = r'D:\Signal Project\AMC-6G-Projects\AMC-RadioML-ESP32\models\best_model_v3.pth'
LOG_SAVE_PATH   = r'D:\Signal Project\AMC-6G-Projects\AMC-RadioML-ESP32\results\train_log_v3.json'

NUM_CLASSES  = 11      # ১১টি মডুলেশন ক্লাস পুরোপুরি লকড
EPOCHS       = 100
BATCH_SIZE   = 256
LR           = 0.0005  # আপনার আগের লগের LR ধরে রাখা হলো
PATIENCE     = 10

# আমাদের সুনির্দিষ্ট ফাইনাল নয়েজ জোন (Ultra, Extreme & Low SNR)
SNR_MIN      = -20
SNR_MAX      = -4
# ─────────────────────────────────────────────────────────


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for xb, yb, _ in loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        out  = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        correct    += (out.argmax(1) == yb).sum().item()
        total      += xb.size(0)

    # জিপিইউ মেমরি ওভারফ্লো এবং ইলিগ্যাল অ্যাক্সেস ঠেকানোর জন্য ক্যাশ ক্লিয়ার
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for xb, yb, _ in loader:
        xb, yb = xb.to(device), yb.to(device)
        out  = model(xb)
        loss = criterion(out, yb)

        total_loss += loss.item() * xb.size(0)
        correct    += (out.argmax(1) == yb).sum().item()
        total      += xb.size(0)

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return total_loss / total, correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*60)
    print(f"🚀 FINAL AMC TRAINING ENGINE V3 | DEVICE: {device}")
    print("="*60)

    print("Loading low SNR dataset...")
    # আপনার ফাইনাল ৪টি জোন কাভার করার জন্য ফ্রেশ লো-এসএনআর লোডার
    train_loader, val_loader, test_loader, num_classes = get_loaders_low_snr(
        file_path = DATASET_PATH,
        batch_size = BATCH_SIZE,
        snr_min = SNR_MIN,
        snr_max = SNR_MAX
    )

    print(f"\n✓ Verified Target Classes in Dataset: {num_classes}")
    print(f"✓ Target SNR Training Range: {SNR_MIN} dB to {SNR_MAX} dB")

    # আপনার শক্তিশালী ResNet-Attention (AMCNet_v3) মডেল ইনিশিয়েলাইজেশন
    model = AMCNet_v3(num_classes=NUM_CLASSES, dropout=0.5).to(device)
    
    # টোটাল প্যারামিটার প্রিন্ট
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ AMCNet_v3 — Total Trainable Parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    no_improve    = 0
    log           = []

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(LOG_SAVE_PATH),   exist_ok=True)

    print(f"\nTraining start — Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | LR: {LR}\n")
    
    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        
        tr_loss, tr_acc   = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        
        epoch_time = time.time() - start_time

        # রিসার্চ পেপারের জন্য লার্নিং কার্ভ ডেটা ট্র্যাক
        log.append({
            "epoch"   : epoch,
            "tr_loss" : round(tr_loss,  4),
            "tr_acc"  : round(tr_acc,   4),
            "val_loss": round(val_loss, 4),
            "val_acc" : round(val_acc,  4)
        })

        print(f"Epoch {epoch:03d}/{EPOCHS} | "
              f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
              f"Time: {epoch_time:.1f}s")

        # বেস্ট মডেল সেভ ও লক মেকানিজম
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve    = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  >>> Best V3 model saved (val_loss={val_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch}.")
                break

    with open(LOG_SAVE_PATH, 'w') as f:
        json.dump(log, f, indent=2)
        
    print("\n" + "="*60)
    print(f"✓ V3 Training complete. Log saved to -> {LOG_SAVE_PATH}")
    print(f"✓ Best V3 weights locked at -> {MODEL_SAVE_PATH}")
    print("="*60)


if __name__ == "__main__":
    main()