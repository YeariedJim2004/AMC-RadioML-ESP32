import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
# dataset.py থেকে low_snr loader ইমপোর্ট করা হলো
from dataset import get_loaders_low_snr
from model import LightweightAMCNet

# ── Config ───────────────────────────────────────────────
DATASET_PATH    = r'D:\Signal Project\AMC-6G-Projects\AMC-RadioML-ESP32\data\RML2016.10a_dict.pkl'
MODEL_SAVE_PATH = r'D:\Signal Project\AMC-6G-Projects\AMC-RadioML-ESP32\models\best_model.pth'
LOG_SAVE_PATH   = r'D:\Signal Project\AMC-6G-Projects\AMC-RadioML-ESP32\results\train_log.json'

NUM_CLASSES  = 11   # ইয়ারিদ, আপনার সিদ্ধান্ত অনুযায়ী ১১টি ক্লাস লক করা হলো
EPOCHS       = 100
BATCH_SIZE   = 256
LR           = 0.001
PATIENCE     = 10

# লো-এসএনআর (Extreme Noise) সিমুলেশন রেঞ্জ
SNR_MIN      = -20
SNR_MAX      = -4
# ─────────────────────────────────────────────────────────


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for xb, yb, _ in loader:  # ← SNR আসে, _ দিয়ে ignore
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        out  = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        correct    += (out.argmax(1) == yb).sum().item()
        total      += xb.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for xb, yb, _ in loader:  # ← SNR আসে, _ দিয়ে ignore
        xb, yb = xb.to(device), yb.to(device)
        out  = model(xb)
        loss = criterion(out, yb)

        total_loss += loss.item() * xb.size(0)
        correct    += (out.argmax(1) == yb).sum().item()
        total      += xb.size(0)

    return total_loss / total, correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*60)
    print(f"🚀 AMC TRAINING ENGINE STARTED | DEVICE: {device}")
    print("="*60)

    # আপনার তৈরি করা dataset.py এর low_snr পাইপলাইন এখানে যুক্ত করা হলো
    train_loader, val_loader, test_loader, num_classes = get_loaders_low_snr(
        file_path = DATASET_PATH,
        batch_size = BATCH_SIZE,
        snr_min = SNR_MIN,
        snr_max = SNR_MAX
    )

    print(f"\n✓ Verified Target Classes in Dataset: {num_classes}")
    print(f"✓ Target SNR Training Range: {SNR_MIN} dB to {SNR_MAX} dB")

    # Model Initialization
    model     = LightweightAMCNet(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    no_improve    = 0
    log           = []

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(LOG_SAVE_PATH),   exist_ok=True)

    print("\n⏳ Training Epochs Progressing...")
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc   = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        # এই ডিকশনারি লগে জমা হয়ে রিসার্চ পেপারের জন্য train_log.json তৈরি করবে
        log.append({
            "epoch"   : epoch,
            "tr_loss" : round(tr_loss,  4),
            "tr_acc"  : round(tr_acc,   4),
            "val_loss": round(val_loss, 4),
            "val_acc" : round(val_acc,  4)
        })

        print(f"Epoch {epoch:03d} | "
              f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc*100:.2f}%")

        # Early Stopping & Best Checkpoint Tracking
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve    = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  ✔ Best model saved (val_loss={val_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch}.")
                break

    # পেপারের গ্রাফ প্লট করার জন্য লগ ফাইলটি রাইট করা হচ্ছে
    with open(LOG_SAVE_PATH, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"\n✓ Training complete. Log saved to -> {LOG_SAVE_PATH}")
    print(f"✓ Best weights locked at -> {MODEL_SAVE_PATH}")
    print("="*60)


if __name__ == "__main__":
    main()