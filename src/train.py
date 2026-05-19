# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
from dataset import get_loaders
from model import LightweightAMCNet

# ─── Config ───────────────────────────────────────────────
DATASET_PATH = r'D:\Signal Project\AMC-6G-Projects\AMC-RadioML-ESP32\data\RML2016.10a_dict.pkl'
MODEL_SAVE_PATH = r'D:\Signal Project\AMC-6G-Projects\AMC-RadioML-ESP32\models\best_model.pth'
LOG_SAVE_PATH   = r'D:\Signal Project\AMC-6G-Projects\AMC-RadioML-ESP32\results\train_log.json'

NUM_CLASSES  = 8
EPOCHS       = 100
BATCH_SIZE   = 256
LR           = 0.001
PATIENCE     = 10          # Early stopping
# ──────────────────────────────────────────────────────────


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for xb, yb in loader:
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

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out  = model(xb)
        loss = criterion(out, yb)

        total_loss += loss.item() * xb.size(0)
        correct    += (out.argmax(1) == yb).sum().item()
        total      += xb.size(0)

    return total_loss / total, correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}\n")

    # Data
    train_loader, val_loader, _, full_dataset = get_loaders(
        pkl_path   = DATASET_PATH,
        batch_size = BATCH_SIZE,
        num_workers= 0
    )

    # Model
    model     = LightweightAMCNet(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss  = float('inf')
    no_improve     = 0
    log            = []

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(LOG_SAVE_PATH),   exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc   = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

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

        # Best model save
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve    = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  ✓ Best model saved (val_loss={val_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    # Save log
    with open(LOG_SAVE_PATH, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"\nTraining complete. Log saved to {LOG_SAVE_PATH}")


if __name__ == "__main__":
    main()
