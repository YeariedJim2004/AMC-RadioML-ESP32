# src/train_v2.py
# CNN v2 Training Script

import json
import os
import sys
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

# dataset.py এবং model_v2.py একই src/ ফোল্ডারে আছে
sys.path.append(os.path.dirname(__file__))
from dataset import get_loaders
from model_v2 import AMCNet_v2, count_parameters

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH   = r"D:\Signal Project\AMC-6G-Projects\AMC-RadioML-ESP32\data\RML2016.10a_dict.pkl"
MODEL_DIR   = r"D:\Signal Project\AMC-6G-Projects\AMC-RadioML-ESP32\models"
RESULTS_DIR = r"D:\Signal Project\AMC-6G-Projects\AMC-RadioML-ESP32\results"

EPOCHS      = 100
BATCH_SIZE  = 256
LR          = 0.0005
PATIENCE    = 15        # v1 ছিল 10, v2 একটু বেশি সুযোগ পাবে

os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")

# ── Data ──────────────────────────────────────────────────────────────────────
print("Loading dataset...")
train_loader, val_loader, test_loader, _ = get_loaders(DATA_PATH, batch_size=BATCH_SIZE)
print(f"Train batches : {len(train_loader)} | Val batches : {len(val_loader)}")

# ── Model ─────────────────────────────────────────────────────────────────────
model = AMCNet_v2(num_classes=11, dropout=0.5).to(device)
print(f"Total params  : {count_parameters(model):,}")

# ── Loss / Optimizer / Scheduler ──────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)

# ── Training Loop ─────────────────────────────────────────────────────────────
best_val_loss = float('inf')
patience_counter = 0
log = []

print(f"\nTraining started — Epochs: {EPOCHS}, Batch: {BATCH_SIZE}, LR: {LR}\n")

for epoch in range(1, EPOCHS + 1):

    # ── Train ──
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for xb, yb, _ in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        train_loss    += loss.item() * xb.size(0)
        preds          = out.argmax(dim=1)
        train_correct += (preds == yb).sum().item()
        train_total   += xb.size(0)

    train_loss /= train_total
    train_acc   = train_correct / train_total

    # ── Validation ──
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for xb, yb, _ in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out  = model(xb)
            loss = criterion(out, yb)

            val_loss    += loss.item() * xb.size(0)
            preds        = out.argmax(dim=1)
            val_correct += (preds == yb).sum().item()
            val_total   += xb.size(0)

    val_loss /= val_total
    val_acc   = val_correct / val_total

    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']

    # ── Log ──
    entry = {
        "epoch"     : epoch,
        "train_loss": round(train_loss, 4),
        "train_acc" : round(train_acc,  4),
        "val_loss"  : round(val_loss,   4),
        "val_acc"   : round(val_acc,    4),
        "lr"        : current_lr
    }
    log.append(entry)

    print(f"Epoch {epoch:03d}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
          f"LR: {current_lr:.6f}")

    # ── Best model save ──
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(),
                   os.path.join(MODEL_DIR, "best_model_v2.pth"))
        print(f"  ✓ Best model saved (val_loss={best_val_loss:.4f})")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (patience={PATIENCE})")
            break

# ── Save log ──────────────────────────────────────────────────────────────────
log_path = os.path.join(RESULTS_DIR, "train_log_v2.json")
with open(log_path, "w") as f:
    json.dump(log, f, indent=2)

print(f"\nTraining complete.")
print(f"Best val loss : {best_val_loss:.4f}")
print(f"Log saved     : {log_path}")
print(f"Model saved   : {os.path.join(MODEL_DIR, 'best_model_v2.pth')}")
