import os
import json
import time
import torch
import torch.nn as nn
from dataset import get_loaders
from model_v3 import AMCNet_v3

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH  = r'D:\Signal Project\AMC-6G-Projects\AMC-RadioML-ESP32\data\RML2016.10a_dict.pkl'
MODEL_DIR  = r'D:\Signal Project\AMC-6G-Projects\AMC-RadioML-ESP32\models'
RESULT_DIR = r'D:\Signal Project\AMC-6G-Projects\AMC-RadioML-ESP32\results'

EPOCHS     = 100
BATCH_SIZE = 256
LR         = 0.0005
PATIENCE   = 15
SNR_MIN    = -4

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── Data ──────────────────────────────────────────────────────────────────────
print("\nLoading dataset...")
train_loader, val_loader, _, num_classes = get_loaders(
    DATA_PATH, batch_size=BATCH_SIZE, snr_min=SNR_MIN
)

# ── Model ─────────────────────────────────────────────────────────────────────
model = AMCNet_v3(num_classes=num_classes, dropout=0.5).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nAMCNet_v3 — Parameters: {total_params:,}")

# ── Optimizer & Scheduler ─────────────────────────────────────────────────────
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=8, verbose=True
)
# Class Weights
class_weights = torch.ones(num_classes)
class_weights[2] = 1.2   # 8PSK
class_weights[3] = 2.0   # QAM16
class_weights[4] = 2.0   # QAM64
class_weights[6] = 2.0   # WBFM
class_weights[7] = 2.0   # AM-DSB
class_weights = class_weights.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)

# ── Training Loop ─────────────────────────────────────────────────────────────
best_val_loss = float('inf')
patience_counter = 0
train_log = []

print(f"\nTraining start — Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | LR: {LR}\n")

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    # Train
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for xb, yb, _ in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        train_loss    += loss.item() * xb.size(0)
        preds          = logits.argmax(dim=1)
        train_correct += (preds == yb).sum().item()
        train_total   += xb.size(0)

    train_loss /= train_total
    train_acc   = train_correct / train_total

    # Validate
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for xb, yb, _ in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits  = model(xb)
            loss    = criterion(logits, yb)
            val_loss    += loss.item() * xb.size(0)
            preds        = logits.argmax(dim=1)
            val_correct += (preds == yb).sum().item()
            val_total   += xb.size(0)

    val_loss /= val_total
    val_acc   = val_correct / val_total

    scheduler.step(val_loss)
    elapsed = time.time() - t0

    print(f"Epoch {epoch:3d}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
          f"Time: {elapsed:.1f}s")

    train_log.append({
        'epoch': epoch,
        'train_loss': round(train_loss, 6),
        'train_acc':  round(train_acc,  6),
        'val_loss':   round(val_loss,   6),
        'val_acc':    round(val_acc,    6),
    })

    # Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        save_path = os.path.join(MODEL_DIR, 'best_model_v3.pth')
        torch.save(model.state_dict(), save_path)
        print(f"  >>> Best model saved (val_loss={best_val_loss:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

# ── Save log ──────────────────────────────────────────────────────────────────
log_path = os.path.join(RESULT_DIR, 'train_log_v3.json')
with open(log_path, 'w') as f:
    json.dump(train_log, f, indent=2)

print(f"\nTraining complete.")
print(f"Best val loss : {best_val_loss:.4f}")
print(f"Model saved   : {os.path.join(MODEL_DIR, 'best_model_v3.pth')}")
print(f"Log saved     : {log_path}")
