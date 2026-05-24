import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from dataset import RadioMLDataset, SELECTED_MODS
from model_v3 import AMCNet_v3

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH  = r'D:\Signal Project\AMC-6G-Projects\AMC-RadioML-ESP32\data\RML2016.10a_dict.pkl'
MODEL_PATH = r'D:\Signal Project\AMC-6G-Projects\AMC-RadioML-ESP32\models\best_model_v3.pth'
RESULT_DIR = r'D:\Signal Project\AMC-6G-Projects\AMC-RadioML-ESP32\results'
SNR_MIN    = -4

os.makedirs(RESULT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── Load model ────────────────────────────────────────────────────────────────
model = AMCNet_v3(num_classes=len(SELECTED_MODS)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"Model loaded: {MODEL_PATH}")

# ── Load dataset ──────────────────────────────────────────────────────────────
print("\nLoading dataset...")
dataset = RadioMLDataset(DATA_PATH, snr_min=SNR_MIN)
loader  = torch.utils.data.DataLoader(
    dataset, batch_size=512, shuffle=False, num_workers=0
)

# ── Inference ─────────────────────────────────────────────────────────────────
all_preds  = []
all_labels = []
all_snrs   = []

with torch.no_grad():
    for xb, yb, sb in loader:
        xb     = xb.to(device)
        logits = model(xb)
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(yb.numpy())
        all_snrs.append(sb.numpy())

all_preds  = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)
all_snrs   = np.concatenate(all_snrs)

overall_acc = (all_preds == all_labels).mean()
print(f"\nOverall Accuracy: {overall_acc * 100:.2f}%")

# ── Per-SNR Accuracy ──────────────────────────────────────────────────────────
snr_levels = sorted(set(all_snrs.astype(int).tolist()))
snr_acc = {}
print("\nPer-SNR Accuracy:")
print(f"{'SNR':>6} | {'Accuracy':>9} | Zone")
print("-" * 35)

for snr in snr_levels:
    mask = all_snrs == snr
    acc  = (all_preds[mask] == all_labels[mask]).mean()
    snr_acc[int(snr)] = round(float(acc), 4)
    zone = "Low" if snr <= -4 else ("Mid" if snr <= 4 else "High")
    print(f"{snr:>6} dB | {acc * 100:>8.2f}% | {zone}")

# ── SNR Curve ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
snrs = list(snr_acc.keys())
accs = [snr_acc[s] * 100 for s in snrs]
ax.plot(snrs, accs, 'o-', color='steelblue', linewidth=2, markersize=6)
ax.axhline(overall_acc * 100, color='red', linestyle='--',
           label=f'Overall: {overall_acc*100:.2f}%')
ax.set_xlabel('SNR (dB)')
ax.set_ylabel('Accuracy (%)')
ax.set_title('CNN v3 — Per-SNR Accuracy (with Instantaneous Frequency)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, 'snr_accuracy_curve_v3.png'), dpi=150)
plt.close()
print(f"\nSNR curve saved.")

# ── Confusion Matrix ──────────────────────────────────────────────────────────
cm      = confusion_matrix(all_labels, all_preds)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
plt.colorbar(im, ax=ax)
ax.set_xticks(range(len(SELECTED_MODS)))
ax.set_yticks(range(len(SELECTED_MODS)))
ax.set_xticklabels(SELECTED_MODS, rotation=45, ha='right')
ax.set_yticklabels(SELECTED_MODS)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('CNN v3 — Confusion Matrix (with Instantaneous Frequency)')

for i in range(len(SELECTED_MODS)):
    for j in range(len(SELECTED_MODS)):
        val   = cm_norm[i, j]
        color = 'white' if val > 0.5 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                color=color, fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, 'confusion_matrix_v3.png'), dpi=150)
plt.close()
print("Confusion matrix saved.")

# ── Per-class Accuracy ────────────────────────────────────────────────────────
print("\nPer-class Accuracy:")
print(f"{'Modulation':>10} | {'Accuracy':>9}")
print("-" * 25)
for i, mod in enumerate(SELECTED_MODS):
    acc  = cm_norm[i, i]
    flag = "OK" if acc >= 0.85 else ("WARN" if acc >= 0.60 else "FAIL")
    print(f"{mod:>10} | {acc * 100:>8.2f}%  [{flag}]")

# ── Save JSON ─────────────────────────────────────────────────────────────────
results = {
    'overall_accuracy': round(float(overall_acc), 6),
    'snr_accuracy': snr_acc,
    'per_class_accuracy': {
        mod: round(float(cm_norm[i, i]), 6)
        for i, mod in enumerate(SELECTED_MODS)
    }
}
with open(os.path.join(RESULT_DIR, 'eval_results_v3.json'), 'w') as f:
    json.dump(results, f, indent=2)

print("\nDone. Check results/ folder.")