"""
evaluate_v2.py — CNN v2 (AMCNet_v2) Evaluation Script
Phase 4: Per-SNR accuracy, confusion matrix, SNR accuracy curve
 
Run from src/ directory:
    cd "D:\Signal Project\AMC-6G-Projects\AMC-RadioML-ESP32\src"
    python evaluate_v2.py
"""
 
import os
import sys
import json
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
 
# ── Local imports ──────────────────────────────────────────────────────────────
from model_v2 import AMCNet_v2
from dataset import RadioMLDataset, SELECTED_MODS
 
# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_FILE   = os.path.join(BASE_DIR, "data", "RML2016.10a_dict.pkl")
MODEL_PATH  = os.path.join(BASE_DIR, "models", "best_model_v2.pth")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
 
# ── Config ─────────────────────────────────────────────────────────────────────
BATCH_SIZE = 256
SEED       = 42
SNR_MIN    = -4   # v2 এর training range: -4 dB to +18 dB
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# ── SNR levels for v2 (-4 dB to +18 dB, step 2) ───────────────────────────────
SNR_LEVELS = list(range(SNR_MIN, 20, 2))   # [-4, -2, 0, 2, ..., 18]
 
 
def load_test_data():
    """
    Dataset load করে test split বানাও।
    dataset.py এর get_loaders() এর মতো same split logic — seed=42, 70/15/15.
    কিন্তু এখানে শুধু test set দরকার, তাই manually করছি।
    """
    print(f"[1/5] Dataset loading: {DATA_FILE}")
    with open(DATA_FILE, "rb") as f:
        raw = pickle.load(f, encoding="latin1")
 
    # RadioMLDataset তৈরি করো (snr_min=-4 → v2 range)
    full_dataset = RadioMLDataset(DATA_FILE, selected_mods=SELECTED_MODS, snr_min=SNR_MIN)
    print(f"      Total samples (SNR ≥ {SNR_MIN} dB): {len(full_dataset)}")
 
    # Reproducible split — same as get_loaders()
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(full_dataset)).tolist()
 
    n_total = len(indices)
    n_train = int(0.70 * n_total)
    n_val   = int(0.15 * n_total)
    # test = remaining 15%
    test_idx = indices[n_train + n_val:]
 
    test_subset = torch.utils.data.Subset(full_dataset, test_idx)
    test_loader = DataLoader(
        test_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0   # Windows: num_workers=0
    )
    print(f"      Test samples: {len(test_subset)}")
    return test_loader, full_dataset
 
 
def load_model():
    print(f"[2/5] Model loading: {MODEL_PATH}")
    model = AMCNet_v2(num_classes=len(SELECTED_MODS))
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"      Parameters: {total_params:,}")
    return model
 
 
def run_inference(model, test_loader):
    """
    Test set এ inference চালাও।
    Returns: all_preds, all_labels, all_snrs (numpy arrays)
    """
    print("[3/5] Running inference ...")
    all_preds  = []
    all_labels = []
    all_snrs   = []
 
    with torch.no_grad():
        for xb, yb, snr_b in test_loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(yb.numpy())
            all_snrs.append(snr_b.numpy())
 
    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_snrs   = np.concatenate(all_snrs)
    return all_preds, all_labels, all_snrs
 
 
def compute_accuracy(all_preds, all_labels, all_snrs):
    """Overall + per-SNR accuracy compute করো।"""
    print("[4/5] Computing accuracy ...")
 
    # Overall
    overall_acc = (all_preds == all_labels).mean() * 100
    print(f"\n  ✅ Overall Accuracy: {overall_acc:.2f}%")
 
    # Per-SNR
    print(f"\n  Per-SNR Accuracy ({SNR_MIN} dB to +18 dB):")
    print(f"  {'SNR (dB)':<12} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
    print(f"  {'-'*45}")
 
    snr_acc_map = {}
    for snr in SNR_LEVELS:
        mask = (all_snrs == snr)
        if mask.sum() == 0:
            snr_acc_map[snr] = None
            continue
        correct = (all_preds[mask] == all_labels[mask]).sum()
        total   = mask.sum()
        acc     = correct / total * 100
        snr_acc_map[snr] = acc
 
        zone = "✅" if snr >= 0 else "⚠️ " if snr >= -4 else "❌"
        print(f"  {zone} {snr:>+4} dB     {correct:<10} {total:<10} {acc:.2f}%")
 
    return overall_acc, snr_acc_map
 
 
def save_confusion_matrix(all_preds, all_labels):
    """Confusion matrix save করো → results/confusion_matrix_v2.png"""
    out_path = os.path.join(RESULTS_DIR, "confusion_matrix_v2.png")
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
 
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=SELECTED_MODS,
        yticklabels=SELECTED_MODS,
        ax=ax,
        linewidths=0.5
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("CNN v2 (AMCNet_v2) — Normalized Confusion Matrix\n"
                 f"(SNR ≥ {SNR_MIN} dB, Test Set)", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\n  📊 Confusion matrix saved → {out_path}")
 
 
def save_snr_curve(snr_acc_map, overall_acc):
    """Per-SNR accuracy curve save করো → results/snr_accuracy_curve_v2.png"""
    out_path = os.path.join(RESULTS_DIR, "snr_accuracy_curve_v2.png")
 
    snrs = [s for s in SNR_LEVELS if snr_acc_map.get(s) is not None]
    accs = [snr_acc_map[s] for s in snrs]
 
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(snrs, accs, marker="o", linewidth=2, markersize=7,
            color="#2563EB", label="CNN v2 (ResNet+SE)")
    ax.axhline(y=overall_acc, color="gray", linestyle="--", linewidth=1.2,
               label=f"Overall: {overall_acc:.2f}%")
    ax.axhline(y=90, color="green", linestyle=":", linewidth=1,
               label="Target 90%")
 
    # Zone shading
    ax.axvspan(SNR_MIN - 1, -0.5, alpha=0.07, color="orange", label="Mid SNR zone")
    ax.axvspan(-0.5, 19,         alpha=0.07, color="blue",   label="High SNR zone")
 
    ax.set_xlabel("SNR (dB)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("CNN v2 — Per-SNR Accuracy\n"
                 "(AMCNet_v2, ResNet + SE Attention)", fontsize=13)
    ax.set_xticks(snrs)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  📈 SNR accuracy curve saved → {out_path}")
 
 
def save_json(overall_acc, snr_acc_map):
    """Results JSON এ save করো → results/eval_results_v2.json"""
    out_path = os.path.join(RESULTS_DIR, "eval_results_v2.json")
    results = {
        "model": "AMCNet_v2",
        "snr_min_train": SNR_MIN,
        "snr_levels_evaluated": SNR_LEVELS,
        "modulations": SELECTED_MODS,
        "overall_accuracy_pct": round(overall_acc, 4),
        "per_snr_accuracy_pct": {
            str(snr): (round(acc, 4) if acc is not None else None)
            for snr, acc in snr_acc_map.items()
        }
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  💾 Results JSON saved → {out_path}")
 
 
def main():
    print("=" * 55)
    print("  AMC Project — Phase 4: evaluate_v2.py")
    print(f"  Device: {DEVICE}")
    print("=" * 55)
 
    test_loader, _ = load_test_data()
    model          = load_model()
    all_preds, all_labels, all_snrs = run_inference(model, test_loader)
    overall_acc, snr_acc_map        = compute_accuracy(all_preds, all_labels, all_snrs)
 
    print("\n[5/5] Saving outputs ...")
    save_confusion_matrix(all_preds, all_labels)
    save_snr_curve(snr_acc_map, overall_acc)
    save_json(overall_acc, snr_acc_map)
 
    print("\n" + "=" * 55)
    print(f"  Phase 4 Complete ✅")
    print(f"  Overall Accuracy: {overall_acc:.2f}%")
    print("=" * 55)
 
 
if __name__ == "__main__":
    main()
 