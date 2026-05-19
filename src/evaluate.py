# src/evaluate.py
import torch
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from dataset import RadioMLDataset
from model   import LightweightAMCNet

DATASET_PATH = r'D:\Signal Project\AMC-6G-Projects\AMC-RadioML-ESP32\data\RML2016.10a_dict.pkl'
MODEL_PATH   = r'D:\Signal Project\AMC-6G-Projects\AMC-RadioML-ESP32\models\best_model.pth'
RESULTS_DIR  = r'D:\Signal Project\AMC-6G-Projects\AMC-RadioML-ESP32\results'
NUM_CLASSES  = 8
BATCH_SIZE   = 256

@torch.no_grad()
def predict_all(model, dataset, device, batch_size=256):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_preds, all_labels, all_snrs = [], [], []
    model.eval()
    for iq, label, snr in loader:
        iq = iq.to(device)
        out = model(iq)
        all_preds.append(out.argmax(1).cpu().numpy())
        all_labels.append(label.numpy())
        all_snrs.append(snr.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels), np.concatenate(all_snrs)

def plot_confusion_matrix(preds, labels, class_names, save_path):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                vmin=0, vmax=1, ax=ax)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix (Normalized)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[SAVED] Confusion matrix -> {save_path}")

def plot_snr_accuracy(snr_acc_dict, save_path):
    snrs = sorted(snr_acc_dict.keys())
    accs = [snr_acc_dict[s] * 100 for s in snrs]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(snrs, accs, marker='o', linewidth=2, color='steelblue')
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Per-SNR Classification Accuracy', fontsize=14)
    ax.set_xticks(snrs)
    ax.set_ylim(0, 105)
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% line')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[SAVED] SNR curve -> {save_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}\n")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    full_dataset = RadioMLDataset(DATASET_PATH)
    class_names  = full_dataset.get_class_names()
    snr_map      = full_dataset.get_snr_index_map()

    model = LightweightAMCNet(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"[INFO] Model loaded\n")

    preds, labels, snrs = predict_all(model, full_dataset, device, BATCH_SIZE)

    overall_acc = (preds == labels).mean()
    print(f"Overall Accuracy: {overall_acc*100:.2f}%")

    snr_acc = {}
    print("\nPer-SNR Accuracy:")
    print(f"{'SNR':>6} | {'Acc':>8} | {'Correct':>8} / {'Total':>6}")
    print("-" * 38)
    for snr_val in sorted(snr_map.keys()):
        idx     = snr_map[snr_val]
        correct = (preds[idx] == labels[idx]).sum()
        acc     = correct / len(idx)
        snr_acc[snr_val] = float(acc)
        print(f"{snr_val:>+6} dB | {acc*100:>7.2f}% | {correct:>8} / {len(idx):>6}")

    result_json = {
        "overall_accuracy": round(float(overall_acc), 4),
        "per_snr_accuracy": {str(k): round(v, 4) for k, v in snr_acc.items()}
    }
    with open(os.path.join(RESULTS_DIR, "eval_results.json"), 'w') as f:
        json.dump(result_json, f, indent=2)

    plot_confusion_matrix(preds, labels, class_names,
                          os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plot_snr_accuracy(snr_acc,
                      os.path.join(RESULTS_DIR, "snr_accuracy_curve.png"))
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()
