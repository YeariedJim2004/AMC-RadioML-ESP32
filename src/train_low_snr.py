import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from dataset import get_loaders_low_snr
from model_v3 import AMCNet_v3

# â”€â”€â”€ Config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
FILE_PATH   = r"D:\Signal Project\AMC-6G-Projects\AMC-RadioML-ESP32\data\RML2016.10a_dict.pkl"
SAVE_PATH   = r"D:\Signal Project\AMC-6G-Projects\AMC-RadioML-ESP32\models\best_model_v3_low_snr.pth"
PRETRAINED  = r"D:\Signal Project\AMC-6G-Projects\AMC-RadioML-ESP32\models\best_model_v3.pth"

BATCH_SIZE  = 128        # à¦›à§‹à¦Ÿ batch â€” better generalization
EPOCHS      = 60
LR          = 5e-4
SNR_MIN     = -20
SNR_MAX     = 0          # 0 dB à¦ªà¦°à§à¦¯à¦¨à§à¦¤ à¦¬à¦¾à¦¡à¦¼à¦¾à¦¨à§‹ à¦¹à¦¯à¦¼à§‡à¦›à§‡
NUM_CLASSES = 11
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def freeze_backbone(model, freeze=True):
    """conv layers freeze à¦•à¦°à§‹, à¦¶à§à¦§à§ classifier layers train à¦¹à¦¬à§‡"""
    for name, param in model.named_parameters():
        if 'fc' in name or 'classifier' in name or 'bn3' in name:
            param.requires_grad = True
        else:
            param.requires_grad = not freeze
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[Freeze] Trainable params: {trainable:,}')


def train():
    print(f'[Device] {DEVICE}')

    train_loader, val_loader, test_loader, num_classes = get_loaders_low_snr(
        FILE_PATH, batch_size=BATCH_SIZE, snr_min=SNR_MIN, snr_max=SNR_MAX
    )

    model = AMCNet_v3(num_classes=NUM_CLASSES).to(DEVICE)
    if os.path.exists(PRETRAINED):
        model.load_state_dict(torch.load(PRETRAINED, map_location=DEVICE, weights_only=True))
        print(f'[Model] Loaded pretrained: {PRETRAINED}')
    else:
        print('[Model] WARNING: No pretrained found')

    # Phase 1: backbone freeze â€” à¦¶à§à¦§à§ head train (epoch 1-20)
    freeze_backbone(model, freeze=True)
    criterion = FocalLoss(gamma=2.0)
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1, eta_min=1e-6)

    best_val_acc = 0.0
    phase = 1

    for epoch in range(1, EPOCHS + 1):

        # Phase 2: epoch 21 à¦¥à§‡à¦•à§‡ à¦¸à¦¬ layers unfreeze
        if epoch == 21 and phase == 1:
            print('\n[Phase 2] Unfreezing all layers...')
            freeze_backbone(model, freeze=False)
            optimizer = Adam(model.parameters(), lr=LR * 0.1, weight_decay=1e-4)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1, eta_min=1e-6)
            phase = 2

        # Train
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x, y, _ in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct    += (out.argmax(1) == y).sum().item()
            total      += x.size(0)
        train_acc = correct / total * 100

        # Val
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                val_correct += (model(x).argmax(1) == y).sum().item()
                val_total   += x.size(0)
        val_acc = val_correct / val_total * 100

        scheduler.step()

        print(f'Epoch [{epoch:02d}/{EPOCHS}] P{phase} | '
              f'Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f'  â˜… Best saved â€” Val Acc: {val_acc:.2f}%')

    print(f'\n[Done] Best Val Acc: {best_val_acc:.2f}%')


if __name__ == '__main__':
    train()
