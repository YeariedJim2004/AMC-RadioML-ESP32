# src/model.py
# CNN v1 — LightweightAMCNet (Baseline)

import torch
import torch.nn as nn


class LightweightAMCNet(nn.Module):
    """
    Lightweight CNN for Automatic Modulation Classification.
    Input : (batch, 2, 128)  — I and Q channels
    Output: (batch, num_classes)
    """

    def __init__(self, num_classes: int = 8):
        super(LightweightAMCNet, self).__init__()

        # --- Block 1 ---
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        # --- Block 2 ---
        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.pool1 = nn.MaxPool1d(kernel_size=2)   # 128 → 64

        # --- Block 3 ---
        self.block3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # --- Block 4 ---
        self.block4 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.pool2 = nn.MaxPool1d(kernel_size=2)   # 64 → 32

        # --- Global Average Pooling ---
        self.gap = nn.AdaptiveAvgPool1d(1)         # (batch, 128, 1)

        # --- Fully Connected Head ---
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, num_classes)
            # CrossEntropyLoss ব্যবহার করবো তাই Softmax এখানে নেই
        )

    def forward(self, x):
        # x shape: (batch, 2, 128)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool1(x)

        x = self.block3(x)
        x = self.block4(x)
        x = self.pool2(x)

        x = self.gap(x)            # (batch, 128, 1)
        x = x.squeeze(-1)          # (batch, 128)

        x = self.classifier(x)    # (batch, num_classes)
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --- Quick sanity check ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = LightweightAMCNet(num_classes=8).to(device)

    dummy = torch.randn(32, 2, 128).to(device)   # batch=32
    output = model(dummy)

    print(f"Input  shape : {dummy.shape}")
    print(f"Output shape : {output.shape}")
    print(f"Total params : {count_parameters(model):,}")
    print("Model OK!")