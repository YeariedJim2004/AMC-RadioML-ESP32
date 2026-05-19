# src/model_v2.py
# CNN v2 — ResNet + Squeeze-and-Excitation Attention (Compact)

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        weight = self.se(x)
        weight = weight.unsqueeze(-1)
        return x * weight


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResBlock, self).__init__()
        padding = kernel_size // 2
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size,
                      padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.se = SEBlock(out_channels, reduction=8)
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv_block(x)
        out = self.se(out)
        out = out + residual
        out = self.relu(out)
        return out


class AMCNet_v2(nn.Module):
    """
    CNN v2 Compact — ~280K params
    Input : (batch, 2, 128)
    Output: (batch, num_classes)
    """
    def __init__(self, num_classes=8, dropout=0.5):
        super(AMCNet_v2, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )

        self.stage1 = ResBlock(32, 64,  kernel_size=3)
        self.pool1  = nn.MaxPool1d(kernel_size=2)        # 128 → 64

        self.stage2 = ResBlock(64, 128, kernel_size=3)
        self.pool2  = nn.MaxPool1d(kernel_size=2)        # 64 → 32

        self.stage3 = ResBlock(128, 128, kernel_size=3)
        self.stage4 = ResBlock(128, 128, kernel_size=3)  # 256 → 128 (compact)

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.pool1(x)
        x = self.stage2(x)
        x = self.pool2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = x.squeeze(-1)
        x = self.classifier(x)
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = AMCNet_v2(num_classes=8).to(device)
    dummy = torch.randn(32, 2, 128).to(device)
    output = model(dummy)
    print(f"Input  shape : {dummy.shape}")
    print(f"Output shape : {output.shape}")
    print(f"Total params : {count_parameters(model):,}")
    print("Model v2 Compact OK!")
