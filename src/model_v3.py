import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.se(x).unsqueeze(-1)
        return x * scale


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3):
        super().__init__()
        pad = kernel // 2

        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, kernel, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.se = SEBlock(out_ch)
        self.relu = nn.ReLU()

        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm1d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.conv(x)
        out = self.se(out)
        out = self.relu(out + self.shortcut(x))
        return out


class AMCNet_v3(nn.Module):
    def __init__(self, num_classes=11, dropout=0.5):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(5, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.block1 = ResBlock(32, 64)
        self.pool1  = nn.MaxPool1d(2)

        self.block2 = ResBlock(64, 128)
        self.pool2  = nn.MaxPool1d(2)

        self.block3 = ResBlock(128, 128)
        self.block4 = ResBlock(128, 128)

        self.gap     = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)