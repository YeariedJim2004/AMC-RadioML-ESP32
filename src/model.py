# src/model.py
import torch
import torch.nn as nn

class LightweightAMCNet(nn.Module):
    def __init__(self, num_classes=11):
        super(LightweightAMCNet, self).__init__()
        
        # ১ম ব্লক: ইনপুট চ্যানেল ৫টি (I, Q, Inst_Freq, Envelope, FFT_Mag)
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # ২য় ব্লক
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # ৩য় ব্লক: গ্লোবাল পুলিং ও ক্লাসিফায়ার লেয়ার
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)  # আউটপুট: ১১টি মডুলেশন ক্লাস
        )
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)       # শেপ হবে: (Batch, 64, 1)
        x = x.squeeze(-1)      # শে放 হবে: (Batch, 64)
        x = self.fc(x)
        return x