import re

with open('src/train_v3.py', 'r', encoding='utf-8') as f:
    content = f.read()

old = 'class FocalLoss(nn.Module):\n    def __init__(self, gamma=2.0):\n        super().__init__()\n        self.gamma = gamma\n    def forward(self, logits, targets):\n        ce = nn.functional.cross_entropy(logits, targets, reduction="none")\n        pt = torch.exp(-ce)\n        return ((1 - pt) ** self.gamma * ce).mean()\n\ncriterion = FocalLoss(gamma=2.0)'

new = '# Class Weights\nclass_weights = torch.ones(num_classes)\nclass_weights[3] = 3.0   # QAM16\nclass_weights[6] = 3.0   # WBFM\nclass_weights = class_weights.to(device)\n\ncriterion = nn.CrossEntropyLoss(weight=class_weights)'

if old in content:
    content = content.replace(old, new)
    with open('src/train_v3.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('SUCCESS')
else:
    print('ERROR — খুঁজে পাওয়া যায়নি')
