import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes, K=7, input_size=116):
        super().__init__()
        self.K = K
        self.conv1 = nn.Conv2d(K, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * (input_size//2) * (input_size//2), 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x, return_logits=False):
        # x: [batch, K, 116, 116]
        x = F.tanh(self.conv1(x))
        x = self.pool(F.tanh(self.conv2(x)))
        x = x.view(x.size(0), -1)
        logits = self.fc2(F.tanh(self.fc1(x)))
        if return_logits:
            return logits
        else:
            preds = torch.argmax(logits, dim=1)
            return preds
