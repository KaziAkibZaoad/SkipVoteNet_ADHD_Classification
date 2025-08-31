import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

class SkipBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.detail = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.approx = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.tanh = nn.Tanh()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        detail_out = self.tanh(self.detail(x))
        approx_out = self.tanh(self.approx(x))
        return self.pool(detail_out + approx_out)


class SkipVoteNet(nn.Module):
    def __init__(self, num_classes: int = 3, K: int = 7, input_size: int = 116):
        super().__init__()
        self.K = K
        self.num_classes = num_classes

        # Skip blocks
        self.skip1 = SkipBlock(1, 64, kernel_size=3)
        self.skip2 = SkipBlock(64, 128, kernel_size=3)
        self.skip3 = SkipBlock(128, 256, kernel_size=3)
        self.skip4 = SkipBlock(256, 512, kernel_size=4)

        # After 4 MaxPool2d operations: input_size / (2^4) = input_size / 16
        self.final_size = input_size // 16

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512 * self.final_size * self.final_size, num_classes)

        print(f"[SkipVoteNet] Initialized:")
        print(f"  Input size: {input_size}x{input_size}")
        print(f"  Final feature map size: {self.final_size}x{self.final_size}")
        print(f"  FC input features: {512 * self.final_size * self.final_size}")
        print(f"  Number of classes: {num_classes}")

    def forward_single_segment(self, seg: torch.Tensor) -> torch.Tensor:
        """Forward pass for a single dFC segment."""
        out = self.skip1(seg)
        out = self.skip2(out)
        out = self.skip3(out)
        out = self.skip4(out)
        out = self.flatten(out)
        out = self.dropout(out)
        return self.fc(out)

    def forward(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        """
        Args:
            x: [batch, K, N, N]
            return_logits:
                - True: averaged logits across K segments (for training)
                - False: majority-voted class per subject (for inference)
        """
        batch_size, K, N, _ = x.shape
        per_segment_logits = []

        for k in range(K):
            seg = x[:, k, :, :].unsqueeze(1)  # [batch, 1, N, N]
            logits = self.forward_single_segment(seg)  # [batch, num_classes]
            per_segment_logits.append(logits)

        per_segment_logits = torch.stack(per_segment_logits, dim=1)  # [batch, K, num_classes]

        if return_logits:
            return per_segment_logits.mean(dim=1)  # [batch, num_classes]

        # Majority vote for inference
        with torch.no_grad():
            segment_labels = per_segment_logits.argmax(dim=2)  # [batch, K]
            subject_preds = []
            for i in range(batch_size):
                counts = Counter(segment_labels[i].cpu().numpy())
                subject_preds.append(counts.most_common(1)[0][0])
            return torch.tensor(subject_preds, device=x.device, dtype=torch.long)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience method for inference."""
        self.eval()
        with torch.no_grad():
            return self.forward(x, return_logits=False)

    def get_segment_predictions(self, x: torch.Tensor) -> dict:
        """Return per-segment predictions and majority vote (for analysis)."""
        batch_size, K, N, _ = x.shape
        per_segment_logits = []

        with torch.no_grad():
            for k in range(K):
                seg = x[:, k, :, :].unsqueeze(1)
                logits = self.forward_single_segment(seg)
                per_segment_logits.append(logits)

        per_segment_logits = torch.stack(per_segment_logits, dim=1)  # [batch, K, num_classes]
        segment_predictions = per_segment_logits.argmax(dim=2)       # [batch, K]

        return {
            "segment_logits": per_segment_logits,
            "segment_predictions": segment_predictions,
            "majority_vote": self.forward(x, return_logits=False),
        }