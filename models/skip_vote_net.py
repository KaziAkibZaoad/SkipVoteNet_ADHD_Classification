import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

class SkipBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.detail = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.approx = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.tanh = nn.Tanh()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        detail_out = self.tanh(self.detail(x))
        approx_out = self.tanh(self.approx(x))
        out = self.pool(detail_out + approx_out)
        return out


class SkipVoteNet(nn.Module):
    def __init__(self, num_classes=3, K=7, input_size=116):
        super().__init__()
        self.K = K  # number of dFC segments
        self.num_classes = num_classes

        # Skip blocks
        self.skip1 = SkipBlock(1, 64, kernel_size=3)
        self.skip2 = SkipBlock(64, 128, kernel_size=3)
        self.skip3 = SkipBlock(128, 256, kernel_size=3)
        self.skip4 = SkipBlock(256, 512, kernel_size=4)

        # Calculate final feature map size dynamically
        # After 4 MaxPool2d operations: input_size / (2^4) = input_size / 16
        final_size = input_size // 16
        self.final_size = final_size
        
        # Flatten + Fully connected
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * final_size * final_size, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        print(f"SkipVoteNet initialized:")
        print(f"  Input size: {input_size}x{input_size}")
        print(f"  Final feature map size: {final_size}x{final_size}")
        print(f"  FC input features: {512 * final_size * final_size}")
        print(f"  Number of classes: {num_classes}")

    def forward_single_segment(self, seg):
        """Forward pass for a single segment"""
        out = self.skip1(seg)        # [batch, 64, N/2, N/2]
        out = self.skip2(out)        # [batch, 128, N/4, N/4]
        out = self.skip3(out)        # [batch, 256, N/8, N/8]
        out = self.skip4(out)        # [batch, 512, N/16, N/16]
        out = self.flatten(out)      # [batch, 512 * (N/16) * (N/16)]
        out = self.dropout(out)      # Regularization
        out = self.fc(out)           # [batch, num_classes]
        return out

    def forward(self, x, return_logits=False):
        """
        Input: x [batch, K, N, N]
        Output: 
          - if return_logits=False: y_pred_subject [batch] → subject-level predicted class (for inference)
          - if return_logits=True: logits [batch, num_classes] → for training with CrossEntropyLoss
        """
        batch_size, K, N, _ = x.shape
        per_segment_logits = []

        for k in range(K):
            seg = x[:, k, :, :].unsqueeze(1)  # [batch, 1, N, N]
            logits = self.forward_single_segment(seg)  # [batch, num_classes]
            per_segment_logits.append(logits)

        # Stack predictions across K segments
        per_segment_logits = torch.stack(per_segment_logits, dim=1)  # [batch, K, num_classes]

        if return_logits:
            # For training: return averaged logits across segments
            avg_logits = per_segment_logits.mean(dim=1)  # [batch, num_classes]
            return avg_logits
        else:
            # For inference: majority voting
            with torch.no_grad():
                # Convert logits to predicted class per segment
                segment_labels = per_segment_logits.argmax(dim=2)  # [batch, K]

                # Majority voting per subject (vectorized)
                subject_preds = []
                for i in range(batch_size):
                    segment_votes = segment_labels[i].cpu().numpy()
                    counts = Counter(segment_votes)
                    majority_class = counts.most_common(1)[0][0]
                    subject_preds.append(majority_class)

                return torch.tensor(subject_preds, device=x.device, dtype=torch.long)
    
    def predict(self, x):
        """Convenience method for inference"""
        self.eval()
        with torch.no_grad():
            return self.forward(x, return_logits=False)
    
    def get_segment_predictions(self, x):
        """Get detailed per-segment predictions for analysis"""
        batch_size, K, N, _ = x.shape
        per_segment_logits = []
        
        with torch.no_grad():
            for k in range(K):
                seg = x[:, k, :, :].unsqueeze(1)  # [batch, 1, N, N]
                logits = self.forward_single_segment(seg)  # [batch, num_classes]
                per_segment_logits.append(logits)
        
        per_segment_logits = torch.stack(per_segment_logits, dim=1)  # [batch, K, num_classes]
        segment_predictions = per_segment_logits.argmax(dim=2)  # [batch, K]
        
        return {
            'segment_logits': per_segment_logits,
            'segment_predictions': segment_predictions,
            'majority_vote': self.forward(x, return_logits=False)
        }