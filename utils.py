import os
import random
import numpy as np
import torch
import csv
from collections import Counter

# ----- Reproducibility -----
def set_seed(seed: int = 42):
    """Set random seeds for reproducible results"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----- Results Logging -----
def log_results(filepath: str, fold: int, accuracy: float):
    """Log fold results to CSV file"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    # Check if file exists to determine if we need headers
    file_exists = os.path.exists(filepath)
    
    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if file is new
        if not file_exists:
            writer.writerow(['Fold', 'Accuracy'])
        
        # Write fold results
        writer.writerow([fold, f'{accuracy:.6f}'])
    
    print(f"Results logged to {filepath}")

def read_results(filepath: str):
    """Read results from CSV file"""
    if not os.path.exists(filepath):
        return None
    
    folds = []
    accuracies = []
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            folds.append(int(row['Fold']))
            accuracies.append(float(row['Accuracy']))
    
    return {
        'folds': folds,
        'accuracies': accuracies,
        'mean': np.mean(accuracies),
        'std': np.std(accuracies)
    }

# ----- Majority voting as in paper -----
def majority_vote_segment_logits(segment_logits: torch.Tensor) -> np.ndarray:
    """
    Two-stage majority voting for 3-class ADHD classification
    segment_logits: [B, K, C] torch tensor
    returns np.array [B] voted labels using:
        1) ADHD vs TDC (labels 1 or 2 vs 0)
        2) if ADHD => decide subtype by majority between {1,2}
    """
    with torch.no_grad():
        preds = segment_logits.argmax(dim=2).cpu().numpy()  # [B,K]
    
    final_labels = []
    for subject_preds in preds:  # For each subject
        vote = Counter(subject_preds)
        adhd_votes = vote.get(1, 0) + vote.get(2, 0)  # ADHDC + ADHDI
        tdc_votes = vote.get(0, 0)
        
        if adhd_votes > tdc_votes:
            # Subject is ADHD, now decide subtype
            adhd_only_votes = [x for x in subject_preds if x != 0]
            if adhd_only_votes:
                subtype = Counter(adhd_only_votes).most_common(1)[0][0]
                final_labels.append(subtype)
            else:
                # Fallback (shouldn't happen)
                final_labels.append(1)  # Default to ADHDC
        else:
            final_labels.append(0)  # TDC
    
    return np.array(final_labels)

def simple_majority_vote(predictions: np.ndarray) -> np.ndarray:
    """
    Simple majority voting for any number of classes
    predictions: [B, K] array where B=subjects, K=segments
    returns: [B] array of majority voted labels
    """
    voted_labels = []
    for subject_preds in predictions:
        vote_counts = Counter(subject_preds)
        majority_label = vote_counts.most_common(1)[0][0]
        voted_labels.append(majority_label)
    return np.array(voted_labels)

# ----- Metrics -----
def accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    """Calculate accuracy"""
    if len(true) == 0:
        return 0.0
    return float((pred == true).mean())

def calculate_per_class_accuracy(pred: np.ndarray, true: np.ndarray, num_classes: int = 3):
    """Calculate per-class accuracy"""
    per_class_acc = {}
    for class_idx in range(num_classes):
        mask = true == class_idx
        if mask.sum() > 0:
            per_class_acc[class_idx] = (pred[mask] == true[mask]).mean()
        else:
            per_class_acc[class_idx] = 0.0
    return per_class_acc

# ----- Early Stopping -----
class EarlyStopping:
    """Early stopping utility"""
    def __init__(self, patience: int = 10, mode: str = "max", min_delta: float = 1e-6):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.num_bad_epochs = 0
        self.should_stop = False

    def step(self, metric: float) -> bool:
        """
        Update early stopping state
        Returns True if training should stop
        """
        if self.best is None:
            self.best = metric
            self.num_bad_epochs = 0
            return False
        
        if self.mode == "max":
            improved = metric > self.best + self.min_delta
        else:  # mode == "min"
            improved = metric < self.best - self.min_delta
        
        if improved:
            self.best = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                self.should_stop = True
        
        return self.should_stop
    
    def reset(self):
        """Reset early stopping state"""
        self.best = None
        self.num_bad_epochs = 0
        self.should_stop = False

# ----- Model Utils -----
def count_parameters(model: torch.nn.Module):
    """Count model parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable}

def save_model(model: torch.nn.Module, filepath: str):
    """Save model state dict"""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

def load_model(model: torch.nn.Module, filepath: str, device: torch.device):
    """Load model state dict"""
    state_dict = torch.load(filepath, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Model loaded from {filepath}")
    return model