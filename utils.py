import os
import random
import numpy as np
import torch
import csv
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# ----- Reproducibility -----
def set_seed(seed: int = 42):
    """Set random seeds for reproducible results"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----- Results Logging (Extended) -----
def log_results(filepath: str, fold: int, metrics: dict):
    """
    Log fold results to CSV file.
    
    metrics: dictionary containing metric_name -> value, e.g.,
    {
        'accuracy': 0.95,
        'precision': 0.94,
        'recall': 0.92,
        'f1': 0.93,
        'auc': 0.96
    }
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    file_exists = os.path.exists(filepath)
    
    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            header = ['Fold'] + list(metrics.keys())
            writer.writerow(header)
        
        row = [fold] + [f'{v:.6f}' if isinstance(v, float) else v for v in metrics.values()]
        writer.writerow(row)
    
    print(f"Results logged to {filepath}")

def read_results(filepath: str):
    """Read results from CSV file"""
    if not os.path.exists(filepath):
        return None
    folds, accuracies = [], []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            folds.append(int(row['Fold']))
            accuracies.append(float(row['Accuracy']))
    return {'folds': folds, 'accuracies': accuracies, 'mean': np.mean(accuracies), 'std': np.std(accuracies)}

# ----- Majority voting -----
def majority_vote_segment_logits(segment_logits: torch.Tensor) -> np.ndarray:
    """Two-stage majority voting for 3-class classification"""
    with torch.no_grad():
        preds = segment_logits.argmax(dim=2).cpu().numpy()
    final_labels = []
    for subject_preds in preds:
        vote = Counter(subject_preds)
        adhd_votes = vote.get(1, 0) + vote.get(2, 0)
        tdc_votes = vote.get(0, 0)
        if adhd_votes > tdc_votes:
            adhd_only_votes = [x for x in subject_preds if x != 0]
            if adhd_only_votes:
                subtype = Counter(adhd_only_votes).most_common(1)[0][0]
                final_labels.append(subtype)
            else:
                final_labels.append(1)
        else:
            final_labels.append(0)
    return np.array(final_labels)

def simple_majority_vote(predictions: np.ndarray) -> np.ndarray:
    """Simple majority voting for any number of classes"""
    voted_labels = []
    for subject_preds in predictions:
        majority_label = Counter(subject_preds).most_common(1)[0][0]
        voted_labels.append(majority_label)
    return np.array(voted_labels)

# ----- Metrics -----
def accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    if len(true) == 0:
        return 0.0
    return float((pred == true).mean())

def calculate_per_class_accuracy(pred: np.ndarray, true: np.ndarray, num_classes: int = 3):
    per_class_acc = {}
    for class_idx in range(num_classes):
        mask = true == class_idx
        per_class_acc[class_idx] = (pred[mask] == true[mask]).mean() if mask.sum() > 0 else 0.0
    return per_class_acc

# ----- Additional Metrics -----
def precision(pred: np.ndarray, true: np.ndarray, average: str = 'macro') -> float:
    if len(true) == 0:
        return 0.0
    return precision_score(true, pred, average=average, zero_division=0)

def recall(pred: np.ndarray, true: np.ndarray, average: str = 'macro') -> float:
    if len(true) == 0:
        return 0.0
    return recall_score(true, pred, average=average, zero_division=0)

def f1(pred: np.ndarray, true: np.ndarray, average: str = 'macro') -> float:
    if len(true) == 0:
        return 0.0
    return f1_score(true, pred, average=average, zero_division=0)

def auc_score(pred_probs: np.ndarray, true: np.ndarray, multi_class: str = 'ovr') -> float:
    if len(true) == 0:
        return 0.0
    return roc_auc_score(true, pred_probs, multi_class=multi_class)

def per_class_metrics(pred: np.ndarray, true: np.ndarray, num_classes: int = 3):
    """
    Returns a dictionary with accuracy, precision, recall, f1 per class
    """
    metrics = {}
    for class_idx in range(num_classes):
        mask = true == class_idx
        metrics[class_idx] = {
            'accuracy': (pred[mask] == true[mask]).mean() if mask.sum() > 0 else 0.0,
            'precision': precision_score(true[mask], pred[mask], zero_division=0) if mask.sum() > 0 else 0.0,
            'recall': recall_score(true[mask], pred[mask], zero_division=0) if mask.sum() > 0 else 0.0,
            'f1': f1_score(true[mask], pred[mask], zero_division=0) if mask.sum() > 0 else 0.0
        }
    return metrics

# ----- Early Stopping -----
class EarlyStopping:
    def __init__(self, patience: int = 10, mode: str = "max", min_delta: float = 1e-6):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.num_bad_epochs = 0
        self.should_stop = False

    def step(self, metric: float) -> bool:
        if self.best is None:
            self.best = metric
            self.num_bad_epochs = 0
            return False
        improved = (metric > self.best + self.min_delta) if self.mode == "max" else (metric < self.best - self.min_delta)
        if improved:
            self.best = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                self.should_stop = True
        return self.should_stop

    def reset(self):
        self.best = None
        self.num_bad_epochs = 0
        self.should_stop = False

# ----- Model Utils -----
def count_parameters(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable}

def save_model(model: torch.nn.Module, filepath: str):
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

def load_model(model: torch.nn.Module, filepath: str, device: torch.device):
    state_dict = torch.load(filepath, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Model loaded from {filepath}")
    return model
