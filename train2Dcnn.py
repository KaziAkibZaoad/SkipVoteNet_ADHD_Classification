import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
import os
from torch.amp import autocast, GradScaler  # Mixed precision
from models.simple_CNN import SimpleCNN
from data.ADHD200_DFC_Dataset import ADHD200DFCDataset
from utils import set_seed, log_results

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import argparse

# ---- Config ----
BATCH_SIZE = 8
LR = 1e-5
EPOCHS = 100
PATIENCE = 10
WEIGHT_DECAY = 1e-3
KFOLDS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_metrics(preds, labels, num_classes):
    """Compute all metrics for logging"""
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted', zero_division=0)
    recall = recall_score(labels, preds, average='weighted', zero_division=0)
    f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    
    # AUC requires probabilities and multi-class handling
    try:
        auc = roc_auc_score(labels, preds, multi_class='ovr', average='weighted')
    except:
        auc = 0.0
    
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }


def train_fold(model, train_loader, val_loader, fold, num_classes):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler()  # Mixed precision

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(EPOCHS):
        # --- Training ---
        model.train()
        train_loss = 0.0
        all_preds = []
        all_labels = []

        for X, y, _, _ in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            with autocast(device_type=DEVICE.type):
                logits = model(X, return_logits=True)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Collect predictions
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                train_loss += loss.item()

        train_metrics = compute_metrics(np.array(all_preds), np.array(all_labels), num_classes)
        avg_train_loss = train_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for X, y, _, _ in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                with autocast(device_type=DEVICE.type):
                    logits = model(X, return_logits=True)
                    loss = criterion(logits, y)
                    val_loss += loss.item()
                    preds = torch.argmax(logits, dim=1)
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(y.cpu().numpy())

        val_metrics = compute_metrics(np.array(val_preds), np.array(val_labels), num_classes)
        avg_val_loss = val_loss / len(val_loader)

        # Print progress
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(f"Epoch {epoch+1}/{EPOCHS}: "
                  f"Train Loss: {avg_train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}")

        # Early stopping
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1} (best val acc: {best_val_acc:.4f})")
                break

    return val_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["2class", "3class"], default="3class")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs("output", exist_ok=True)

    dataset = ADHD200DFCDataset(
        phenotype_csv=r"F:\ML_Project\common_dataset.csv", 
        fc_root_dir=r"F:\ML_Project\FC_results"
    )
    print(f"Dataset loaded: {len(dataset)} samples")

    # Convert labels if 2-class
    if args.mode == "2class":
        for idx in range(len(dataset)):
            dataset.meta.iloc[idx, dataset.meta.columns.get_loc("DX")] = 0 if dataset.meta.iloc[idx]["DX"] == 0 else 1
        num_classes = 2
        result_file = os.path.join("output", "1_2D_CNN_2class.csv")
    else:
        num_classes = 3
        result_file = os.path.join("output", "1_2D_CNN_3class.csv")

    labels = [dataset[i][1] for i in range(len(dataset))]
    labels_np = np.array(labels)
    kf = StratifiedKFold(n_splits=KFOLDS, shuffle=True, random_state=args.seed)
    all_fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(dataset)), labels_np), 1):
        print(f"\nFold {fold}/{KFOLDS}")
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        model = SimpleCNN(num_classes=num_classes, K=7, input_size=116).to(DEVICE)
        fold_metrics = train_fold(model, train_loader, val_loader, fold, num_classes)
        all_fold_metrics.append(fold_metrics)

        print(f"Fold {fold} metrics: {fold_metrics}")
        log_results(result_file, fold, fold_metrics)

    # Print overall metrics
    print("\nFINAL RESULTS")
    for m in ["accuracy", "precision", "recall", "f1", "auc"]:
        values = [fold[m] for fold in all_fold_metrics]
        print(f"{m.capitalize()}: Mean={np.mean(values):.4f}, Std={np.std(values):.4f}")

if __name__ == "__main__":
    main()
