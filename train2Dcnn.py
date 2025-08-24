
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import os
from torch.cuda.amp import autocast, GradScaler  # Added for mixed precision training
from models.simple_CNN import SimpleCNN
from data.ADHD200_DFC_Dataset import ADHD200DFCDataset
from models.skip_vote_net import SkipVoteNet
from utils import set_seed, log_results

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


def train_fold(model, train_loader, val_loader, fold, num_classes):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler()  # Added for mixed precision training

    best_val_acc = 0.0
    patience_counter = 0

    print(f"Training fold {fold} with {len(train_loader)} train batches, {len(val_loader)} val batches")

    for epoch in range(EPOCHS):
        # --- Training ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (X, y, _, _) in enumerate(train_loader):
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            
            # Use mixed precision training
            with autocast():
                logits = model(X, return_logits=True)  # [batch, num_classes]
                loss = criterion(logits, y)
            
            scaler.scale(loss).backward()  # Scale loss for mixed precision
            scaler.step(optimizer)
            scaler.update()
            
            # Calculate training accuracy
            with torch.no_grad():
                _, predicted = torch.max(logits, 1)
                train_correct += (predicted == y).sum().item()
                train_total += y.size(0)
                train_loss += loss.item()

        train_acc = train_correct / train_total if train_total > 0 else 0.0
        avg_train_loss = train_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for X, y, _, _ in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                
                # Use mixed precision for validation
                with autocast():
                    predictions = model(X, return_logits=False)  # [batch] - majority vote
                    val_correct += (predictions == y).sum().item()
                    val_total += y.size(0)
                    
                    # Compute validation loss
                    logits = model(X, return_logits=True)
                    loss = criterion(logits, y)
                    val_loss += loss.item()

        val_acc = val_correct / val_total if val_total > 0 else 0.0
        avg_val_loss = val_loss / len(val_loader)
        
        # Print progress every 10 epochs or on first/last epoch
        if epoch % 10 == 0 or epoch == EPOCHS - 1 or patience_counter >= PATIENCE - 1:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS}: "
                  f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Early stopping based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1} (best val acc: {best_val_acc:.4f})")
                break

    return best_val_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["2class", "3class"], default="3class",
                        help="Classification mode: 2class (ADHD vs TDC) or 3class (ADHDI, ADHDC, TDC)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Set random seed for reproducibility
    set_seed(args.seed)
    
    print(f"Starting training in {args.mode} mode")
    print(f"Device: {DEVICE}")
    # Added: Print detailed CUDA information
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    else:
        print("Warning: CUDA not available, using CPU")
    print(f"Random seed: {args.seed}")
    print("-" * 50)

    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Load dataset
    dataset = ADHD200DFCDataset(
        phenotype_csv=r"F:\ML_Project\common_dataset.csv", 
        fc_root_dir=r"F:\ML_Project\FC_results"
    )
    print(f"Dataset loaded: {len(dataset)} samples")

    # ---- Prepare labels based on mode ----
    if args.mode == "2class":
        print("Converting to 2-class problem (TDC vs ADHD)")
        # Convert labels: TDC=0 stays 0, ADHDC=1 and ADHDI=2 become 1
        for idx in range(len(dataset)):
            current_dx = dataset.meta.iloc[idx]["DX"]
            new_dx = 0 if current_dx == 0 else 1
            dataset.meta.iloc[idx, dataset.meta.columns.get_loc("DX")] = new_dx
        
        num_classes = 2
        result_file = os.path.join("output", "1_2D_CNN_2class.csv")
    else:
        print("Using 3-class problem (TDC vs ADHDC vs ADHDI)")
        num_classes = 3
        result_file = os.path.join("output", "1_2D_CNN_3class.csv")

    # Check label distribution
    labels = [dataset[i][1] for i in range(len(dataset))]
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_dist = dict(zip(unique_labels, counts))
    print(f"Label distribution: {label_dist}")
    
    if args.mode == "2class":
        print(f"  Class 0 (TDC): {label_dist.get(0, 0)} samples")
        print(f"  Class 1 (ADHD): {label_dist.get(1, 0)} samples")
    else:
        print(f"  Class 0 (TDC): {label_dist.get(0, 0)} samples")
        print(f"  Class 1 (ADHDC): {label_dist.get(1, 0)} samples") 
        print(f"  Class 2 (ADHDI): {label_dist.get(2, 0)} samples")

    # K-Fold Cross Validation
    kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=args.seed)
    all_fold_accuracies = []

    print(f"\nStarting {KFOLDS}-fold cross validation")
    print("=" * 60)

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(dataset))), 1):
        print(f"\nFold {fold}/{KFOLDS}")
        print("-" * 30)
        
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        print(f"Train samples: {len(train_subset)}, Val samples: {len(val_subset)}")

        # Initialize model for this fold
        
        model = SimpleCNN(num_classes=num_classes, K=7, input_size=116).to(DEVICE)
        
        # Added: Confirm model is on GPU
        print(f"Model device: {next(model.parameters()).device}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Train the fold
        fold_acc = train_fold(model, train_loader, val_loader, fold, num_classes)
        all_fold_accuracies.append(fold_acc)
        
        print(f"Fold {fold} final validation accuracy: {fold_acc:.4f}")

        # Log results
        log_results(result_file, fold, fold_acc)

    # Print final results
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS ({args.mode.upper()})")
    print(f"{'='*60}")
    print(f"Fold accuracies: {[f'{acc:.4f}' for acc in all_fold_accuracies]}")
    print(f"Mean accuracy: {np.mean(all_fold_accuracies):.4f}")
    print(f"Std accuracy: {np.std(all_fold_accuracies):.4f}")
    print(f"Results saved to: {result_file}")
    print(f"Device used: {DEVICE}")


if __name__ == "__main__":
    main()