import numpy as np
import argparse
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from collections import Counter

from data.ADHD200_DFC_Dataset import ADHD200DFCDataset
from utils import log_results

KFOLDS = 5


def majority_vote(preds_per_segment):
    """Given [K] predicted segment labels, return majority vote label."""
    counts = Counter(preds_per_segment)
    return counts.most_common(1)[0][0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["2class", "3class"], default="3class",
                        help="Classification mode: 2class (ADHD vs TDC) or 3class (ADHDI, ADHDC, TDC)")
    args = parser.parse_args()

    dataset = ADHD200DFCDataset(r"F:\ML_Project\phenotype.csv", r"F:\ML_Project\FC_results")

    # Flatten dFC segments for SVM
    X_segments = []
    y_labels = []
    sids = []

    for i in range(len(dataset)):
        x, y, sid, _ = dataset[i]
        X_segments.append(x)  # shape: [K, N, N]
        y_labels.append(y)
        sids.append(sid)

    X_segments = np.array(X_segments)  # [num_subjects, K, N, N]
    y_labels = np.array(y_labels)
    sids = np.array(sids)

    # Adjust labels for 2class mode
    if args.mode == "2class":
        y_labels = np.where(y_labels == 0, 0, 1)  # TDC=0, ADHD(any)=1
        result_file = "svm_2class.csv"
    else:
        result_file = "svm_3class.csv"

    print(f"Running SVM evaluation ({args.mode})")

    kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_segments), 1):
        X_train = X_segments[train_idx]
        y_train = y_labels[train_idx]
        X_val = X_segments[val_idx]
        y_val = y_labels[val_idx]

        # Flatten each segment for SVM input: [num_subjects*K, N*N]
        X_train_flat = X_train.reshape(-1, X_train.shape[2]*X_train.shape[3])
        y_train_rep = np.repeat(y_train, X_train.shape[1])  # repeat label for each segment

        X_val_flat = X_val.reshape(-1, X_val.shape[2]*X_val.shape[3])
        y_val_rep = np.repeat(y_val, X_val.shape[1])

        clf = SVC(kernel="rbf", C=1.0, gamma="scale")
        clf.fit(X_train_flat, y_train_rep)

        # Predict per segment
        y_val_pred_segments = clf.predict(X_val_flat).reshape(len(X_val), X_val.shape[1])

        # Majority vote per subject
        y_val_pred_subjects = np.array([majority_vote(preds) for preds in y_val_pred_segments])

        acc = accuracy_score(y_val, y_val_pred_subjects)
        fold_accuracies.append(acc)
        print(f"Fold {fold}: ACC={acc:.4f}")
        log_results(result_file, fold, acc)

    print(f"\nMean ACC={np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")


if __name__ == "__main__":
    main()
