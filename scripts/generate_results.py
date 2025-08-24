import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = r"F:\ML_Project\output"
SUMMARY_FILE = os.path.join(RESULTS_DIR, "summary.csv")


def load_results(file):
    """Load per-fold results CSV and compute mean ± std."""
    df = pd.read_csv(file)
    mean_acc = df["Accuracy"].mean()
    std_acc = df["Accuracy"].std()
    fold_acc = df["Accuracy"].values
    return mean_acc, std_acc, fold_acc


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Expected files (ensure train.py + eval.py write here)
    result_files = {
        "Skip-Vote-Net (2class)": os.path.join(RESULTS_DIR, "skip_vote_net_2class.csv"),
        "Skip-Vote-Net (3class)": os.path.join(RESULTS_DIR, "skip_vote_net_3class.csv"),
        "SVM (2class)": os.path.join(RESULTS_DIR, "svm_2class.csv"),
        "SVM (3class)": os.path.join(RESULTS_DIR, "svm_3class.csv"),
    }

    summary = []
    foldwise_data = {}

    for model, file in result_files.items():
        if os.path.exists(file):
            mean_acc, std_acc, fold_acc = load_results(file)
            summary.append({"Model": model, "MeanACC": mean_acc, "StdACC": std_acc})
            foldwise_data[model] = fold_acc
            print(f"{model}: {mean_acc:.4f} ± {std_acc:.4f}")
        else:
            print(f"⚠️ Missing {file}, skipping...")

    # Save summary CSV
    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(SUMMARY_FILE, index=False)
    print(f"\nSaved summary results → {SUMMARY_FILE}")

    # Plot mean ± std ACC
    if not df_summary.empty:
        plt.figure(figsize=(8, 5))
        x = range(len(df_summary))
        plt.bar(x, df_summary["MeanACC"], yerr=df_summary["StdACC"], capsize=5)
        plt.xticks(x, df_summary["Model"], rotation=20, ha="right")
        plt.ylabel("Accuracy")
        plt.title("Skip-Vote-Net vs SVM (2class & 3class)")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "summary_plot.png"))
        plt.show()
        print(f"📊 Saved plot → results/summary_plot.png")

    # --- Fold-wise plots with shaded error bars ---
    if foldwise_data:
        plt.figure(figsize=(10, 6))
        for model, folds in foldwise_data.items():
            folds = np.array(folds)
            mean = np.mean(folds)
            std = np.std(folds)
            x = np.arange(1, len(folds)+1)
            plt.plot(x, folds, marker='o', label=f"{model} (mean={mean:.2f})")
            plt.fill_between(x, folds - std, folds + std, alpha=0.2)  # shaded region for std
        plt.xlabel("Fold")
        plt.ylabel("Accuracy")
        plt.title("Fold-wise Accuracy per Model (with Std Shading)")
        plt.xticks(np.arange(1, len(folds)+1))
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        foldwise_plot_file = os.path.join(RESULTS_DIR, "foldwise_accuracy_shaded.png")
        plt.savefig(foldwise_plot_file)
        plt.show()
        print(f"📊 Saved fold-wise shaded plot → {foldwise_plot_file}")


if __name__ == "__main__":
    main()
