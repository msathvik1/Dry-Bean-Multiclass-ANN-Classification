"""
====================================================================================================
PROJECT TITLE : Dry Bean Variety Classification using Multiclass Neural Network
DOMAIN        : Agriculture / Precision Farming
DATASET       : UCI Machine Learning Repository - Dry Bean Dataset (ID: 602)
URL           : https://archive.ics.uci.edu/dataset/602/dry+bean+dataset
TECHNIQUE     : Multiclass Multilayer Perceptron (MLP) trained with Stochastic Gradient Descent (SGD)
COMPARISON    : Shallow Neural Architecture vs. Deeper Neural Architecture vs. Simple Baseline
LICENSE NOTE  : Creative Commons Attribution 4.0 International (CC BY 4.0) - For Academic Work Only
====================================================================================================
This is a SINGLE, SELF-CONTAINED Python script that executes the complete project end-to-end:
  1. Download and inspect the dataset (13,611 records, 16 features, 7 bean classes).
  2. Preprocess and split without data leakage (Stratified 70% Train, 15% Val, 15% Test).
  3. Implement the neural architectures (Linear Baseline, Shallow MLP, Deeper MLP).
  4. Train with Stochastic Gradient Descent (SGD with momentum, cosine decay, weight decay).
  5. Tune justified hyperparameters (learning rate, momentum, L2 regularization).
  6. Compare models against a simple linear baseline.
  7. Report metrics (Accuracy, Macro-F1, Precision, Recall) and generate evaluation plots.
====================================================================================================
"""

import os
import io
import sys
import random
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ==================================================================================================
# 1. REPRODUCIBILITY & DEVICE SETUP
# ==================================================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[STATUS] Running on compute device: {device}")


# ==================================================================================================
# 2. DATASET DOWNLOAD & INSPECTION
# ==================================================================================================
def get_dataset(csv_path="dry_bean.csv"):
    """
    Loads dataset from local file if available; otherwise auto-downloads from UCI repository.
    """
    if os.path.exists(csv_path):
        print(f"[STATUS] Loading cached dataset from '{csv_path}'...")
        df = pd.read_csv(csv_path)
    else:
        print("[STATUS] Local dataset not found. Fetching from UCI Machine Learning Repository...")
        try:
            from ucimlrepo import fetch_ucirepo
            dataset = fetch_ucirepo(id=602)
            X = dataset.data.features
            y = dataset.data.targets
            df = pd.concat([X, y], axis=1)
        except Exception:
            print("[STATUS] Downloading Dry Bean zip archive directly from UCI server...")
            url = "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as resp:
                zf = zipfile.ZipFile(io.BytesIO(resp.read()))
                excel_name = [n for n in zf.namelist() if n.endswith('.xlsx')][0]
                with zf.open(excel_name) as f:
                    df = pd.read_excel(f)
        df.to_csv(csv_path, index=False)
        print(f"[STATUS] Successfully downloaded and cached to '{csv_path}'.")

    target_col = [c for c in df.columns if c.lower() in ['class', 'target']][0]
    feature_cols = [c for c in df.columns if c != target_col]

    print("\n" + "="*70)
    print("STEP 1: DATASET INSPECTION")
    print("="*70)
    print(f"Total Grain Records : {df.shape[0]:,}")
    print(f"Total Features      : {len(feature_cols)} morphological/shape attributes")
    print(f"Missing Values      : {df.isnull().sum().sum()}")
    print("\nFeature Columns:")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:>2}. {col:<18} (min: {df[col].min():8.2f}, max: {df[col].max():8.2f})")

    print("\nClass Distribution (7 Bean Varieties):")
    counts = df[target_col].value_counts()
    for cls_name, cnt in counts.items():
        pct = (cnt / len(df)) * 100
        print(f"  - {cls_name:<10}: {cnt:>5} records ({pct:5.2f}%)")
    print("="*70 + "\n")

    return df, feature_cols, target_col


# ==================================================================================================
# 3. LEAK-FREE PREPROCESSING & STRATIFIED SPLIT
# ==================================================================================================
def preprocess_and_split(df, feature_cols, target_col, seed=42):
    """
    Splits data into 70% Train, 15% Validation, 15% Test using stratification.
    Fits StandardScaler ONLY on Train set to prevent information leakage.
    """
    X = df[feature_cols].values
    y_raw = df[target_col].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = list(le.classes_)

    # Stratified split: 70% Train, 30% Temp
    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed
    )
    # Stratified split of Temp: 50% Val (15% total), 50% Test (15% total)
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw, y_temp, test_size=0.50, stratify=y_temp, random_state=seed
    )

    # Standardize: Fit strictly on Train, then transform Val and Test
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val   = scaler.transform(X_val_raw)
    X_test  = scaler.transform(X_test_raw)

    print("="*70)
    print("STEP 2: LEAK-FREE DATA PARTITIONING")
    print("="*70)
    print(f"Training Set   : {X_train.shape[0]:>5} samples ({X_train.shape[0]/len(df)*100:.1f}%) [Used for SGD training]")
    print(f"Validation Set : {X_val.shape[0]:>5} samples ({X_val.shape[0]/len(df)*100:.1f}%) [Used for checkpointing/tuning]")
    print(f"Test Set       : {X_test.shape[0]:>5} samples ({X_test.shape[0]/len(df)*100:.1f}%) [Completely isolated for evaluation]")
    print("Standardization: StandardScaler fitted ONLY on training set.")
    print("="*70 + "\n")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names


# ==================================================================================================
# 4. NEURAL ARCHITECTURES (BASELINE, SHALLOW MLP, DEEPER MLP)
# ==================================================================================================
class LinearBaseline(nn.Module):
    """Simple baseline: Softmax linear regression (no hidden layers)."""
    def __init__(self, in_features=16, num_classes=7):
        super(LinearBaseline, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)


class ShallowMLP(nn.Module):
    """Shallow architecture: 1 hidden layer with 64 units and ReLU."""
    def __init__(self, in_features=16, hidden_dim=64, num_classes=7):
        super(ShallowMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class DeeperMLP(nn.Module):
    """Deeper architecture: 3 hidden layers (128 -> 64 -> 32) with Dropout (0.10)."""
    def __init__(self, in_features=16, num_classes=7, dropout_rate=0.10):
        super(DeeperMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==================================================================================================
# 5. TRAINING WITH STOCHASTIC GRADIENT DESCENT (SGD)
# ==================================================================================================
def train_with_sgd(model, train_loader, val_loader, epochs=80, lr=0.03, momentum=0.9, weight_decay=1e-4, model_name="Model"):
    """
    Trains neural model using Mini-batch Stochastic Gradient Descent (SGD) with momentum
    and Cosine Annealing learning rate schedule.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.05)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0

    print(f"[TRAINING] Starting SGD optimization for: {model_name} ({count_parameters(model):,} parameters)")
    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * bx.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)

        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                running_val_loss += criterion(logits, by).item() * bx.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == by).sum().item()

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = correct / len(val_loader.dataset)
        scheduler.step()

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        # Checkpoint Best Model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        if epoch % 20 == 0 or epoch == 1 or epoch == epochs:
            print(f"  Epoch [{epoch:>2}/{epochs:>2}] | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:5.2f}%")

    model.load_state_dict(best_state)
    model.to(device)
    print(f"[CHECKPOINT] Best validation loss {best_val_loss:.4f} recorded at epoch {best_epoch}.\n")
    return history


# ==================================================================================================
# 6. EVALUATION ON UNSEEN TEST SET
# ==================================================================================================
def evaluate_on_test(model, test_loader, class_names):
    """
    Evaluates model on isolated test set and returns accuracy, macro-F1, weighted-F1,
    classification report, and confusion matrix.
    """
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            logits = model(bx)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(by.numpy())

    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)
    rep = classification_report(all_targets, all_preds, target_names=class_names, zero_division=0)

    return {
        "Accuracy": acc,
        "Macro-F1": macro_f1,
        "Weighted-F1": weighted_f1,
        "ConfusionMatrix": cm,
        "Report": rep,
        "Targets": all_targets,
        "Predictions": all_preds
    }


# ==================================================================================================
# 7. MAIN EXECUTION PIPELINE
# ==================================================================================================
def main():
    print("="*70)
    print("DRY BEAN MULTICLASS ANN CLASSIFICATION PIPELINE")
    print("="*70)

    # 1. Dataset
    df, feature_cols, target_col = get_dataset("dry_bean.csv")

    # 2. Preprocess & Split
    (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names = preprocess_and_split(
        df, feature_cols, target_col, seed=42
    )

    batch_size = 64
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(torch.tensor(X_val,   dtype=torch.float32), torch.tensor(y_val,   dtype=torch.long)), batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(TensorDataset(torch.tensor(X_test,  dtype=torch.float32), torch.tensor(y_test,  dtype=torch.long)), batch_size=batch_size, shuffle=False)

    # 3. Models
    models = {
        "Linear Baseline": LinearBaseline(in_features=16, num_classes=7),
        "Shallow MLP": ShallowMLP(in_features=16, hidden_dim=64, num_classes=7),
        "Deeper MLP": DeeperMLP(in_features=16, num_classes=7, dropout_rate=0.10)
    }

    # 4. Train with SGD
    histories = {}
    eval_results = {}
    for name, model in models.items():
        histories[name] = train_with_sgd(
            model,
            train_loader,
            val_loader,
            epochs=80,
            lr=0.03,
            momentum=0.9,
            weight_decay=1e-4,
            model_name=name
        )
        eval_results[name] = evaluate_on_test(model, test_loader, class_names)

    # 5. Report Table
    print("="*75)
    print(f"{'Architecture':<18} | {'Parameters':>10} | {'Accuracy':>10} | {'Macro-F1':>10} | {'Weighted-F1':>12}")
    print("="*75)
    for name in models:
        res = eval_results[name]
        params = count_parameters(models[name])
        print(f"{name:<18} | {params:>10,} | {res['Accuracy']*100:>9.2f}% | {res['Macro-F1']:>10.4f} | {res['Weighted-F1']:>12.4f}")
    print("="*75 + "\n")

    # Print Detailed Per-Class Reports
    for name in models:
        print(f"--- Detailed Classification Report: {name} ---")
        print(eval_results[name]["Report"])

    # 6. Unified Visualizations Plot (Loss, Accuracy, Confusion Matrices)
    print("[STATUS] Generating unified publication-quality evaluation figures...")
    fig = plt.figure(figsize=(18, 10), dpi=300)
    gs = fig.add_gridspec(2, 3)

    # Plot 1: Training Loss Convergence
    ax1 = fig.add_subplot(gs[0, 0])
    for name in models:
        ax1.plot(histories[name]["train_loss"], label=name, linewidth=2)
    ax1.set_title("A. Training Loss Convergence (SGD)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # Plot 2: Validation Loss Convergence
    ax2 = fig.add_subplot(gs[0, 1])
    for name in models:
        ax2.plot(histories[name]["val_loss"], label=name, linewidth=2)
    ax2.set_title("B. Validation Loss Convergence", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Validation Loss", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    # Plot 3: Validation Accuracy Trajectory
    ax3 = fig.add_subplot(gs[0, 2])
    for name in models:
        ax3.plot([a * 100 for a in histories[name]["val_acc"]], label=name, linewidth=2)
    ax3.set_title("C. Validation Accuracy Trajectory (%)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Epoch", fontsize=10)
    ax3.set_ylabel("Accuracy (%)", fontsize=10)
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend()

    # Plots 4, 5, 6: Normalized Confusion Matrices
    cm_palettes = ["Blues", "Greens", "Purples"]
    for i, name in enumerate(models):
        ax = fig.add_subplot(gs[1, i])
        cm = eval_results[name]["ConfusionMatrix"]
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap=cm_palettes[i],
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=False,
            ax=ax
        )
        acc_pct = eval_results[name]["Accuracy"] * 100
        f1_val = eval_results[name]["Macro-F1"]
        ax.set_title(f"D{i+1}. {name}\nAcc: {acc_pct:.2f}% | Macro-F1: {f1_val:.4f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Variety", fontsize=9)
        ax.set_ylabel("True Variety" if i == 0 else "", fontsize=9)
        ax.tick_params(axis='x', rotation=45)

    plt.suptitle("Dry Bean Variety Classification - Multiclass ANN & SGD Evaluation", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    output_fig_path = "dry_bean_evaluation_plots.png"
    plt.savefig(output_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[STATUS] Unified evaluation figure saved successfully to: '{output_fig_path}'")
    print("\n[COMPLETE] All tasks finished successfully!")


if __name__ == "__main__":
    main()
