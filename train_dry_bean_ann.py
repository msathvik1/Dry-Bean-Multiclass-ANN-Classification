"""
=================================================================================
Project: Dry Bean Variety Classification using Multiclass Neural Network
Domain : Agriculture
Dataset: UCI Dry Bean Dataset (ID 602, Koklu & Ozkan, 2020)
Target : 7 Bean Varieties (BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA)
Features: 16 Morphological and Shape Features
Technique: Multiclass Multilayer Perceptron (MLP) trained via Stochastic Gradient Descent (SGD)
Architectures Compared: Shallow MLP vs Deeper MLP vs Linear Baseline
=================================================================================
"""

import os
import sys
import argparse
import random
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
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# -------------------------------------------------------------------------
# 1. Reproducibility & Device Configuration
# -------------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------------------------
# 2. Data Loading & Inspection
# -------------------------------------------------------------------------
def load_and_inspect_dataset(data_path="dry_bean.csv"):
    """
    Loads the UCI Dry Bean dataset either from local CSV or via ucimlrepo.
    Performs data integrity verification.
    """
    if os.path.exists(data_path):
        print(f"[INFO] Loading dataset from local cache: {data_path}")
        df = pd.read_csv(data_path)
    else:
        print("[INFO] Local file not found. Fetching from UCI Machine Learning Repository (ID 602)...")
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=602)
        X = dataset.data.features
        y = dataset.data.targets
        df = pd.concat([X, y], axis=1)
        df.to_csv(data_path, index=False)
        print(f"[INFO] Dataset fetched and cached locally to: {data_path}")

    # Identify target column name ('Class' in standard UCI dry bean dataset)
    target_col = [col for col in df.columns if col.lower() in ['class', 'target']][0]
    feature_cols = [col for col in df.columns if col != target_col]

    print("\n" + "="*60)
    print("DATASET INSPECTION SUMMARY")
    print("="*60)
    print(f"Total Samples  : {len(df):,}")
    print(f"Total Features : {len(feature_cols)} (Morphological & Shape)")
    print(f"Missing Values : {df.isnull().sum().sum()}")
    print(f"Feature List   : {feature_cols}")
    print("\nClass Distribution:")
    class_counts = df[target_col].value_counts()
    for cls_name, count in class_counts.items():
        pct = (count / len(df)) * 100
        print(f"  - {cls_name:<10}: {count:>5} grains ({pct:5.2f}%)")
    print("="*60 + "\n")

    return df, feature_cols, target_col


# -------------------------------------------------------------------------
# 3. Leak-Free Preprocessing & Stratified Splitting
# -------------------------------------------------------------------------
def prepare_data(df, feature_cols, target_col, train_size=0.70, val_size=0.15, test_size=0.15, seed=42):
    """
    Splits dataset into Train (70%), Validation (15%), and Test (15%) with stratification.
    Standardizes features by fitting ONLY on the Training set to strictly eliminate data leakage.
    """
    X = df[feature_cols].values
    y_raw = df[target_col].values

    # Encode string class names into integer labels [0 .. 6]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    class_names = list(label_encoder.classes_)

    # Step 1: Split into Train (70%) and Temp (30%)
    temp_size = val_size + test_size
    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        X, y,
        test_size=temp_size,
        stratify=y,
        random_state=seed
    )

    # Step 2: Split Temp into Validation (50% of 30% = 15%) and Test (50% of 30% = 15%)
    val_ratio_of_temp = val_size / temp_size
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw, y_temp,
        train_size=val_ratio_of_temp,
        stratify=y_temp,
        random_state=seed
    )

    # Step 3: Zero-Leakage Standardization
    # Standardize features using mu and sigma computed EXCLUSIVELY from X_train
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    print(f"[SPLIT] Training Set   : {X_train.shape[0]:>5} samples ({X_train.shape[0]/len(df)*100:.1f}%)")
    print(f"[SPLIT] Validation Set : {X_val.shape[0]:>5} samples ({X_val.shape[0]/len(df)*100:.1f}%)")
    print(f"[SPLIT] Test Set       : {X_test.shape[0]:>5} samples ({X_test.shape[0]/len(df)*100:.1f}%)")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names, scaler


# -------------------------------------------------------------------------
# 4. Neural Network Architectures
# -------------------------------------------------------------------------
class LinearBaseline(nn.Module):
    """
    Linear Multinomial Classifier (Softmax Regression Baseline).
    16 inputs -> 7 class logits. No hidden non-linear layers.
    """
    def __init__(self, in_features=16, num_classes=7):
        super(LinearBaseline, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)


class ShallowMLP(nn.Module):
    """
    Shallow Multiclass MLP:
    16 inputs -> Hidden(64) -> ReLU -> Output(7)
    """
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
    """
    Deeper Multiclass MLP:
    16 inputs -> Hidden(128) -> ReLU -> Dropout(0.1)
              -> Hidden(64)  -> ReLU -> Dropout(0.1)
              -> Hidden(32)  -> ReLU
              -> Output(7)
    """
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


# -------------------------------------------------------------------------
# 5. Training with Stochastic Gradient Descent (SGD)
# -------------------------------------------------------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, epochs=80, model_name="Model"):
    """
    Trains the given neural architecture using Stochastic Gradient Descent (SGD).
    Tracks convergence (training loss, validation loss, validation accuracy).
    Checkpoints the model weights achieving the lowest validation loss.
    """
    model.to(device)
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": []
    }

    best_val_loss = float("inf")
    best_state_dict = None
    best_epoch = 0

    print(f"\n--- Training {model_name} with Stochastic Gradient Descent (SGD) ---")
    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_train_loss = 0.0
        train_samples = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * batch_x.size(0)
            train_samples += batch_x.size(0)

        epoch_train_loss = running_train_loss / train_samples

        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        val_samples = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)

                running_val_loss += loss.item() * batch_x.size(0)
                val_samples += batch_x.size(0)

                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(batch_y.cpu().numpy())

        epoch_val_loss = running_val_loss / val_samples
        epoch_val_acc = accuracy_score(all_targets, all_preds)
        epoch_val_macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        history["val_macro_f1"].append(epoch_val_macro_f1)

        # Checkpoint Best Model Weights
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        if epoch % 10 == 0 or epoch == epochs or epoch == 1:
            print(f"Epoch [{epoch:>3}/{epochs:>3}] | Train Loss: {epoch_train_loss:.4f} | "
                  f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:5.2f}% | "
                  f"Val Macro-F1: {epoch_val_macro_f1:.4f}")

    print(f"[BEST] {model_name} achieved lowest Val Loss {best_val_loss:.4f} at epoch {best_epoch}.")
    model.load_state_dict(best_state_dict)
    model.to(device)
    return history, best_epoch


# -------------------------------------------------------------------------
# 6. Evaluation Function
# -------------------------------------------------------------------------
def evaluate_model(model, test_loader, criterion, class_names):
    """
    Evaluates model on the untouched test partition.
    Returns test loss, accuracy, macro-F1, weighted-F1, per-class report, and confusion matrix.
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            running_loss += loss.item() * batch_x.size(0)
            total_samples += batch_x.size(0)

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(batch_y.cpu().numpy())

    test_loss = running_loss / total_samples
    accuracy = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)
    macro_precision = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    macro_recall = recall_score(all_targets, all_preds, average="macro", zero_division=0)

    report_dict = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True, zero_division=0)
    report_text = classification_report(all_targets, all_preds, target_names=class_names, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)

    metrics = {
        "Test Loss": test_loss,
        "Accuracy": accuracy,
        "Macro-F1": macro_f1,
        "Weighted-F1": weighted_f1,
        "Macro-Precision": macro_precision,
        "Macro-Recall": macro_recall,
        "report_dict": report_dict,
        "report_text": report_text,
        "confusion_matrix": cm,
        "predictions": all_preds,
        "targets": all_targets
    }
    return metrics


# -------------------------------------------------------------------------
# 7. Visualization Suite
# -------------------------------------------------------------------------
def plot_convergence(histories, output_path="results/convergence_comparison.png"):
    """
    Plots training and validation loss convergence curves for Shallow and Deeper MLPs across epochs.
    """
    plt.figure(figsize=(12, 5), dpi=300)
    epochs_range = range(1, len(histories["Shallow MLP"]["train_loss"]) + 1)

    # Subplot 1: Training Loss Comparison
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, histories["Linear Baseline"]["train_loss"], label="Linear Baseline", color="#7f7f7f", linestyle=":")
    plt.plot(epochs_range, histories["Shallow MLP"]["train_loss"], label="Shallow MLP (Train)", color="#1f77b4", linewidth=2)
    plt.plot(epochs_range, histories["Deeper MLP"]["train_loss"], label="Deeper MLP (Train)", color="#d62728", linewidth=2)
    plt.title("SGD Training Loss Convergence", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Cross-Entropy Loss", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True)

    # Subplot 2: Validation Loss Comparison
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, histories["Linear Baseline"]["val_loss"], label="Linear Baseline", color="#7f7f7f", linestyle=":")
    plt.plot(epochs_range, histories["Shallow MLP"]["val_loss"], label="Shallow MLP (Val)", color="#1f77b4", linewidth=2)
    plt.plot(epochs_range, histories["Deeper MLP"]["val_loss"], label="Deeper MLP (Val)", color="#d62728", linewidth=2)
    plt.title("SGD Validation Loss Convergence", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Validation Cross-Entropy Loss", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[PLOT] Convergence plot saved to: {output_path}")


def plot_accuracy_curves(histories, output_path="results/accuracy_comparison.png"):
    """
    Plots validation accuracy trajectory over epochs for all models.
    """
    plt.figure(figsize=(8, 5), dpi=300)
    epochs_range = range(1, len(histories["Shallow MLP"]["val_acc"]) + 1)

    plt.plot(epochs_range, [acc * 100 for acc in histories["Linear Baseline"]["val_acc"]],
             label="Linear Baseline", color="#7f7f7f", linestyle="--")
    plt.plot(epochs_range, [acc * 100 for acc in histories["Shallow MLP"]["val_acc"]],
             label="Shallow MLP", color="#1f77b4", linewidth=2.2)
    plt.plot(epochs_range, [acc * 100 for acc in histories["Deeper MLP"]["val_acc"]],
             label="Deeper MLP", color="#d62728", linewidth=2.2)

    plt.title("Validation Accuracy Trajectory over Epochs (SGD Optimization)", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Validation Accuracy (%)", fontsize=11)
    plt.ylim(85, 95)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[PLOT] Accuracy trajectory plot saved to: {output_path}")


def plot_confusion_matrices(eval_results, class_names, output_path="results/confusion_matrices.png"):
    """
    Generates side-by-side normalized confusion matrices for Baseline, Shallow MLP, and Deeper MLP.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    models = ["Linear Baseline", "Shallow MLP", "Deeper MLP"]

    for idx, model_name in enumerate(models):
        cm = eval_results[model_name]["confusion_matrix"]
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues" if idx == 1 else ("Reds" if idx == 2 else "Greys"),
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=False,
            ax=axes[idx]
        )
        acc = eval_results[model_name]["Accuracy"] * 100
        f1 = eval_results[model_name]["Macro-F1"]
        axes[idx].set_title(f"{model_name}\nAcc: {acc:.2f}% | Macro-F1: {f1:.4f}", fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Predicted Bean Variety", fontsize=10)
        axes[idx].set_ylabel("True Bean Variety" if idx == 0 else "", fontsize=10)
        axes[idx].tick_params(axis='x', rotation=45)

    plt.suptitle("Normalized Confusion Matrices on Test Set (N = 2,042)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Confusion matrices saved to: {output_path}")


# -------------------------------------------------------------------------
# 8. Main Pipeline Execution
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Dry Bean Variety Classification with Multiclass ANN & SGD")
    parser.add_argument("--data_path", type=str, default="dry_bean.csv", help="Path to dry_bean.csv")
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Mini-batch size for SGD")
    parser.add_argument("--lr", type=float, default=0.03, help="SGD Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD Momentum factor")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="L2 regularization (weight decay)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--outdir", type=str, default="results", help="Directory to save artifacts and plots")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)

    # 1. Ingest & Inspect
    df, feature_cols, target_col = load_and_inspect_dataset(args.data_path)

    # 2. Leakage-free Split and Standardization
    (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names, scaler = prepare_data(
        df, feature_cols, target_col, train_size=0.70, val_size=0.15, test_size=0.15, seed=args.seed
    )

    # 3. Create PyTorch Datasets & DataLoaders
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds   = TensorDataset(torch.tensor(X_val,   dtype=torch.float32), torch.tensor(y_val,   dtype=torch.long))
    test_ds  = TensorDataset(torch.tensor(X_test,  dtype=torch.float32), torch.tensor(y_test,  dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

    num_features = len(feature_cols)
    num_classes  = len(class_names)

    # 4. Instantiate Models
    models = {
        "Linear Baseline": LinearBaseline(in_features=num_features, num_classes=num_classes),
        "Shallow MLP": ShallowMLP(in_features=num_features, hidden_dim=64, num_classes=num_classes),
        "Deeper MLP": DeeperMLP(in_features=num_features, num_classes=num_classes, dropout_rate=0.10)
    }

    print("\n" + "="*60)
    print("NEURAL NETWORK ARCHITECTURE COMPLEXITY")
    print("="*60)
    for name, model in models.items():
        print(f"Model: {name:<18} | Learnable Parameters: {count_parameters(model):>6,}")
    print("="*60)

    # 5. Train each model with Stochastic Gradient Descent (SGD)
    criterion = nn.CrossEntropyLoss()
    histories = {}
    eval_results = {}

    for name, model in models.items():
        # Setup Stochastic Gradient Descent (SGD) optimizer
        # Justified hyperparameters:
        # lr=0.03 (avoids divergence, sufficient step size for normalized inputs)
        # momentum=0.9 (dampens oscillations, accelerates in consistent gradient directions)
        # weight_decay=1e-4 (L2 penalty prevents weight explosion)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)

        history, best_epoch = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler=scheduler,
            epochs=args.epochs,
            model_name=name
        )
        histories[name] = history

        # Evaluate on Test Set
        metrics = evaluate_model(model, test_loader, criterion, class_names)
        eval_results[name] = metrics

    # 6. Generate Plots
    print("\n[INFO] Generating publication-quality evaluation figures...")
    plot_convergence(histories, output_path=os.path.join(args.outdir, "convergence_comparison.png"))
    plot_accuracy_curves(histories, output_path=os.path.join(args.outdir, "accuracy_comparison.png"))
    plot_confusion_matrices(eval_results, class_names, output_path=os.path.join(args.outdir, "confusion_matrices.png"))

    # 7. Summary Metrics Table & Reports
    summary_data = []
    for name, res in eval_results.items():
        summary_data.append({
            "Architecture": name,
            "Parameters": count_parameters(models[name]),
            "Test Accuracy (%)": round(res["Accuracy"] * 100, 2),
            "Macro-F1": round(res["Macro-F1"], 4),
            "Weighted-F1": round(res["Weighted-F1"], 4),
            "Macro-Precision": round(res["Macro-Precision"], 4),
            "Macro-Recall": round(res["Macro-Recall"], 4),
            "Test Loss": round(res["Test Loss"], 4)
        })

    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = os.path.join(args.outdir, "metrics_summary_table.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\n[INFO] Summary metrics table saved to: {summary_csv_path}")

    print("\n" + "="*80)
    print("EXPERIMENTAL EVALUATION SUMMARY (TEST SET)")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)

    # Detailed Per-Class Classification Reports
    report_txt_path = os.path.join(args.outdir, "classification_reports.txt")
    with open(report_txt_path, "w") as f:
        f.write("="*80 + "\n")
        f.write("DRY BEAN MULTICLASS CLASSIFICATION - DETAILED REPORTS\n")
        f.write("="*80 + "\n\n")
        for name, res in eval_results.items():
            header = f"--- {name} (Parameters: {count_parameters(models[name]):,}) ---"
            print("\n" + header)
            print(res["report_text"])
            f.write(header + "\n")
            f.write(res["report_text"] + "\n\n")
    print(f"[INFO] Classification reports saved to: {report_txt_path}")


if __name__ == "__main__":
    main()
