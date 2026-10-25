"""
=================================================================================
Project: Dry Bean Variety Classification using Multiclass Neural Network
Alternative Implementation: Pure Scikit-Learn MLPClassifier with SGD Solver
Domain : Agriculture
Dataset: UCI Dry Bean Dataset (ID 602)
=================================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

def main():
    data_path = "dry_bean.csv"
    if not os.path.exists(data_path):
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=602)
        X_df = dataset.data.features
        y_df = dataset.data.targets
        df = pd.concat([X_df, y_df], axis=1)
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)

    target_col = [col for col in df.columns if col.lower() in ['class', 'target']][0]
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].values
    y_raw = df[target_col].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = list(le.classes_)

    # Stratified 70/15/15 Split
    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    # Zero-Leakage Standardization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    print(f"Data ready: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")

    # 1. Baseline: Logistic Regression (Multinomial Softmax)
    baseline = LogisticRegression(solver='saga', max_iter=300, random_state=42)
    baseline.fit(X_train, y_train)

    # 2. Shallow MLP: 1 hidden layer with 64 units, SGD solver
    shallow_mlp = MLPClassifier(
        hidden_layer_sizes=(64,),
        activation='relu',
        solver='sgd',
        learning_rate_init=0.03,
        momentum=0.9,
        alpha=1e-4, # L2 regularization
        max_iter=100,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15
    )
    shallow_mlp.fit(X_train, y_train)

    # 3. Deeper MLP: 3 hidden layers (128, 64, 32), SGD solver
    deeper_mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='sgd',
        learning_rate_init=0.03,
        momentum=0.9,
        alpha=1e-4, # L2 regularization
        max_iter=100,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15
    )
    deeper_mlp.fit(X_train, y_train)

    # Evaluation
    models = {
        "Logistic Regression Baseline": baseline,
        "Shallow MLP (64)": shallow_mlp,
        "Deeper MLP (128-64-32)": deeper_mlp
    }

    print("\n" + "="*70)
    print("SCIKIT-LEARN EVALUATION RESULTS (TEST SET)")
    print("="*70)
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        print(f"\nModel: {name}")
        print(f"Accuracy: {acc*100:.2f}% | Macro-F1: {macro_f1:.4f}")
        print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

if __name__ == "__main__":
    main()
