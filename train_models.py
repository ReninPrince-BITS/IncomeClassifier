"""
Train 6 classification models on Adult dataset and save to model/.
Run once to generate model files for the Streamlit app.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)
import xgboost as xgb
import joblib

# Paths
DATA_PATH = "adult.csv"
MODEL_DIR = "model"
TARGET = "income"

# Categorical columns to encode
CAT_COLS = ["workclass", "education", "marital.status", "occupation",
            "relationship", "race", "sex", "native.country"]
NUM_COLS = ["age", "fnlwgt", "education.num", "capital.gain", "capital.loss", "hours.per.week"]


def load_and_preprocess(path=DATA_PATH):
    """Load CSV, replace ? with mode, encode categoricals, scale numerics."""
    df = pd.read_csv(path)
    # Replace missing marker
    df = df.replace("?", np.nan)
    for c in CAT_COLS:
        df[c] = df[c].fillna(df[c].mode()[0] if not df[c].mode().empty else "Unknown")
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(y)
    # Encode categoricals
    X_cat = pd.get_dummies(X[CAT_COLS], drop_first=True)
    X_num = X[NUM_COLS]
    X_combined = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)
    return X_scaled, y, scaler, le, X_combined.columns.tolist(), df


def eval_metrics(y_true, y_pred, y_proba=None):
    """Compute required metrics. y_proba for AUC (binary)."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba[:, 1]) if y_proba is not None else 0.0
    return {"Accuracy": acc, "AUC": auc, "Precision": prec, "Recall": rec, "F1": f1, "MCC": mcc}


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("Loading and preprocessing data...")
    X, y, scaler, label_enc, feature_names, df = load_and_preprocess()
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(label_enc, os.path.join(MODEL_DIR, "label_encoder.joblib"))
    joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.joblib"))

    train_idx, test_idx = train_test_split(np.arange(len(X)), test_size=0.2, random_state=42)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    df.iloc[test_idx].to_csv("test_data.csv", index=False)
    print("Test set saved to test_data.csv (upload in Streamlit for fair evaluation).")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "K-Nearest Neighbor": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": xgb.XGBClassifier(eval_metric="logloss", random_state=42),
    }

    results = []
    for name, clf in models.items():
        print(f"Training {name}...")
        clf.fit(X_train, y_train)
        joblib.dump(clf, os.path.join(MODEL_DIR, f"{name.replace(' ', '_').replace('-', '_')}.joblib"))
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test) if hasattr(clf, "predict_proba") else None
        m = eval_metrics(y_test, y_pred, y_proba)
        m["Model"] = name
        results.append(m)
        print(f"  Accuracy: {m['Accuracy']:.4f}, AUC: {m['AUC']:.4f}")

    # Save results table for README
    res_df = pd.DataFrame(results)
    res_df = res_df[["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
    res_df.to_csv(os.path.join(MODEL_DIR, "metrics_summary.csv"), index=False)
    print("\nDone. Models and metrics saved in", MODEL_DIR)


if __name__ == "__main__":
    main()
