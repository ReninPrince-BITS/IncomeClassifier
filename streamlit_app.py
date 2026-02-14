"""
Streamlit app: upload CSV (test data), pick a model, view metrics, plots, and predictions.
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report,
    roc_curve
)

MODEL_DIR = "model"
TARGET = "income"
CAT_COLS = ["workclass", "education", "marital.status", "occupation",
            "relationship", "race", "sex", "native.country"]
NUM_COLS = ["age", "fnlwgt", "education.num", "capital.gain", "capital.loss", "hours.per.week"]


def load_artifacts():
    """Load scaler, label encoder, feature names from model/."""
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    label_enc = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.joblib"))
    return scaler, label_enc, feature_names


def preprocess_df(df, scaler, feature_names):
    """Match training preprocessing: fill ?, encode cats, scale."""
    df = df.copy()
    if TARGET in df.columns:
        df = df.drop(columns=[TARGET])
    df = df.replace("?", np.nan)
    for c in CAT_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].mode()[0] if not df[c].mode().empty else "Unknown")
    X_cat = pd.get_dummies(df[CAT_COLS], drop_first=True)
    # Align columns with training (missing cols = 0)
    for f in feature_names:
        if f not in X_cat.columns and f not in NUM_COLS:
            X_cat[f] = 0
    X_num = df[NUM_COLS]
    X_combined = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    X_combined = X_combined.reindex(columns=feature_names, fill_value=0)
    return scaler.transform(X_combined)


def get_model_names():
    """List saved .joblib files (exclude scaler, encoder, feature_names)."""
    skip = {"scaler.joblib", "label_encoder.joblib", "feature_names.joblib", "metrics_summary.csv"}
    names = []
    for f in os.listdir(MODEL_DIR):
        if f.endswith(".joblib") and f not in skip:
            names.append(f.replace(".joblib", "").replace("_", " "))
    return sorted(names)


def load_model(name):
    """Load model by display name (map back to filename)."""
    fname = name.replace(" ", "_").replace("-", "_") + ".joblib"
    return joblib.load(os.path.join(MODEL_DIR, fname))


st.set_page_config(page_title="ML Assignment 2 - Classification", layout="wide")
st.title("Classification Models - Adult Income Dataset")

# Download test data / source data
dl_tab1, dl_tab2 = st.tabs(["Download test data", "Download source data"])
with dl_tab1:
    if os.path.exists("test_data.csv"):
        _test_df = pd.read_csv("test_data.csv")
        st.download_button(
            "Download test_data.csv",
            data=_test_df.to_csv(index=False).encode("utf-8"),
            file_name="test_data.csv",
            mime="text/csv",
            key="dl_test_data"
        )
        st.caption(f"Test set: {len(_test_df):,} rows, {len(_test_df.columns)} columns.")
    else:
        st.info("test_data.csv not found. Run train_models.py to generate it.")
with dl_tab2:
    if os.path.exists("adult.csv"):
        _adult_df = pd.read_csv("adult.csv")
        st.download_button(
            "Download adult.csv",
            data=_adult_df.to_csv(index=False).encode("utf-8"),
            file_name="adult.csv",
            mime="text/csv",
            key="dl_source_data"
        )
        st.caption(f"Source data: {len(_adult_df):,} rows, {len(_adult_df.columns)} columns.")
    else:
        st.info("adult.csv not found in the app directory.")

# Load preprocessing artifacts once
if "scaler" not in st.session_state:
    try:
        st.session_state.scaler, st.session_state.label_enc, st.session_state.feature_names = load_artifacts()
    except Exception as e:
        st.error(f"Could not load model artifacts from '{MODEL_DIR}/'. Run train_models.py first.")
        st.stop()

# CSV upload (test data)
uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"])
if not uploaded:
    st.info("Upload a CSV with the same columns as the training data (including 'income' for metrics).")
    st.stop()

df = pd.read_csv(uploaded)
if TARGET not in df.columns:
    st.warning(f"CSV must contain a column '{TARGET}' to show metrics.")
    y_true = None
else:
    y_true = st.session_state.label_enc.transform(df[TARGET].astype(str))

# Model dropdown
model_names = get_model_names()
if not model_names:
    st.error(f"No model files found in '{MODEL_DIR}/'. Run train_models.py first.")
    st.stop()

selected = st.selectbox("Select model", model_names)
clf = load_model(selected)

# Preprocess and predict
try:
    X = preprocess_df(df, st.session_state.scaler, st.session_state.feature_names)
except Exception as e:
    st.error(f"Preprocessing failed: {e}")
    st.stop()

y_pred = clf.predict(X)
y_proba = clf.predict_proba(X) if hasattr(clf, "predict_proba") else None

# Metrics
st.subheader("Evaluation metrics")
if y_true is not None and len(y_true) == len(y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba[:, 1]) if y_proba is not None else 0.0
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
        "Value": [acc, auc, prec, rec, f1, mcc]
    })
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
else:
    st.write("Predictions only (no ground truth):", pd.Series(y_pred).value_counts())

# Confusion matrix / classification report
st.subheader("Confusion matrix & classification report")
if y_true is not None and len(y_true) == len(y_pred):
    cm = confusion_matrix(y_true, y_pred)
    st.write("Confusion matrix:")
    st.dataframe(pd.DataFrame(cm), use_container_width=True, hide_index=True)
    st.write("Classification report:")
    report = classification_report(
        y_true, y_pred,
        target_names=st.session_state.label_enc.classes_.tolist(),
        output_dict=True
    )
    st.dataframe(pd.DataFrame(report).T, use_container_width=True, hide_index=True)
else:
    st.write("Upload a CSV with 'income' column to see confusion matrix and report.")

# Plots
st.subheader("Plots")
class_names = st.session_state.label_enc.classes_.tolist()
fig_col1, fig_col2 = st.columns(2)

with fig_col1:
    if y_true is not None and len(y_true) == len(y_pred):
        fig_cm, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion matrix")
        st.pyplot(fig_cm)
        plt.close()
    pred_counts = pd.Series(y_pred).map(lambda i: class_names[i]).value_counts()
    fig_dist, ax = plt.subplots()
    pred_counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Count")
    ax.set_title("Predicted class distribution")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig_dist)
    plt.close()

with fig_col2:
    if y_true is not None and len(y_true) == len(y_pred):
        metrics_dict = {
            "Accuracy": acc, "AUC": auc, "Precision": prec, "Recall": rec, "F1": f1, "MCC": mcc
        }
        fig_metrics, ax = plt.subplots()
        names = list(metrics_dict.keys())
        values = list(metrics_dict.values())
        ax.barh(names, values, color="teal", alpha=0.8)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Score")
        ax.set_title("Evaluation metrics")
        st.pyplot(fig_metrics)
        plt.close()
    if y_proba is not None and y_true is not None and len(y_true) == len(y_pred):
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        fig_roc, ax = plt.subplots()
        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("ROC curve")
        ax.legend(loc="lower right")
        st.pyplot(fig_roc)
        plt.close()

# Predicted output option
st.subheader("Predicted output")
show_predictions = st.checkbox("Show predicted output", value=False)
if show_predictions:
    pred_labels = st.session_state.label_enc.inverse_transform(y_pred)
    out = df.copy()
    out["predicted"] = pred_labels
    if y_true is not None and len(y_true) == len(y_pred):
        out["actual"] = st.session_state.label_enc.inverse_transform(y_true)
    all_cols = list(out.columns)
    default_n = min(5, len([c for c in all_cols if c not in ("predicted", "actual")]))
    data_cols = [c for c in all_cols if c not in ("predicted", "actual")]
    default_cols = data_cols[:default_n] + ["predicted"]
    if "actual" in out.columns:
        default_cols.append("actual")
    selected_cols = st.multiselect("Columns to show", options=all_cols, default=default_cols)
    n_show = st.slider("Number of rows to show", min_value=10, max_value=max(10, min(500, len(out))), value=min(50, len(out)), step=10)
    if selected_cols:
        st.dataframe(out[selected_cols].head(n_show), use_container_width=True, hide_index=True)
    st.caption(f"Showing first {n_show} of {len(out)} rows.")
