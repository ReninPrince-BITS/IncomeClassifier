# ML Assignment 2 - Classification on Adult Income Dataset

## Problem statement

Predict whether a person’s income is **≤50K** or **>50K** (binary classification) using the UCI Adult (Census Income) dataset. The app trains six classifiers, evaluates them with standard metrics, and exposes a Streamlit UI to upload test CSV, select a model, and view metrics plus confusion matrix/classification report.

## Dataset description

- **Source:** UCI Machine Learning Repository (Census / Adult Income).
- **Task:** Binary classification (income ≤50K vs >50K).
- **Instances:** 32,561 (meets minimum 500).
- **Features:** 14 (meets minimum 12).  
  - Numeric: `age`, `fnlwgt`, `education.num`, `capital.gain`, `capital.loss`, `hours.per.week`.  
  - Categorical: `workclass`, `education`, `marital.status`, `occupation`, `relationship`, `race`, `sex`, `native.country`.
- **Target:** `income` (<=50K, >50K).
- **Preprocessing:** Missing values (`?`) replaced with column mode; categoricals one-hot encoded; numeric features standardized.

## Models used

| ML Model Name        | Accuracy | AUC   | Precision | Recall | F1    | MCC   |
|----------------------|----------|-------|-----------|--------|-------|-------|
| Logistic Regression  | 0.8468   | 0.9017| 0.7172    | 0.5791 | 0.6407| 0.5498|
| Decision Tree        | 0.8153   | 0.7431| 0.6092    | 0.6064 | 0.6078| 0.4870|
| K-Nearest Neighbor   | 0.8184   | 0.8280| 0.6294    | 0.5602 | 0.5928| 0.4777|
| Naive Bayes          | 0.3553   | 0.6416| 0.2647    | 0.9740 | 0.4162| 0.1736|
| Random Forest        | 0.8514   | 0.8995| 0.7177    | 0.6103 | 0.6596| 0.5684|
| XGBoost              | 0.8715   | 0.9266| 0.7730    | 0.6448 | 0.7031| 0.6261|

## Observations (model performance)

| ML Model Name        | Observation |
|----------------------|-------------|
| Logistic Regression  | Good baseline; fast and interpretable; AUC and accuracy competitive. |
| Decision Tree        | No scaling needed; lower AUC suggests weaker probability calibration. |
| K-Nearest Neighbor   | Benefits from standardization; moderate accuracy and AUC. |
| Naive Bayes          | Poor accuracy here; Gaussian assumption on encoded features may not fit well. |
| Random Forest        | Strong accuracy and AUC; robust to overfitting vs single tree. |
| XGBoost              | Best accuracy and AUC; best F1 and MCC among the six models. |

## How to run

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Train models and save to `model/`**
   ```bash
   python train_models.py
   ```
3. **Run Streamlit app**
   ```bash
   streamlit run streamlit_app.py
   ```
4. In the app: upload a CSV with the same columns as the training data (including `income` for metrics), select a model, and view metrics and confusion matrix/classification report.

## Repository structure

```
project-folder/
├── streamlit_app.py   # Streamlit UI
├── train_models.py    # Train 6 models and save to model/
├── requirements.txt
├── README.md
├── adult.csv          # Dataset (optional; use your own path in train_models.py)
└── model/             # Created by train_models.py: *.joblib, metrics_summary.csv
```

## Deployment (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. Go to https://streamlit.io/cloud, sign in with GitHub.
3. New App → select repo, branch (e.g. `main`), main file `streamlit_app.py` → Deploy.
4. Ensure `model/` (with all `.joblib` and artifacts) is committed so the app can load models.
