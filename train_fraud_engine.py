import os
import joblib
import pandas as pd
import numpy as np
import shap
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve, auc, fbeta_score, classification_report

DATA_PATH = os.path.join("data", "creditcard.csv")

# 1. LOAD REAL DATASET
if not os.path.exists(DATA_PATH):
    print(f"📥 'creditcard.csv' not found locally in 'data/' folder.")
    print("🔄 Downloading dataset automatically via URL...")
    os.makedirs("data", exist_ok=True)
    url = "https://raw.githubusercontent.com/psundaravadivel/Credit-Card-Fraud-Detection/main/CreditcardDataset.csv"
    df = pd.read_csv(url)
    df.to_csv(DATA_PATH, index=False)
    print("✅ Download complete! Saved to 'data/creditcard.csv'.")
else:
    print(f"📥 Loading dataset from local file: '{DATA_PATH}'...")
    df = pd.read_csv(DATA_PATH)

print(f"📊 Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"🚨 Fraud Ratio: {df['Class'].value_counts(normalize=True)[1]:.4%} ({df['Class'].sum()} fraud cases)")

# Separate features & target
X = df.drop(columns=["Class"])
y = df["Class"]
feature_names = X.columns.tolist()

# 2. STRATIFIED TRAIN / TEST SPLIT (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. SMOTE RESAMPLING ON TRAINING DATA
print("⚖️ Applying SMOTE to balance class distribution in training set...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 4. BENCHMARK CANDIDATE MODELS
candidate_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    "XGBoost Classifier": XGBClassifier(n_estimators=120, max_depth=6, learning_rate=0.08, eval_metric="logloss", random_state=42, n_jobs=-1)
}

results = {}
best_model_name = None
best_pr_auc = -1.0
best_model_obj = None

print("\n" + "=" * 65)
print("🏆 EVALUATING MODELS ON REAL UNBALANCED TEST SET")
print("=" * 65)

for name, model in candidate_models.items():
    print(f"\n⚡ Training {name}...")
    model.fit(X_train_res, y_train_res)
    probs = model.predict_proba(X_test)[:, 1]

    # Calculate Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(y_test, probs)
    pr_auc = auc(recall, precision)

    # Calculate F2-Score (Focuses heavily on Fraud Recall)
    preds = (probs >= 0.5).astype(int)
    f2 = fbeta_score(y_test, preds, beta=2)

    results[name] = {"PR-AUC": pr_auc, "F2-Score": f2}
    print(f"   - PR-AUC Metric : {pr_auc:.4f}")
    print(f"   - F2-Score      : {f2:.4f}")

    if pr_auc > best_pr_auc:
        best_pr_auc = pr_auc
        best_model_name = name
        best_model_obj = model

print("\n" + "=" * 65)
print(f"🥇 CHAMPION MODEL: '{best_model_name}' (PR-AUC = {best_pr_auc:.4f})")
print("=" * 65)

# 5. BUILD SHAP EXPLAINER FOR CHAMPION MODEL
print("\n🧠 Constructing SHAP TreeExplainer Engine...")
explainer = shap.TreeExplainer(best_model_obj)

# 6. EXPORT MODEL ARTIFACTS
print("💾 Saving Model Artifacts...")
joblib.dump(best_model_obj, "fraud_champion_model.joblib")
joblib.dump(explainer, "shap_explainer.joblib")
joblib.dump(feature_names, "feature_names.joblib")
joblib.dump({"champion": best_model_name, "pr_auc": float(best_pr_auc)}, "model_metadata.joblib")

print("🎉 PHASE 1 COMPLETE! Artifacts exported successfully.")