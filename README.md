🛡️ Enterprise Credit Card Fraud Intelligence & SHAP Platform
An AI-powered web application that detects high-risk credit card transactions in real-time using XGBoost, SMOTE Class Balancing, and TreeSHAP Explainable AI.

🌐 Live App: https://credit-card-fraud-intelligence-569xkwbbjjfehajfqvymkk.streamlit.app/

📌 1. Executive Summary
Traditional "black-box" AI flags fraud without explaining why. This platform combines predictive machine learning with Explainable AI (TreeSHAP) to calculate precise fraud probabilities and show the exact financial drivers behind every decision.

🏗️ 2. System Architecture
Input Payload: Transaction features (Time, Amount, PCA vectors V1–V28).

XGBoost Engine: Evaluates inputs and predicts a 0–100% fraud probability score.

TreeSHAP Engine: Calculates game-theoretic feature attribution in real time.

Streamlit UI: Displays risk badges, probability scores, and risk driver tables.

📊 3. Key Features & Data Pipeline
SMOTE Resampling: Handles severe class imbalance without synthetic distortion.

PR-AUC Optimization: Prioritizes detecting rare fraud events while minimizing false alarms.

Core Inputs:

Time & Amount: Transaction timestamp and monetary dollar value ($).

V1–V28: Anonymized PCA features protecting privacy (key fraud signals include V14, V12, V17, and V4).

💡 4. How to Read Results (Layman's Guide)
Risk Tier Badges
🟢 LOW RISK (0%–34%): Approved transaction. Standard behavior.

🟡 SUSPICIOUS (35%–74%): Medium risk. Requires secondary verification (SMS OTP).

🔴 CRITICAL RISK (75%–100%): High-risk fraud. Transaction automatically rejected.

SHAP Risk Drivers Table
Feature & Value: The specific input evaluated (e.g., Amount = $350.00).

SHAP Impact: Positive values (+1.85) push risk up toward fraud; negative values (-0.92) pull risk down toward safe.

Direction: 🔴 Increases Risk (raised alerts) vs. 🟢 Decreases Risk (proved legitimacy).

🛠️ 5. Quick Start & Tech Stack
Stack: Streamlit, XGBoost, SHAP, Scikit-Learn, Pandas, Joblib

Run Locally:

Bash
pip install -r requirements.txt
streamlit run app_streamlit.py --server.headless=true --server.enableCORS=false