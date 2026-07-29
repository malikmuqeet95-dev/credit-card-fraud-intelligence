# 🛡️ Enterprise Credit Card Fraud Intelligence & SHAP Platform

A production-grade, microservice-based Machine Learning web application engineered to detect high-risk banking transactions in real-time. Built with **Flask**, **XGBoost**, **SMOTE Imbalance Resampling**, and **TreeSHAP Explainability Engine**.

---

## 🌟 Key Features
- **Multi-Model ML Pipeline:** Evaluates Logistic Regression, Random Forest, and XGBoost using **PR-AUC (Precision-Recall AUC)** and **F2-Score** benchmarks.
- **Explainable AI (XAI):** Real-time **TreeSHAP feature attribution** breaking down top risk drivers for compliance and risk managers.
- **Custom FinTech Command Center:** Built with Flask REST endpoints, dark obsidian UI CSS, and dynamic client-side JavaScript rendering.

---

## 🛠️ Tech Stack
- **Backend Framework:** Flask (Python REST API)
- **Machine Learning & XAI:** `xgboost`, `scikit-learn`, `imbalanced-learn` (SMOTE), `shap`, `joblib`
- **WSGI Server:** `gunicorn`
- **Deployment:** Render Web Service