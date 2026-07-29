import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

# Initialize Flask App
app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent

# Load Machine Learning & SHAP Artifacts
print("Loading ML model artifacts and SHAP explainer into Flask memory...")

try:
    model = joblib.load(BASE_DIR / "fraud_champion_model.joblib")
    explainer = joblib.load(BASE_DIR / "shap_explainer.joblib")
    feature_names = joblib.load(BASE_DIR / "feature_names.joblib")
    metadata = joblib.load(BASE_DIR / "model_metadata.joblib")
    print(
        f"Loaded champion model: {metadata.get('champion', 'Trained Classifier')} (PR-AUC: {metadata.get('pr_auc', 0.0):.4f})"
    )
except Exception as e:
    print(f"Error loading model artifacts: {e}")
    print(
        "Ensure you ran `python train_fraud_engine.py` using the real creditcard.csv dataset first!"
    )
    model, explainer, feature_names, metadata = None, None, None, {}


# --- ROUTES & REST API ENDPOINTS ---


@app.route("/")
def index():
    """Renders the main Financial Intelligence Command Center UI."""
    return render_template(
        "index.html",
        features=feature_names or [],
        metadata=metadata or {},
    )


@app.route("/api/v1/metadata", methods=["GET"])
def get_metadata():
    """Returns model metadata and feature list for UI setup."""
    if not metadata:
        return jsonify({"status": "error", "message": "Model not loaded"}), 500

    return jsonify(
        {
            "status": "success",
            "champion_model": metadata.get("champion", "N/A"),
            "pr_auc_score": metadata.get("pr_auc", 0.0),
            "feature_count": len(feature_names),
            "features": feature_names,
        }
    )


@app.route("/api/v1/predict", methods=["POST"])
def predict_fraud():
    """REST API endpoint for real-time transaction fraud scoring and SHAP feature attribution."""
    if model is None or explainer is None:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "ML Engine unavailable. Artifacts not loaded.",
                }
            ),
            500,
        )

    try:
        data = request.get_json(force=True)

        if not data or "features" not in data:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "Invalid payload format. Expected JSON: {'features': { 'Time': ..., 'V1': ..., 'Amount': ... }}",
                    }
                ),
                400,
            )

        input_dict = data["features"]

        # Ensure input contains all expected features in correct order
        input_vector = []
        for feat in feature_names:
            val = input_dict.get(feat, 0.0)
            input_vector.append(float(val))

        # Convert to DataFrame
        input_df = pd.DataFrame([input_vector], columns=feature_names)

        # 1. Compute Model Prediction & Fraud Probability Score
        fraud_probability = float(model.predict_proba(input_df)[0][1])

        # 2. Compute Real-Time SHAP Feature Attribution Values
        shap_values = explainer(input_df)

        # Handle SHAP output dimensions safely (Tree vs Linear Explainer formats)
        if hasattr(shap_values, "values"):
            vals = shap_values.values
            if len(vals.shape) == 3:  # Binary classification output array
                vals = vals[:, :, 1]
            feature_impacts = vals[0]
            base_value = float(
                shap_values.base_values[0][1]
                if len(np.array(shap_values.base_values).shape) > 1
                else shap_values.base_values[0]
            )
        else:
            feature_impacts = shap_values[0]
            base_value = float(explainer.expected_value)

        # Structure Top SHAP Risk Drivers (Sorted by highest positive impact toward fraud)
        shap_breakdown = []
        for feat, value, impact in zip(
            feature_names, input_vector, feature_impacts
        ):
            shap_breakdown.append(
                {
                    "feature": feat,
                    "actual_value": round(float(value), 4),
                    "shap_impact": round(float(impact), 4),
                    "risk_direction": (
                        "Increases Risk" if impact > 0 else "Decreases Risk"
                    ),
                }
            )

        # Sort features by absolute SHAP impact
        shap_breakdown.sort(
            key=lambda x: abs(x["shap_impact"]), reverse=True
        )

        # Set Dynamic Risk Category
        if fraud_probability >= 0.75:
            risk_tier = "CRITICAL RISK"
            action = "AUTOMATIC REJECTION / BLOCK CARD"
        elif fraud_probability >= 0.35:
            risk_tier = "SUSPICIOUS / ELEVATED RISK"
            action = "TRIGGER MULTI-FACTOR AUTHENTICATION (MFA)"
        else:
            risk_tier = "LOW RISK"
            action = "APPROVED TRANSACTION"

        return jsonify(
            {
                "status": "success",
                "fraud_probability": round(fraud_probability, 4),
                "fraud_percentage": f"{fraud_probability * 100:.2f}%",
                "risk_tier": risk_tier,
                "recommended_action": action,
                "base_expectation": round(base_value, 4),
                "top_risk_drivers": shap_breakdown[
                    :8
                ],  # Top 8 impactful features for UI
            }
        )

    except Exception as err:
        return (
            jsonify(
                {"status": "error", "message": f"Inference Failed: {str(err)}"}
            ),
            500,
        )


if __name__ == "__main__":
    # Local development server
    app.run(host="0.0.0.0", port=5000, debug=True)
