import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Enterprise Fraud Intelligence & SHAP",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. INJECT CUSTOM CYBER FINTECH CSS (Matching Phase 3 HTML/CSS Styling)
st.markdown(
    """
<style>
    /* Dark Theme Base Colors */
    :root {
        --bg-main: #070a13;
        --bg-card: #0d1527;
        --bg-input: #131f37;
        --border-color: #1e293b;
        --accent-blue: #0284c7;
        --accent-teal: #14b8a6;
        --accent-crimson: #f43f5e;
        --text-muted: #94a3b8;
    }

    /* Override Streamlit Main Container */
    .stApp {
        background-color: var(--bg-main) !important;
        color: #f8fafc !important;
    }

    /* Custom Header Banner */
    .header-banner {
        background: linear-gradient(135deg, #0d1527 0%, #111e38 50%, #0369a1 100%);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .header-title {
        font-size: 1.8rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
    }

    .header-subtitle {
        color: var(--text-muted);
        font-size: 0.95rem;
        margin-top: 0.25rem;
    }

    .active-badge {
        background: rgba(20, 184, 166, 0.15);
        border: 1px solid var(--accent-teal);
        color: var(--accent-teal);
        padding: 0.4rem 1rem;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 700;
    }

    /* Decision Analytics Display Cards */
    .metric-card {
        background-color: var(--bg-input);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }

    .risk-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: var(--text-muted);
        font-weight: 700;
    }

    .probability-number {
        font-size: 3rem;
        font-weight: 900;
        margin: 0.5rem 0;
    }

    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 800;
        letter-spacing: 0.5px;
    }

    .status-low { background: rgba(20, 184, 166, 0.15); color: var(--accent-teal); border: 1px solid var(--accent-teal); }
    .status-med { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid #f59e0b; }
    .status-critical { background: rgba(244, 63, 94, 0.15); color: var(--accent-crimson); border: 1px solid var(--accent-crimson); }

    /* Custom Input Styling */
    div[data-baseweb="input"] {
        background-color: var(--bg-input) !important;
        border-radius: 8px !important;
        border: 1px solid var(--border-color) !important;
        color: white !important;
    }

    /* Custom Button Styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 0.75rem !important;
        border-radius: 10px !important;
        border: none !important;
        box-shadow: 0 0 20px rgba(2, 132, 199, 0.4) !important;
        transition: all 0.3s ease !important;
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #0369a1 0%, #075985 100%) !important;
        box-shadow: 0 0 30px rgba(56, 189, 248, 0.6) !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


# 3. LOAD MODEL & SHAP ARTIFACTS
@st.cache_resource
def load_artifacts():
    model = joblib.load("fraud_champion_model.joblib")
    explainer = joblib.load("shap_explainer.joblib")
    features = joblib.load("feature_names.joblib")
    meta = joblib.load("model_metadata.joblib")
    return model, explainer, features, meta


try:
    model, explainer, feature_names, metadata = load_artifacts()
except Exception as e:
    st.error(
        f"❌ Error loading ML artifacts: {e}. Make sure you ran train_fraud_engine.py first!"
    )
    st.stop()

# 4. SIDEBAR CONFIGURATION
st.sidebar.markdown("## ⚙️ Engine Status")
st.sidebar.info(
    f"**Champion Model:** {metadata.get('champion', 'Trained Classifier')}"
)
st.sidebar.metric(
    "Evaluation PR-AUC", f"{metadata.get('pr_auc', 0.0):.4f}"
)
st.sidebar.markdown("---")
st.sidebar.markdown("**System Architecture:**")
st.sidebar.caption("• XGBoost + TreeSHAP Engine")
st.sidebar.caption("• SMOTE Imbalance Balancing")
st.sidebar.caption("• Real-Time Feature Attribution")

# 5. HEADER BANNER
st.markdown(
    """
    <div class="header-banner">
        <div>
            <div class="header-title">🛡️ Enterprise Fraud Intelligence & SHAP Platform</div>
            <div class="header-subtitle">Real-time transaction risk scoring with TreeSHAP feature attribution & automated ML monitoring.</div>
        </div>
        <div class="active-badge">⚡ XGBoost Engine: Active</div>
    </div>
""",
    unsafe_allow_html=True,
)

# 6. MAIN WORKSPACE LAYOUT (2 Columns)
col1, col2 = st.columns([1.1, 1], gap="large")

with col1:
    st.markdown("### 📡 Transaction Payload Inspection")

    with st.form("fraud_form"):
        sub_col1, sub_col2 = st.columns(2)

        with sub_col1:
            time_val = st.number_input("Time (Seconds)", value=45210, step=1)
            v14_val = st.number_input(
                "PCA Feature V14 (Critical)", value=-2.14, step=0.01
            )
            v17_val = st.number_input(
                "PCA Feature V17", value=-1.20, step=0.01
            )

        with sub_col2:
            amount_val = st.number_input(
                "Transaction Amount ($)", value=350.00, step=0.01
            )
            v12_val = st.number_input("PCA Feature V12", value=-1.85, step=0.01)
            v4_val = st.number_input("PCA Feature V4", value=1.45, step=0.01)

        submit_btn = st.form_submit_button(
            "🔍 Evaluate Fraud Risk & Compute SHAP"
        )

with col2:
    st.markdown("### 📊 Real-Time Decision Analytics")

    if submit_btn:
        # Build complete feature vector
        input_dict = {
            "Time": float(time_val),
            "Amount": float(amount_val),
            "V14": float(v14_val),
            "V12": float(v12_val),
            "V17": float(v17_val),
            "V4": float(v4_val),
        }
        # Fill remaining PCA features with 0.0
        for feat in feature_names:
            if feat not in input_dict:
                input_dict[feat] = 0.0

        input_vector = [input_dict[feat] for feat in feature_names]
        input_df = pd.DataFrame([input_vector], columns=feature_names)

        # 1. Inference Prediction
        fraud_prob = float(model.predict_proba(input_df)[0][1])
        fraud_percentage = f"{fraud_prob * 100:.2f}%"

        # 2. Determine Risk Tier & Colors
        if fraud_prob >= 0.75:
            risk_tier = "CRITICAL RISK - AUTOMATIC REJECTION"
            badge_class = "status-critical"
            num_color = "#f43f5e"
        elif fraud_prob >= 0.35:
            risk_tier = "SUSPICIOUS - TRIGGER MFA"
            badge_class = "status-med"
            num_color = "#fbbf24"
        else:
            risk_tier = "LOW RISK - APPROVED TRANSACTION"
            badge_class = "status-low"
            num_color = "#14b8a6"

        # Render Assessed Score Box
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="risk-title">Assessed Fraud Probability</div>
                <div class="probability-number" style="color: {num_color};">{fraud_percentage}</div>
                <div class="status-badge {badge_class}">{risk_tier}</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

        # 3. Compute Real-Time SHAP Values
        shap_values = explainer(input_df)

        if hasattr(shap_values, "values"):
            vals = shap_values.values
            if len(vals.shape) == 3:
                vals = vals[:, :, 1]
            feature_impacts = vals[0]
        else:
            feature_impacts = shap_values[0]

        shap_breakdown = []
        for feat, val, impact in zip(
            feature_names, input_vector, feature_impacts
        ):
            shap_breakdown.append(
                {
                    "Feature": feat,
                    "Actual Value": round(float(val), 4),
                    "SHAP Impact": round(float(impact), 4),
                    "Direction": (
                        "🔴 Increases Risk"
                        if impact > 0
                        else "🟢 Decreases Risk"
                    ),
                    "abs_impact": abs(float(impact)),
                }
            )

        # Sort by highest absolute SHAP impact
        shap_breakdown.sort(key=lambda x: x["abs_impact"], reverse=True)
        shap_df = pd.DataFrame(shap_breakdown).drop(columns=["abs_impact"])

        st.markdown("**Top SHAP Risk Drivers (Feature Attribution)**")
        st.dataframe(
            shap_df.head(8),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "👉 Adjust payload values on the left and click **'Evaluate Fraud Risk'** to compute inferences."
        )