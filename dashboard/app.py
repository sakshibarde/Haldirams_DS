import os
import json
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import shap
from scipy.stats import ks_2samp

# --- Paths & Lazy Loading (CORRECTED VERSION) ---
# Construct the absolute path to the artifacts directory relative to this script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ART_DIR = os.path.join(BASE_DIR, "artifacts")
PIPE_PATH = os.path.join(ART_DIR, "inference_pipeline.joblib")
COLS_PATH = os.path.join(ART_DIR, "expected_columns.json")
REF_PATH = os.path.join(ART_DIR, "reference_sample.csv")

LABEL_MAP = {0: "Not Underperforming", 1: "Underperforming"}

@st.cache_resource(show_spinner="Loading model artifacts...")
def load_artifacts():
    """Loads the pipeline, expected columns, and reference data."""
    load_error = None
    pipeline, expected_cols, ref = None, None, None
    try:
        pipeline = joblib.load(PIPE_PATH)
        with open(COLS_PATH) as f:
            expected_cols = json.load(f)["expected_input_cols"]
    except Exception as e:
        load_error = f"Artifact load failed: {e}. Ensure 'inference_pipeline.joblib' and 'expected_columns.json' are in the 'artifacts' folder."

    if os.path.exists(REF_PATH):
        try:
            ref = pd.read_csv(REF_PATH)
        except Exception as e:
            load_error = f"Reference sample load failed: {e}"

    return pipeline, expected_cols, ref, load_error

inference_pipeline, EXPECTED_COLS, REF, LOAD_ERR = load_artifacts()

st.set_page_config(page_title="Haldiram's Performance Dashboard", layout="wide")
st.title("📊 Haldiram's Product Performance — Predictions & Insights")
st.caption("A Streamlit dashboard for predictions, SHAP explanations, fairness, and drift checks.")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload CSV for analysis", type=["csv"])
    sensitive_attr = st.text_input("Sensitive attribute (grouping column)", value="category_reclassified")
    target_attr = st.text_input("Ground-truth column (optional)", value="is_underperforming")
    threshold = st.slider("Probability threshold for 'Underperforming'", 0.0, 1.0, 0.5, 0.01)
    st.divider()
    st.write("Artifacts status:", "✅ `OK`" if inference_pipeline and EXPECTED_COLS else f"❌ `Error: {LOAD_ERR}`")

# --- Helper Functions ---
def align_columns(df, expected_cols):
    """Aligns DataFrame columns to a predefined list, adding missing ones as None."""
    for c in expected_cols:
        if c not in df.columns:
            df[c] = None
    return df[expected_cols]

def predict_df(df):
    """Generates predictions and probabilities from the pipeline."""
    proba = inference_pipeline.predict_proba(df)[:, 1]
    preds = (proba >= threshold).astype(int)
    return preds, proba

def psi(reference, current, bins=10):
    """Calculates the Population Stability Index (PSI) between two series."""
    ref_clean = reference.dropna().astype(float)
    cur_clean = current.dropna().astype(float)
    if len(ref_clean) < 20 or len(cur_clean) < 20: return np.nan
    
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.quantile(ref_clean, quantiles)
    edges[0], edges[-1] = -np.inf, np.inf
    
    ref_counts = np.histogram(ref_clean, bins=edges)[0]
    cur_counts = np.histogram(cur_clean, bins=edges)[0]
    
    ref_perc = np.where(ref_counts == 0, 1e-6, ref_counts / len(ref_clean))
    cur_perc = np.where(cur_counts == 0, 1e-6, cur_counts / len(cur_clean))
    
    return float(np.sum((cur_perc - ref_perc) * np.log(cur_perc / ref_perc)))

# --- Tabs ---
tab_pred, tab_shap, tab_fair, tab_drift = st.tabs(["🔮 Predict", "🔎 SHAP Explanations", "⚖️ Fairness Audit", "🌊 Data Drift"])

# --- Predict Tab ---
with tab_pred:
    st.subheader("Batch Predictions from CSV")
    if not all([inference_pipeline, EXPECTED_COLS]):
        st.warning("Cannot make predictions. Artifacts not loaded correctly.")
    else:
        df_in = None
        if uploaded:
            try:
                df_in = pd.read_csv(io.BytesIO(uploaded.getvalue()))
                st.dataframe(df_in.head(), use_container_width=True)
            except Exception as e:
                st.error(f"Could not read uploaded CSV: {e}")
        else:
            st.info("Upload a CSV file to see predictions.")
            # Demo record for Haldiram's
            demo = {
                "rating": 3.8, "price_whole": 450, "mrp": 450,
                "number_of_global_ratings": 25, "number_of_reviews": 15,
                "discount_percentage_cleaned": 0, "product_weight_grams": 500,
                "category_reclassified": "Sweets & Desserts"
            }
            df_in = pd.DataFrame([demo])
            st.write("Showing prediction for a sample record:")

        if df_in is not None:
            aligned_df = align_columns(df_in.copy(), EXPECTED_COLS)
            preds, proba = predict_df(aligned_df)
            out_df = pd.DataFrame({
                "prediction": preds,
                "label": [LABEL_MAP.get(p, "Unknown") for p in preds],
                "proba_underperforming": proba
            })
            st.success(f"Predicted {len(out_df)} rows.")
            st.dataframe(out_df.head(50), use_container_width=True)

            # Optional: Display metrics if ground truth is available
            if target_attr and target_attr in df_in.columns:
                y_true = df_in[target_attr].astype(int)
                y_pred = out_df["prediction"]
                st.write("---")
                st.subheader("Performance Metrics (vs. Ground Truth)")
                st.text(classification_report(y_true, y_pred, target_names=LABEL_MAP.values()))
                cm = confusion_matrix(y_true, y_pred)
                cm_df = pd.DataFrame(cm, index=[f"True: {v}" for v in LABEL_MAP.values()], columns=[f"Pred: {v}" for v in LABEL_MAP.values()])
                st.write("**Confusion Matrix**")
                st.dataframe(cm_df, use_container_width=True)

# --- SHAP Tab ---
with tab_shap:
    st.subheader("Global Feature Importance (SHAP)")
    st.caption("This tab explains which features have the biggest impact on the model's predictions overall.")
    
    base_data = REF if REF is not None else (pd.read_csv(io.BytesIO(uploaded.getvalue())) if uploaded else None)
    if not all([inference_pipeline, base_data is not None]):
        st.warning("Cannot compute SHAP values. A reference dataset (either `reference_sample.csv` or an uploaded file) is required.")
    else:
        with st.spinner("Calculating SHAP values... This may take a moment."):
            model = inference_pipeline.named_steps['classifier']
            preprocessor = inference_pipeline.named_steps['preprocessor']
            
            # Prepare data for SHAP
            X_shap = preprocessor.transform(align_columns(base_data, EXPECTED_COLS))
            feature_names = preprocessor.get_feature_names_out()

            # Select appropriate SHAP explainer
            if isinstance(model, (RandomForestClassifier, DecisionTreeClassifier)):
                explainer = shap.TreeExplainer(model, X_shap)
                shap_values = explainer.shap_values(X_shap)
                shap_values_for_plot = shap_values[1] if isinstance(shap_values, list) else shap_values
            elif isinstance(model, LogisticRegression):
                explainer = shap.LinearExplainer(model, X_shap)
                shap_values = explainer.shap_values(X_shap)
                shap_values_for_plot = shap_values
            else: # Fallback for other models
                explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_shap, 50))
                shap_values = explainer.shap_values(X_shap, nsamples=100)
                shap_values_for_plot = shap_values[1]
                
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values_for_plot, X_shap, feature_names=feature_names, show=False)
            st.pyplot(fig, clear_figure=True)

# --- Fairness Tab ---
with tab_fair:
    st.subheader("Fairness Audit: Group Performance Comparison")
    if not uploaded:
        st.info("Upload a CSV file to perform a fairness audit.")
    elif sensitive_attr not in (df := pd.read_csv(io.BytesIO(uploaded.getvalue()))).columns:
        st.warning(f"Sensitive attribute '{sensitive_attr}' not found in the uploaded file. Please select a valid column.")
    else:
        aligned_df = align_columns(df.copy(), EXPECTED_COLS)
        preds, _ = predict_df(aligned_df)
        df['prediction'] = preds
        
        st.write(f"**Selection Rate by '{sensitive_attr}'**")
        st.caption("The percentage of products in each group predicted as 'Underperforming'.")
        selection_rate = df.groupby(sensitive_attr)['prediction'].value_counts(normalize=True).unstack().fillna(0)
        st.dataframe(selection_rate[[1]].rename(columns={1: 'Underperforming Rate'}), use_container_width=True)

# --- Drift Tab ---
with tab_drift:
    st.subheader("Data Drift Detection (vs. Reference Sample)")
    if REF is None:
        st.warning("Cannot perform drift analysis. `reference_sample.csv` not found in artifacts.")
    elif not uploaded:
        st.info("Upload a CSV file to compare with the reference data.")
    else:
        current_df = pd.read_csv(io.BytesIO(uploaded.getvalue()))
        numeric_cols = [c for c in EXPECTED_COLS if pd.api.types.is_numeric_dtype(REF[c]) and c in current_df.columns and pd.api.types.is_numeric_dtype(current_df[c])]
        
        drift_report = []
        for col in numeric_cols:
            psi_score = psi(REF[col], current_df[col])
            ks_stat, ks_pvalue = ks_2samp(REF[col].dropna(), current_df[col].dropna())
            drift_report.append({"feature": col, "psi": psi_score, "ks_pvalue": ks_pvalue})
        
        drift_df = pd.DataFrame(drift_report).sort_values("psi", ascending=False)
        st.dataframe(drift_df, use_container_width=True)
        st.caption("PSI > 0.2 indicates significant drift. KS p-value < 0.05 indicates a statistically significant difference in distributions.")
