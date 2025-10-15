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
# st.caption("A storytelling dashboard: Context → Data → EDA → Modeling → XAI → Fairness → Monitoring.")

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
with st.expander("1) Problem Statement — what are we solving?", expanded=True):
    st.markdown(
        "- **Goal**: Identify which products are likely to be 'Underperforming' so teams can take timely actions (pricing, promotion, assortment).\n"
        "- **Why it matters**: Prevent stock sitting idle, improve conversion, and focus on best-sellers.\n"
        "- **How the app helps**: Upload product data, get predictions with simple explanations and quality checks."
    )

with st.expander("2) Dataset overview & Feature glossary", expanded=True):
    # Always use local dashboard data for storytelling
    data_path = os.path.join(BASE_DIR, "data", "haldirams_cleaned.csv")
    sample_df = None
    if os.path.exists(data_path):
        try:
            sample_df = pd.read_csv(data_path)
        except Exception:
            sample_df = None

    if sample_df is not None:
        st.write("A quick peek at the data (first 10 rows):")
        st.dataframe(sample_df.head(10), use_container_width=True)

        # Simple profile: rows, columns, missingness
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", len(sample_df))
        with col2:
            st.metric("Columns", len(sample_df.columns))
        # with col3:
        #     st.metric("Missing cells", int(sample_df.isna().sum().sum()))
        with col3:
            num_cols = sum(pd.api.types.is_numeric_dtype(sample_df[c]) for c in sample_df.columns)
            st.metric("Numeric features", int(num_cols))

        st.markdown("**Feature glossary**")
        glossary_items = {
            "rating": "Average product rating given by customers (0–5).",
            "price_whole": "Selling price shown to the customer.",
            "mrp": "Maximum Retail Price (list price).",
            "number_of_global_ratings": "Total count of ratings received.",
            "number_of_reviews": "Total count of text reviews.",
            "discount_percentage_cleaned": "Approximate discount percentage.",
            "product_weight_grams": "Pack weight in grams.",
            "category_reclassified": "Simplified product category label.",
            "is_underperforming": "Target label: 1 means likely underperforming (optional in uploads)."
        }
        for k, v in glossary_items.items():
            st.markdown(f"- **{k}**: {v}")
    else:
        st.info("No reference data available to preview. Upload a CSV in the sidebar to explore.")

with st.expander("3) Exploratory Data Analysis (EDA)", expanded=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
    except Exception:
        sns = None

    # Always use local dashboard data for EDA
    eda_df = None
    data_path = os.path.join(BASE_DIR, "data", "haldirams_cleaned.csv")
    if os.path.exists(data_path):
        try:
            eda_df = pd.read_csv(data_path)
        except Exception:
            eda_df = None

    if eda_df is None:
        st.info("No data available for EDA. Upload a CSV to enable charts.")
    else:
        st.markdown("- **What we look at**: basic distributions, top categories, and simple relationships.")
        num_cols = [c for c in eda_df.columns if pd.api.types.is_numeric_dtype(eda_df[c])]

        # Numeric summary
        if num_cols:
            st.markdown("**Numeric summary (describe)**")
            st.dataframe(eda_df[num_cols].describe().T, use_container_width=True)

            # Histogram of a key numeric feature
            key = "rating" if "rating" in num_cols else num_cols[0]
            fig, ax = plt.subplots()
            if sns is not None:
                sns.histplot(eda_df[key].dropna(), kde=True, ax=ax)
            else:
                ax.hist(eda_df[key].dropna(), bins=30, color="#4C78A8")
            ax.set_title(f"Distribution of {key}")
            st.pyplot(fig, clear_figure=True)
            st.caption("Insight: Most products cluster around the typical values; outliers may need special handling.")

        # Top categories by count
        if "category_reclassified" in eda_df.columns:
            cat_counts = eda_df["category_reclassified"].value_counts().head(10)
            fig, ax = plt.subplots()
            cat_counts.sort_values().plot(kind="barh", ax=ax, color="#72B7B2")
            ax.set_title("Top categories by product count")
            st.pyplot(fig, clear_figure=True)
            st.caption("Insight: Heavily represented categories may drive most predictions; ensure balance across groups.")

        # Relationship example: price vs rating
        if set(["price_whole", "rating"]).issubset(eda_df.columns):
            fig, ax = plt.subplots()
            if sns is not None:
                sns.scatterplot(x="price_whole", y="rating", data=eda_df.sample(min(len(eda_df), 1000)), ax=ax, s=30, alpha=0.6)
            else:
                sample_df2 = eda_df.sample(min(len(eda_df), 1000))
                ax.scatter(sample_df2["price_whole"], sample_df2["rating"], s=15, alpha=0.6)
            ax.set_title("Price vs Rating (sample)")
            st.pyplot(fig, clear_figure=True)
            st.caption("Inference: If higher prices correlate with higher/lower ratings, pricing strategy may need revision.")

with st.expander("4) Modeling & Experiment Summary", expanded=True):
    st.markdown("**What the model does (plain-English)**")
    st.markdown("- Learns patterns from past product data to predict 'Underperforming' vs 'Not Underperforming'.")
    st.markdown("- Uses a preprocessing step to clean and encode features before modeling.")

    # Show currently loaded classifier from pipeline (if any)
    if inference_pipeline is not None:
        try:
            model = inference_pipeline.named_steps.get('classifier', None)
            if model is not None:
                st.markdown(f"- Current classifier in production pipeline: `{type(model).__name__}`")
        except Exception:
            pass

    # Load precomputed experiment comparison exported from notebook (JSON or CSV)
    exp_json = os.path.join(ART_DIR, "experiment_results.json")
    exp_csv = os.path.join(ART_DIR, "experiment_results.csv")
    exp_df = None
    if os.path.exists(exp_json):
        try:
            with open(exp_json, "r") as f:
                exp_df = pd.DataFrame(json.load(f))
        except Exception:
            exp_df = None
    elif os.path.exists(exp_csv):
        try:
            exp_df = pd.read_csv(exp_csv)
        except Exception:
            exp_df = None

    core_models = {"LogisticRegression", "DecisionTree", "RandomForest"}
    if exp_df is not None and {"model", "metric", "value"}.issubset(exp_df.columns):
        exp_df = exp_df.copy()
        # Keep only the requested three models if present
        if set(exp_df["model"].unique()) & core_models:
            exp_df = exp_df[exp_df["model"].isin(core_models)]
        st.subheader("Experiment comparison (from notebooks)")
        st.dataframe(exp_df, use_container_width=True)

        # Simple chart: best metric per model (higher is better assumed)
        try:
            pivot = exp_df.pivot_table(index="model", columns="metric", values="value", aggfunc="mean")
            fig, ax = plt.subplots()
            pivot.plot(kind="bar", ax=ax)
            ax.set_title("Model metrics comparison")
            st.pyplot(fig, clear_figure=True)
        except Exception:
            pass

        # Plain-English insights
        st.markdown("**Insights**")
        try:
            best_rows = exp_df.sort_values("value", ascending=False).groupby("metric").head(1)
            for _, r in best_rows.iterrows():
                st.markdown(f"- For **{r['metric']}**, best model is **{r['model']}** (value: {r['value']:.3f}).")
        except Exception:
            pass
        # Explicit selection statement
        st.success("Selected best model: LogisticRegression (based on our comparative analysis and explainability considerations).")
        st.caption("Place `experiment_results.json` or `experiment_results.csv` in `dashboard/artifacts/` with columns: model, metric, value. Export from your notebook to populate this section.")
    else:
        # Friendly fallback: show the three-model comparison narrative (using current notebook results)
        st.subheader("Three-model comparison (from latest notebook run)")
        st.markdown("Below is a concise comparison of the three candidates on the test set. 'Tuned F1' refers to the model after hyperparameter tuning.")

        # Notebook-reported metrics
        comp_df = pd.DataFrame([
            {"model": "LogisticRegression", "accuracy": 0.8992, "baseline_f1_underperf": 0.8286, "tuned_f1_underperf": 0.8514},
            {"model": "DecisionTree",      "accuracy": 0.8824, "baseline_f1_underperf": 0.7778, "tuned_f1_underperf": 0.8058},
            {"model": "RandomForest",       "accuracy": 0.9076, "baseline_f1_underperf": 0.8406, "tuned_f1_underperf": 0.8514},
        ])
        st.dataframe(comp_df, use_container_width=True)

        cols = st.columns(3)
        with cols[0]:
            st.metric("Highest accuracy", "RandomForest", help="RandomForest shows the top overall accuracy (~0.908).")
        with cols[1]:
            st.metric("Top tuned F1 (tie)", "LogReg & RF", help="Both LogisticRegression and RandomForest reach ~0.851 tuned F1 for class 1.")
        with cols[2]:
            st.metric("Operational choice", "LogReg", help="Chosen for simplicity, stability, and ease of explanation.")

        st.markdown("**Why we selected Logistic Regression**")
        st.markdown("- Ties for best tuned F1 on the 'Underperforming' class (~0.851), which we prioritize for catching struggling products.")
        st.markdown("- Demonstrated very strong recall for 'Underperforming' in the tuned evaluation, aligning with business needs to minimize misses.")
        st.markdown("- Simpler and more explainable than ensembles, making decisions easier to communicate and trust.")
        st.success("Selected best model: LogisticRegression")

        st.caption("Tip: You can still export exact metrics via `experiment_results.json`/`.csv` to power the charted comparison automatically.")
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
