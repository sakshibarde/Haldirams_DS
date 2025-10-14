🧭 Responsible AI Checklist
Project: Haldiram's Underperforming Product Predictor
Version: 1.0 | Date: 2025-10-15
Owners: Data Science, Product Management, Marketing

1️⃣ Purpose & Scope
Goal: Predict if a product is likely to be Underperforming or Not Underperforming.

Use Case: Internal analytics for product management, marketing strategy, and inventory planning.

Not for: Making automated decisions about pricing, delisting products, or any decisions that directly impact customers or suppliers without human review.

2️⃣ Data Governance
✅ No personal identifiers (name, phone, address, email) are used.

✅ Columns are limited to public product metadata (rating, price_whole, category_reclassified, etc.) and aggregated sales/review counts.

✅ A representative dataset sample is documented in artifacts/reference_sample.csv.

✅ Model artifacts are versioned and tracked using MLflow and stored in a Git repository.

3️⃣ Fairness
Sensitive Attributes
Primary: category_reclassified (grouped into Gift/Combo vs. Other for audit).

Rationale: To ensure the model does not unfairly penalize specific product categories, such as seasonal gift packs.

Metrics
Demographic Parity Difference: ≤ 0.15

Equalized Odds Difference: ≤ 0.15

Action:

Green ≤ 0.15 → ✅ Fair

Amber 0.15–0.25 → ⚠️ Review Required

Red > 0.25 → ❌ Mitigation Needed

Mitigation
Pre-processing: Reweight samples during training to give more importance to underrepresented product categories.

In-processing: Use Fairlearn algorithms like ExponentiatedGradient to train a model that optimizes for both accuracy and fairness.

Post-processing: Apply different prediction thresholds for each product category group to equalize error rates.

Monitoring
Run a quarterly fairness audit using the "⚖️ Fairness Audit" tab in the Streamlit dashboard.

Set up alerts if the demographic parity difference exceeds 0.25 in production monitoring.

4️⃣ Explainability
Global explainability is provided via a SHAP summary plot to show the main drivers of underperformance across all products.

Local explainability for individual predictions is provided via LIME.

SHAP visualizations are available in the "🔎 SHAP Explanations" tab of the Streamlit dashboard.

Each explanation includes a disclaimer:

Explanations are statistical approximations based on the model's learned patterns and do not represent direct causal relationships.

5️⃣ Privacy & Consent
PII: None used or stored. The review_text column was not used for modeling to avoid processing user-generated content.

Consent: Data is derived from public product listings and aggregated sales data. The use of this data for internal analytics is standard and ethical.

Access: Controlled via GitHub repository permissions. The deployed API is for internal use only.

Secrets: No secrets or API keys are required for this model's operation.

6️⃣ Drift & Monitoring
Drift is tracked using PSI (Population Stability Index) and the Kolmogorov-Smirnov (KS) test in the Streamlit "🌊 Data Drift" tab.

Heuristic thresholds for alerting:

PSI ≥ 0.2 → High Drift (retraining may be required)

PSI 0.1–0.2 → Medium Drift (monitor closely)

PSI < 0.1 → Stable

7️⃣ Safety & Misuse Prevention
Predictions are for internal dashboards only to support human decision-making.

The model's output (a risk score) is a recommendation, not an automated action.

Human validation is required by a product manager before any action (e.g., a marketing campaign, inventory adjustment) is taken based on a prediction.

Misuse prevention is handled by keeping the API and dashboard for internal access only.

8️⃣ Responsible Deployment
✅ All tests (pytest) and lint checks (flake8) passed in the GitHub Actions CI/CD pipeline.

✅ No PII is included in any saved model artifacts.

✅ Fairness metrics were within defined thresholds during the final