# 🧭 Responsible AI Checklist  
**Project:** Haldiram's Underperforming Product Predictor  
**Version:** 1.0 | **Date:** 2025-10-15  
**Owners:** Data Science, Product Management, Marketing  

---

## 1️⃣ Purpose & Scope  

**Goal:** Predict if a product is likely to be *Underperforming* or *Not Underperforming*.  

**Use Case:**  
- Internal analytics for product management, marketing strategy, and inventory planning.  

**Not for:**  
- Automated decisions about pricing, delisting products, or any decisions directly impacting customers or suppliers without human review.  

---

## 2️⃣ Data Governance  

✅ No personal identifiers (name, phone, address, email) are used.  
✅ Columns limited to public product metadata (`rating`, `price_whole`, `category_reclassified`, etc.) and aggregated sales/review counts.  
✅ Representative dataset sample documented in `artifacts/reference_sample.csv`.  
✅ Model artifacts versioned and tracked using **MLflow**, stored in a **Git repository**.  

---

## 3️⃣ Fairness  

**Sensitive Attributes:**  
- Primary: `category_reclassified` (grouped into *Gift/Combo* vs. *Other* for audit).  

**Rationale:**  
Ensure the model does not unfairly penalize specific product categories (e.g., seasonal gift packs).  

**Metrics:**  
| Metric | Threshold | Meaning |
|---------|------------|----------|
| Demographic Parity Difference | ≤ 0.15 | ✅ Fair |
| Equalized Odds Difference | ≤ 0.15 | ✅ Fair |

**Action Guide:**  
- **Green ≤ 0.15** → ✅ Fair  
- **Amber 0.15–0.25** → ⚠️ Review Required  
- **Red > 0.25** → ❌ Mitigation Needed  

**Mitigation Strategies:**  
- **Pre-processing:** Reweight samples during training to emphasize underrepresented categories.  
- **In-processing:** Use **Fairlearn ExponentiatedGradient** to optimize accuracy + fairness.  
- **Post-processing:** Adjust prediction thresholds per category group to equalize error rates.  

**Monitoring:**  
- Run quarterly fairness audits in the *⚖️ Fairness Audit* tab (Streamlit dashboard).  
- Trigger alerts if demographic parity difference > 0.25 in production.  

---

## 4️⃣ Explainability  

- **Global Explainability:** SHAP summary plots showing top factors influencing underperformance.  
- **Local Explainability:** LIME explanations for individual predictions.  
- SHAP visualizations available in the *🔎 SHAP Explanations* tab of the Streamlit dashboard.  

**Disclaimer:**  
> Explanations are statistical approximations based on the model’s learned patterns and do not represent direct causal relationships.  

---

## 5️⃣ Privacy & Consent  

- **PII:** None used or stored.  
- `review_text` column excluded from modeling to avoid processing user-generated content.  
- **Consent:** Data sourced from public listings and aggregated sales data — ethical and standard for internal analytics.  
- **Access:** Controlled via GitHub repository permissions; internal API only.  
- **Secrets:** None required for model operation.  

---

## 6️⃣ Drift & Monitoring  

**Metrics Tracked:**  
- **PSI (Population Stability Index)**  
- **Kolmogorov–Smirnov (KS) Test**  

**Thresholds:**  
| PSI Value | Drift Level | Action |
|------------|--------------|--------|
| ≥ 0.2 | 🔴 High Drift | Retraining recommended |
| 0.1–0.2 | 🟠 Medium Drift | Monitor closely |
| < 0.1 | 🟢 Stable | No action needed |

Drift metrics visualized in *🌊 Data Drift* tab (Streamlit dashboard).  

---

## 7️⃣ Safety & Misuse Prevention  

- Predictions used **only for internal dashboards** to support human decision-making.  
- Model outputs a **risk score** — a recommendation, not an automated decision.  
- Human validation (Product Manager) required before any action (e.g., campaign, inventory adjustment).  
- API and dashboard restricted to **internal access only**.  

---

## 8️⃣ Responsible Deployment  

✅ All **pytest** and **flake8** checks passed via GitHub Actions CI/CD.  
✅ No PII included in saved model artifacts.  
✅ Fairness metrics within defined thresholds at final deployment.  
