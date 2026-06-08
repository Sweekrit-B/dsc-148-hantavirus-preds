
# required imports
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.use("Agg")

# Langing Page Configuration
st.set_page_config(
    page_title="Hantavirus Risk Predictor",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Dark clinical background */
.stApp {
    background-color: #0d1117;
    color: #e6edf3;
}

/* Header banner */
.hero-banner {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    border: 1px solid #21262d;
    border-left: 4px solid #f85149;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    border-radius: 0 8px 8px 0;
}
.hero-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem;
    font-weight: 600;
    color: #f85149;
    letter-spacing: -0.02em;
    margin: 0 0 0.4rem 0;
}
.hero-sub {
    font-size: 0.9rem;
    color: #8b949e;
    font-weight: 300;
    margin: 0;
}

/* Section labels */
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    color: #8b949e;
    text-transform: uppercase;
    border-bottom: 1px solid #21262d;
    padding-bottom: 0.5rem;
    margin-bottom: 1.2rem;
}

/* Risk result card */
.risk-card {
    padding: 2rem;
    border-radius: 8px;
    text-align: center;
    margin-bottom: 1.5rem;
    border: 1px solid;
}
.risk-high {
    background: rgba(248, 81, 73, 0.08);
    border-color: #f85149;
}
.risk-medium {
    background: rgba(210, 153, 34, 0.08);
    border-color: #d29922;
}
.risk-low {
    background: rgba(63, 185, 80, 0.08);
    border-color: #3fb950;
}
.risk-pct {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 3.5rem;
    font-weight: 600;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.risk-label {
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.risk-desc {
    font-size: 0.8rem;
    color: #8b949e;
    margin-top: 0.5rem;
}

/* Metric row */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.metric-box {
    flex: 1;
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.3rem;
    font-weight: 600;
    color: #58a6ff;
}
.metric-lbl {
    font-size: 0.72rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
}

/* Input card */
.input-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* Streamlit overrides */
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stNumberInput"] > div > div > input {
    background: #0d1117 !important;
    border-color: #30363d !important;
    color: #e6edf3 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
}
label {
    color: #8b949e !important;
    font-size: 0.8rem !important;
    font-weight: 400 !important;
}
.stButton > button {
    background: #f85149 !important;
    color: white !important;
    border: none !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    width: 100% !important;
    padding: 0.8rem !important;
    font-size: 0.9rem !important;
    border-radius: 6px !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover {
    opacity: 0.85 !important;
}
div[data-testid="stAlert"] {
    background: #161b22 !important;
    border-color: #30363d !important;
    color: #8b949e !important;
}
.disclaimer-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 3px solid #d29922;
    border-radius: 0 6px 6px 0;
    padding: 0.8rem 1rem;
    font-size: 0.78rem;
    color: #8b949e;
    margin-top: 1rem;
}

/* Progress bar override */
div[data-testid="stProgress"] > div > div {
    background: #f85149 !important;
}
</style>
""", unsafe_allow_html=True)


# loading the model
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


# building the feature vector
def build_feature_vector(inputs: dict, feature_cols: list) -> pd.DataFrame:
    """
    Replicates the notebook preprocessing for a single patient:
    ordinal encoding, OHE, symptom binarization, cy_ columns set to NaN.
    Returns a single-row DataFrame aligned to `feature_cols`.
    """
    row = {col: np.nan for col in feature_cols}

    # user choice ordinal encodings
    severity_map = {"Mild": 0, "Moderate": 1, "Severe": 2, "Critical": 3}
    viral_map    = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}

    row["severity_encoded"]    = severity_map[inputs["severity"]]
    row["viral_load_encoded"]  = viral_map[inputs["viral_load"]]

    # numeric patient clinical information
    for key in ["age", "incubation_days", "length_of_stay_days",
                "days_to_diagnosis", "hospitalized", "icu_admission",
                "mechanical_ventilation", "dialysis", "year"]:
        if key in row:
            row[key] = inputs.get(key, np.nan)

    # syndrome one-hot encoding
    for val in ["HPS", "HFRS"]:
        col = f"syndrome_{val}"
        if col in row:
            row[col] = 1.0 if inputs["syndrome"] == val else 0.0

    # sex one-hot encoding
    for val in ["Male", "Female"]:
        col = f"sex_{val}"
        if col in row:
            row[col] = 1.0 if inputs["sex"] == val else 0.0

    # geographic setting one-hot encoding
    for val in ["Rural", "Urban", "Remote", "Peri-urban"]:
        col = f"geographic_setting_{val}"
        if col in row:
            row[col] = 1.0 if inputs["geo_setting"] == val else 0.0

    # treatment protocol one-hot encoding
    for val in ["Supportive care only", "IV fluids + oxygen",
                "Vasopressors + mechanical ventilation", "ECMO + full ICU support",
                "Dialysis + supportive"]:
        col = f"treatment_protocol_{val}"
        if col in row:
            row[col] = 1.0 if inputs["treatment"] == val else 0.0

    # exposure type one-hot encoding
    for val in ["Rural dwelling", "Agricultural work", "Camping/hiking",
                "Cleaning rodent-infested area", "Cruise ship exposure",
                "Close contact with HPS patient"]:
        col = f"exposure_type_{val}"
        if col in row:
            row[col] = 1.0 if inputs["exposure"] == val else 0.0

    # blood type one-hot encoding
    for val in ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]:
        col = f"blood_type_{val}"
        if col in row:
            row[col] = 1.0 if inputs["blood_type"] == val else 0.0

    # symptom flags
    symptom_cols = [c for c in feature_cols if c.startswith("sym_")]
    for sc in symptom_cols:
        symptom_name = sc[4:].replace("_", " ")
        row[sc] = 1.0 if symptom_name in inputs["symptoms"] else 0.0

    #  Contextual / cy_ columns → NaN 
    for col in feature_cols:
        if col.startswith("cy_") or col in ["avg_temperature", "avg_precipitation",
                                             "rural_pop_pct", "urban_pop_pct",
                                             "pop_density", "gni_per_capita",
                                             "life_expectancy"]:
            row[col] = np.nan

    # has_cy_data = 0 since no country-year data provided
    if "has_cy_data" in row:
        row["has_cy_data"] = 0

    df = pd.DataFrame([row])[feature_cols]
    return df


#  SHAP waterfall 
def shap_waterfall(model, X_row: pd.DataFrame, threshold: float):
    try:
        pipeline = model["pipeline"]
        # Transform through imputer + scaler only
        X_transformed = pd.DataFrame(
            pipeline[:-1].transform(X_row),
            columns=X_row.columns
        )
        clf = pipeline.named_steps["classifier"]
        explainer = shap.TreeExplainer(clf)
        sv = explainer(X_transformed)

        # Handle 3-D (multi-output) SHAP arrays
        if sv.values.ndim == 3:
            vals    = sv.values[0, :, 1]
            base    = float(sv.base_values[0, 1]) if sv.base_values.ndim > 1 else float(sv.base_values[0])
        else:
            vals    = sv.values[0]
            base    = float(sv.base_values[0])

        # Keep only top-15 by absolute magnitude
        top_idx  = np.argsort(np.abs(vals))[-15:][::-1]
        top_vals = vals[top_idx]
        top_names = X_row.columns[top_idx].tolist()

        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor("#161b22")
        ax.set_facecolor("#161b22")

        colors = ["#f85149" if v > 0 else "#3fb950" for v in top_vals]
        y_pos = np.arange(len(top_names))
        ax.barh(y_pos, top_vals, color=colors, height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_names, fontsize=8, color="#c9d1d9",
                           fontfamily="monospace")
        ax.set_xlabel("SHAP value (impact on mortality probability)", fontsize=8,
                      color="#8b949e")
        ax.tick_params(colors="#8b949e", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.axvline(0, color="#30363d", linewidth=0.8)
        ax.set_title("Feature contributions for this patient",
                     fontsize=9, color="#8b949e", pad=10)
        plt.tight_layout()
        return fig
    except Exception as e:
        return None


#  Symptom list 
ALL_SYMPTOMS = [
    "Myalgia", "Nausea", "Hypotension", "Abdominal Pain", "Fever", "Headache",
    "Backache", "Conjunctival Injection", "Facial Flushing", "Nosebleed",
    "Proteinuria", "Oliguria", "Blurred Vision", "Chills", "Fatigue", "Diarrhea",
    "Tachycardia", "Vomiting", "Dizziness", "Dyspnea", "Cough",
    "Pulmonary Edema", "Petechiae"
]

#  Model performance constants 
MODEL_STATS = {
    "LightGBM":           {"f2": 0.5957, "mcc": 0.3961, "threshold": 0.12},
    "XGBoost":            {"f2": 0.5850, "mcc": 0.3941, "threshold": 0.11},
    "Random Forest":      {"f2": 0.5848, "mcc": 0.3919, "threshold": 0.20},
    "Logistic Regression":{"f2": 0.5978, "mcc": 0.3992, "threshold": 0.51},
    "Baseline (LR)":      {"f2": 0.2593, "mcc": 0.2927, "threshold": 0.50},
}

#  User Interface 

# Header
st.markdown("""
<div class="hero-banner">
  <p class="hero-title">🦠 Hantavirus Mortality Risk Predictor</p>
  <p class="hero-sub">DSC 148 · Bhatnagar & Tadepalli · UCSD &nbsp;|&nbsp;
     LightGBM classifier trained on synthetic global epidemiological data</p>
</div>
""", unsafe_allow_html=True)

model_bundle = load_model()

# Column layout 
left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.markdown('<div class="section-label">Patient Information</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        syndrome    = st.selectbox("Syndrome type", ["HPS", "HFRS"])
        severity    = st.selectbox("Disease severity",
                                   ["Mild", "Moderate", "Severe", "Critical"])
        viral_load  = st.selectbox("Viral load category",
                                   ["Low", "Medium", "High", "Critical"])
        sex         = st.selectbox("Sex", ["Male", "Female"])
        blood_type  = st.selectbox("Blood type",
                                   ["O+", "A+", "B+", "AB+", "O-", "A-", "B-", "AB-"])
    with c2:
        age              = st.number_input("Age", min_value=0, max_value=110, value=45)
        incubation_days  = st.number_input("Incubation days", min_value=1, max_value=60, value=14)
        los              = st.number_input("Length of stay (days)", min_value=0, max_value=120, value=7)
        days_to_dx       = st.number_input("Days to diagnosis", min_value=0, max_value=30, value=3)
        year             = st.number_input("Year", min_value=2000, max_value=2030, value=2026)

    st.markdown('<div class="section-label" style="margin-top:1.2rem">Clinical Interventions</div>',
                unsafe_allow_html=True)
    ci1, ci2, ci3 = st.columns(3)
    with ci1:
        hospitalized = st.checkbox("Hospitalized", value=True)
        icu          = st.checkbox("ICU admission", value=False)
    with ci2:
        mech_vent    = st.checkbox("Mechanical ventilation", value=False)
        dialysis     = st.checkbox("Dialysis", value=False)
    with ci3:
        treatment = st.selectbox("Treatment protocol",
                                 ["Supportive care only",
                                  "IV fluids + oxygen",
                                  "Vasopressors + mechanical ventilation",
                                  "ECMO + full ICU support",
                                  "Dialysis + supportive"])

    st.markdown('<div class="section-label" style="margin-top:1.2rem">Exposure & Setting</div>',
                unsafe_allow_html=True)
    ex1, ex2 = st.columns(2)
    with ex1:
        exposure = st.selectbox("Exposure type",
                                ["Rural dwelling", "Agricultural work",
                                 "Camping/hiking", "Cleaning rodent-infested area",
                                 "Cruise ship exposure", "Close contact with HPS patient"])
    with ex2:
        geo_setting = st.selectbox("Geographic setting",
                                   ["Rural", "Urban", "Remote", "Peri-urban"])

    st.markdown('<div class="section-label" style="margin-top:1.2rem">Symptoms</div>',
                unsafe_allow_html=True)
    symptoms = st.multiselect("Select all that apply", ALL_SYMPTOMS,
                              default=["Fever", "Myalgia"])

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("⚡  Predict Mortality Risk")

# Results 
with right:
    st.markdown('<div class="section-label">Risk Assessment</div>', unsafe_allow_html=True)

    if not predict_btn:
        st.markdown("""
        <div style="background:#161b22;border:1px dashed #30363d;border-radius:8px;
                    padding:3rem 2rem;text-align:center;color:#8b949e;">
            <div style="font-size:2.5rem;margin-bottom:1rem">🩺</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.85rem">
                Fill in patient details and click<br><b>Predict Mortality Risk</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        inputs = dict(
            syndrome=syndrome, severity=severity, viral_load=viral_load,
            sex=sex, blood_type=blood_type, age=age,
            incubation_days=incubation_days, length_of_stay_days=los,
            days_to_diagnosis=days_to_dx, year=year,
            hospitalized=int(hospitalized), icu_admission=int(icu),
            mechanical_ventilation=int(mech_vent), dialysis=int(dialysis),
            treatment=treatment, exposure=exposure, geo_setting=geo_setting,
            symptoms=symptoms,
        )

        if model_bundle is None:
            # Demo mode: rule-based approximation 
            sev_score   = {"Mild": 0.04, "Moderate": 0.12, "Severe": 0.30, "Critical": 0.52}[severity]
            viral_score = {"Low": 0.0, "Medium": 0.02, "High": 0.05, "Critical": 0.10}[viral_load]
            icu_adj     = 0.08 if icu else 0.0
            mv_adj      = 0.10 if mech_vent else 0.0
            hps_adj     = 0.05 if syndrome == "HPS" else 0.0
            age_adj     = max(0, (age - 40) * 0.003)
            prob = min(0.97, sev_score + viral_score + icu_adj + mv_adj + hps_adj + age_adj)
            model_name  = "Demo (rule-based)"
            threshold   = 0.12
            is_demo     = True
        else:
            pipeline    = model_bundle["pipeline"]
            feature_cols = model_bundle["feature_cols"]
            threshold   = model_bundle.get("threshold", 0.12)
            model_name  = model_bundle.get("model_name", "LightGBM")
            X_row       = build_feature_vector(inputs, feature_cols)
            prob        = float(pipeline.predict_proba(X_row)[0, 1])
            is_demo     = False

        # Risk tier
        if prob >= 0.40:
            tier, tier_cls, tier_desc = "HIGH RISK", "risk-high", \
                "Immediate clinical attention recommended"
            color = "#f85149"
        elif prob >= 0.20:
            tier, tier_cls, tier_desc = "MODERATE RISK", "risk-medium", \
                "Careful monitoring and prompt escalation if deterioration"
            color = "#d29922"
        else:
            tier, tier_cls, tier_desc = "LOW RISK", "risk-low", \
                "Standard monitoring protocols apply"
            color = "#3fb950"

        st.markdown(f"""
        <div class="risk-card {tier_cls}">
            <div class="risk-pct" style="color:{color}">{prob*100:.1f}%</div>
            <div class="risk-label" style="color:{color}">{tier}</div>
            <div class="risk-desc">{tier_desc}</div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(min(prob, 1.0))

        # Model metrics
        stats = MODEL_STATS.get(model_name, MODEL_STATS["LightGBM"])
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-box">
                <div class="metric-val">{stats['f2']:.3f}</div>
                <div class="metric-lbl">F₂ Score</div>
            </div>
            <div class="metric-box">
                <div class="metric-val">{stats['mcc']:.3f}</div>
                <div class="metric-lbl">MCC</div>
            </div>
            <div class="metric-box">
                <div class="metric-val">{stats['threshold']:.2f}</div>
                <div class="metric-lbl">Threshold</div>
            </div>
            <div class="metric-box">
                <div class="metric-val" style="font-size:0.85rem">{model_name.replace(' ', '<br>')}</div>
                <div class="metric-lbl">Model</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # SHAP explanation (only if real model loaded)
        if not is_demo and model_bundle is not None:
            st.markdown('<div class="section-label" style="margin-top:1rem">Feature Contributions (SHAP)</div>',
                        unsafe_allow_html=True)
            X_row = build_feature_vector(inputs, feature_cols)
            fig   = shap_waterfall(model_bundle, X_row, threshold)
            if fig:
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                st.caption("SHAP explanation unavailable for this model type.")
        elif is_demo:
            st.markdown('<div class="section-label" style="margin-top:1rem">Key Risk Drivers</div>',
                        unsafe_allow_html=True)
            # Manual approximate contributions for demo mode
            contribs = {
                "severity_encoded":          {"Mild":-.15,"Moderate":-.06,"Severe":.12,"Critical":.28}[severity],
                "syndrome_HPS":              0.05 if syndrome=="HPS" else -0.01,
                "icu_admission":             0.08 if icu else 0.0,
                "mechanical_ventilation":    0.10 if mech_vent else 0.0,
                "viral_load_encoded":        {"Low":-.04,"Medium":0,"High":.03,"Critical":.07}[viral_load],
                "age":                       max(0, (age-40)*0.003),
            }
            top = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:6]

            fig2, ax2 = plt.subplots(figsize=(6, 3.5))
            fig2.patch.set_facecolor("#161b22")
            ax2.set_facecolor("#161b22")
            names = [t[0] for t in top]
            vals2 = [t[1] for t in top]
            colors2 = ["#f85149" if v > 0 else "#3fb950" for v in vals2]
            ax2.barh(names, vals2, color=colors2, height=0.5)
            ax2.axvline(0, color="#30363d", linewidth=0.8)
            ax2.tick_params(colors="#8b949e", labelsize=8)
            ax2.set_xlabel("Approximate contribution", fontsize=8, color="#8b949e")
            for sp in ax2.spines.values(): sp.set_edgecolor("#30363d")
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

        # Binary verdict
        verdict = "⚠️ MORTALITY PREDICTED" if prob >= threshold else "✅ SURVIVAL PREDICTED"
        vcolor  = "#f85149" if prob >= threshold else "#3fb950"
        st.markdown(f"""
        <div style="text-align:center;margin-top:1rem;padding:0.8rem;
                    background:#161b22;border:1px solid #30363d;border-radius:6px;">
            <span style="font-family:'IBM Plex Mono',monospace;font-size:0.85rem;
                         font-weight:600;color:{vcolor}">{verdict}</span>
            <span style="font-size:0.75rem;color:#8b949e;margin-left:0.8rem">
                @ threshold = {threshold:.2f}</span>
        </div>
        """, unsafe_allow_html=True)

        if is_demo:
            st.markdown("""
            <div class="disclaimer-box">
                ⚠️ <b>Demo mode</b> — <code>model.pkl</code> not found.
                Predictions use a rule-based approximation only.
                Run <code>save_model.py</code> from your notebook environment first.
            </div>
            """, unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div class="disclaimer-box" style="margin-top:1.5rem">
        <b>Research use only.</b> This tool was built on <i>synthetic</i> data and is intended
        solely for academic demonstration. It must not be used for real clinical decisions.
    </div>
    """, unsafe_allow_html=True)

# ── Bottom: model comparison table ────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-label">Model Comparison</div>', unsafe_allow_html=True)

rows = []
for name, s in MODEL_STATS.items():
    rows.append({"Model": name, "F₂ Score": s["f2"], "MCC": s["mcc"],
                 "Threshold": s["threshold"],
                 "F₂ + MCC": round(s["f2"] + s["mcc"], 4)})

df_stats = pd.DataFrame(rows).set_index("Model")
st.dataframe(
    df_stats.style
        .highlight_max(subset=["F₂ Score", "MCC", "F₂ + MCC"], color="#1c3a1c")
        .format({"F₂ Score": "{:.4f}", "MCC": "{:.4f}",
                 "Threshold": "{:.2f}", "F₂ + MCC": "{:.4f}"}),
    use_container_width=True,
)
