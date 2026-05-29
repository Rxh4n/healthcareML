"""
Interactive UK cancer-risk calculator (Streamlit web app).

Run it:
    cd C:\\Users\\rehan\\healthcareML
    streamlit run app.py

A browser tab opens with a form. Enter a person's details and get their
estimated 10-year cancer risk plus a breakdown of what is driving it.

NOTE: educational/research demo only, trained on synthetic data. NOT medical
advice. See README.md.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

import config
import individual_model

st.set_page_config(page_title="UK Cancer Risk Calculator", layout="centered")

# Friendly label -> model value mappings.
DEPRIVATION = {
    "Most deprived 20% of areas": 1,
    "More deprived than average": 2,
    "Average": 3,
    "Less deprived than average": 4,
    "Least deprived 20% of areas": 5,
}
ALCOHOL = {"Low / none": "low", "Moderate": "moderate", "Heavy": "heavy"}
BAND_COLOR = {"LOW": "#1a9850", "MODERATE": "#fd8d3c", "HIGH": "#d73027"}


def bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "underweight"
    if bmi < 25:
        return "normal"
    if bmi < 30:
        return "overweight"
    return "obese"


@st.cache_resource
def ensure_model():
    """Train the model once if it hasn't been built yet."""
    if not (config.MODELS / "individual_model.joblib").exists():
        with st.spinner("Training the model for the first time (~30s)..."):
            import synthetic_data
            synthetic_data.generate_cohort().to_csv(
                config.DATA_PROCESSED / "synthetic_individuals.csv", index=False)
            individual_model.train()
    return True


def driver_chart(drivers: list[dict]):
    drivers = drivers[:8][::-1]  # biggest at top
    labels = [d["factor"] for d in drivers]
    logs = [d["log_odds"] for d in drivers]
    colors = ["#d73027" if v > 0 else "#1a9850" for v in logs]

    fig, ax = plt.subplots(figsize=(7, 0.5 * len(labels) + 1))
    ax.barh(labels, logs, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    for i, d in enumerate(drivers):
        ax.text(d["log_odds"], i, f"  x{d['multiplier']:.2f}",
                va="center", ha="left" if d["log_odds"] > 0 else "right",
                fontsize=9)
    ax.set_xlabel("Effect on risk (right = increases, left = decreases)")
    ax.set_title("What's driving this estimate\n(vs a typical 50-year-old, mid-deprivation)")
    ax.margins(x=0.2)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.title("UK Cancer Risk Calculator")
st.caption("Estimates a person's 10-year risk of developing cancer from lifestyle, "
           "demographic and area-deprivation factors.")

st.error(
    "**Educational / research demo only — NOT medical advice.** This model is "
    "trained on **synthetic data** calibrated to published UK risk factors. It "
    "cannot diagnose anyone. For real concerns, speak to a GP."
)

ensure_model()

with st.form("risk_form"):
    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("Age", 18, 90, 50)
        sex = st.selectbox("Sex", ["female", "male"])
        ethnicity = st.selectbox("Ethnicity", ["White", "Asian", "Black", "Mixed", "Other"])
        deprivation_label = st.selectbox("Deprivation of your area", list(DEPRIVATION))
        smoking = st.selectbox("Smoking status", ["never", "former", "current"])
    with c2:
        height_cm = st.number_input("Height (cm)", 120, 220, 170)
        weight_kg = st.number_input("Weight (kg)", 35, 250, 75)
        alcohol_label = st.selectbox("Alcohol intake", list(ALCOHOL))
        diabetes = st.checkbox("Diagnosed with diabetes")
        physically_inactive = st.checkbox("Physically inactive (under ~150 min/week)")
        family_history = st.checkbox("Family history of cancer")

    bmi = weight_kg / (height_cm / 100) ** 2
    st.caption(f"Calculated BMI: **{bmi:.1f}** ({bmi_category(bmi)})")
    submitted = st.form_submit_button("Calculate my 10-year risk", type="primary")

if submitted:
    person = {
        "age": age, "sex": sex, "ethnicity": ethnicity,
        "imd_quintile": DEPRIVATION[deprivation_label],
        "smoking": smoking, "bmi_cat": bmi_category(bmi),
        "diabetes": int(diabetes), "alcohol": ALCOHOL[alcohol_label],
        "physically_inactive": int(physically_inactive),
        "family_history": int(family_history),
    }
    result = individual_model.predict_risk(person)
    drivers = individual_model.explain_risk(person)

    color = BAND_COLOR[result["band"]]
    st.markdown(
        f"<div style='text-align:center;padding:1.2em;border-radius:10px;"
        f"background:{color}20;border:2px solid {color}'>"
        f"<div style='font-size:1.1em;color:#444'>Estimated 10-year cancer risk</div>"
        f"<div style='font-size:3.2em;font-weight:700;color:{color}'>{result['risk_pct']}</div>"
        f"<div style='font-size:1.3em;font-weight:600;color:{color}'>{result['band']} risk</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.write("")
    if drivers:
        st.pyplot(driver_chart(drivers))
    st.caption(
        "Risk multipliers are relative to a reference person (age 50, female, "
        "White, never-smoker, normal BMI, average deprivation, no other risk "
        "factors). This is a synthetic-data demonstration, not a diagnosis."
    )
