"""
Generate a SYNTHETIC cohort of UK individuals for the personal-risk model.

Why synthetic? Real individual-level UK cancer data (UK Biobank, CPRD, NHS
Digital) is only available after an ethics/data-access application, so it cannot
be downloaded here. Instead we simulate people whose:

  * marginal distributions match the UK adult population, and
  * cancer outcome is generated from a logistic model whose coefficients are
    REAL published relative risks (see config.ODDS_RATIOS).

The result is structurally realistic data on which the full modelling pipeline
runs today. To use real data later, replace generate_cohort() with a loader for
your approved dataset that returns the same columns — nothing downstream changes.

Run directly to write a CSV preview:  python synthetic_data.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

RNG = np.random.default_rng(config.RANDOM_SEED)

ETHNICITY = {"White": 0.81, "Asian": 0.096, "Black": 0.04, "Mixed": 0.029, "Other": 0.025}
BMI_BASE = {"underweight": 0.02, "normal": 0.34, "overweight": 0.38, "obese": 0.26}


def _choice(options: dict[str, float], n: int) -> np.ndarray:
    keys = list(options)
    p = np.array([options[k] for k in keys], dtype=float)
    return RNG.choice(keys, size=n, p=p / p.sum())


def generate_cohort(n: int = config.N_SYNTHETIC) -> pd.DataFrame:
    """Return a DataFrame of synthetic individuals + a binary cancer_10yr label."""
    age = np.clip(RNG.normal(50, 18, n), 18, 90).round().astype(int)
    sex = np.where(RNG.random(n) < 0.51, "female", "male")
    ethnicity = _choice(ETHNICITY, n)
    imd_quintile = RNG.integers(1, 6, n)  # 1 = most deprived ... 5 = least

    # --- correlated lifestyle factors -------------------------------------- #
    # More deprived areas -> more smoking, obesity and inactivity (real gradient).
    deprivation_load = (3 - imd_quintile) / 4.0          # +0.5 (q1) .. -0.5 (q5)
    older = (age - 50) / 40.0

    p_current = np.clip(0.13 + 0.12 * deprivation_load, 0.02, 0.45)
    p_former = np.clip(0.30 + 0.10 * older, 0.10, 0.55)
    u = RNG.random(n)
    smoking = np.where(u < p_current, "current",
                       np.where(u < p_current + p_former, "former", "never"))

    p_obese = np.clip(0.26 + 0.12 * deprivation_load + 0.08 * older, 0.05, 0.55)
    bmi_cat = np.empty(n, dtype=object)
    ub = RNG.random(n)
    for i in range(n):
        if ub[i] < p_obese[i]:
            bmi_cat[i] = "obese"
        else:
            bmi_cat[i] = RNG.choice(
                ["underweight", "normal", "overweight"], p=[0.03, 0.45, 0.52]
            )

    # Diabetes risk rises with age and obesity (real conditional structure).
    is_obese = (bmi_cat == "obese").astype(float)
    logit_dia = -3.2 + 0.045 * (age - 50) + 1.0 * is_obese + 0.6 * deprivation_load
    diabetes = (RNG.random(n) < 1 / (1 + np.exp(-logit_dia))).astype(int)

    alcohol = _choice({"low": 0.35, "moderate": 0.45, "heavy": 0.20}, n)

    p_inactive = np.clip(0.22 + 0.12 * deprivation_load + 0.10 * older, 0.05, 0.6)
    physically_inactive = (RNG.random(n) < p_inactive).astype(int)

    family_history = (RNG.random(n) < 0.15).astype(int)

    df = pd.DataFrame({
        "age": age, "sex": sex, "ethnicity": ethnicity, "imd_quintile": imd_quintile,
        "smoking": smoking, "bmi_cat": bmi_cat, "diabetes": diabetes,
        "alcohol": alcohol, "physically_inactive": physically_inactive,
        "family_history": family_history,
    })

    prob = _true_risk(df)
    df["cancer_10yr"] = (RNG.random(n) < prob).astype(int)
    df["_true_prob"] = prob  # for calibration reference only; not a model input
    return df


def _true_risk(df: pd.DataFrame) -> np.ndarray:
    """10-year cancer probability from the published-OR logistic model."""
    OR = config.ODDS_RATIOS
    lp = np.full(len(df), config.BASELINE_INTERCEPT, dtype=float)
    lp += config.AGE_LOG_ODDS_PER_YEAR * (df["age"].to_numpy() - config.AGE_REF)

    def add(col):
        lp_local = df[col].map(lambda v: np.log(OR[col][v])).to_numpy()
        return lp_local

    for col in ["sex", "ethnicity", "smoking", "bmi_cat", "diabetes",
                "imd_quintile", "alcohol", "physically_inactive", "family_history"]:
        lp += add(col)

    return 1.0 / (1.0 + np.exp(-lp))


if __name__ == "__main__":
    cohort = generate_cohort()
    path = config.DATA_PROCESSED / "synthetic_individuals.csv"
    cohort.to_csv(path, index=False)
    print(f"Generated {len(cohort):,} synthetic individuals -> {path}")
    print(f"Overall 10-year cancer incidence: {cohort['cancer_10yr'].mean():.1%}")
    print(f"Mean true risk: {cohort['_true_prob'].mean():.1%} "
          f"(min {cohort['_true_prob'].min():.1%}, max {cohort['_true_prob'].max():.1%})")
    print("\nIncidence by smoking status:")
    print(cohort.groupby("smoking")["cancer_10yr"].mean().round(3).to_string())
    print("\nIncidence by deprivation quintile (1=most deprived):")
    print(cohort.groupby("imd_quintile")["cancer_10yr"].mean().round(3).to_string())
