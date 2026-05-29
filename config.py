"""
Central configuration: paths, real-data indicator specs, and the published
epidemiological effect sizes used to generate realistic synthetic individuals.

Nothing in here is secret or environment-specific, so it is safe to commit.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"

for _p in (DATA_RAW, DATA_PROCESSED, MODELS, OUTPUTS):
    _p.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

# --------------------------------------------------------------------------- #
# AREA-LEVEL model: real UK open data (OHID Fingertips)
# --------------------------------------------------------------------------- #
# Geography: "Upper tier local authorities (post 4/23)".
FINGERTIPS_AREA_TYPE = 502        # upper-tier local authority
FINGERTIPS_PARENT_AREA_TYPE = 15  # England
FINGERTIPS_BASE = "https://fingertips.phe.org.uk/api"

# Outcome: all-cancer incidence is NOT published free at local-authority level,
# so we use the standard area-level cancer-burden indicator instead: the
# under-75 cancer mortality rate (Public Health Outcomes Framework). This is a
# well-established proxy for "how much cancer affects people in this area".
AREA_OUTCOME = {
    "id": 40501,
    "name": "cancer_mortality_u75",
    "label": "Under-75 mortality rate from cancer (per 100,000)",
}

# Risk-factor features, all confirmed available at area type 502.
AREA_FEATURES = {
    93553: "imd_deprivation_score",   # Deprivation score (IMD 2019)
    93088: "obesity_pct",             # Overweight (incl. obesity) prevalence, adults
    241:   "diabetes_pct",            # Diabetes: QOF prevalence
    92443: "smoking_pct",             # Smoking prevalence, adults 18+ (APS)
    93015: "physically_inactive_pct", # Percentage of physically inactive adults
    93764: "alcohol_admissions",      # Alcohol-related hospital admissions (narrow)
    92310: "pct_aged_65_plus",        # % population aged 65+
}

# --------------------------------------------------------------------------- #
# INDIVIDUAL-level model: synthetic data calibrated to published UK relative
# risks. These are ILLUSTRATIVE effect sizes drawn from the epidemiological
# literature (Cancer Research UK, ONS, and large cohort studies). They are NOT
# fitted to a specific patient dataset; their purpose is to make the synthetic
# data structurally realistic so the pipeline is meaningful and ready to be
# re-pointed at real data (UK Biobank / CPRD) later.
#
# Values are odds ratios (OR) for developing any cancer within 10 years,
# relative to the reference person:
#   age 50, female, White, never-smoker, normal BMI (18.5-25), no diabetes,
#   IMD quintile 3 (mid), light/no alcohol, physically active, no family history.
# --------------------------------------------------------------------------- #
N_SYNTHETIC = 60_000

# Continuous age effect (log-odds added per year above/below 50).
# Cancer incidence roughly doubles every ~8-9 years in mid-late life.
AGE_REF = 50
AGE_LOG_ODDS_PER_YEAR = 0.085

# Categorical odds ratios (reference category has OR 1.0 and is omitted).
ODDS_RATIOS = {
    "sex": {"female": 1.00, "male": 1.18},
    "ethnicity": {  # all-cancer incidence is highest in the White group in the UK
        "White": 1.00, "Asian": 0.70, "Black": 0.78, "Mixed": 0.85, "Other": 0.85,
    },
    "smoking": {"never": 1.00, "former": 1.25, "current": 1.55},
    "bmi_cat": {"normal": 1.00, "overweight": 1.10, "obese": 1.22, "underweight": 1.05},
    "diabetes": {0: 1.00, 1: 1.15},
    "imd_quintile": {  # 1 = most deprived ... 5 = least deprived
        1: 1.28, 2: 1.15, 3: 1.00, 4: 0.95, 5: 0.90,
    },
    "alcohol": {"low": 1.00, "moderate": 1.08, "heavy": 1.22},
    "physically_inactive": {0: 1.00, 1: 1.12},
    "family_history": {0: 1.00, 1: 1.80},
}

# Intercept sets the baseline 10-year risk for the reference person.
# Chosen so the simulated population mean 10-year incidence is realistic (~8-10%).
BASELINE_INTERCEPT = -3.15
