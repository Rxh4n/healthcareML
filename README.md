# UK Cancer Risk — Healthcare ML

Two complementary models that estimate cancer risk in the UK from socioeconomic,
demographic and lifestyle factors (age, ethnicity, deprivation, obesity,
diabetes, smoking, alcohol, physical activity, family history):

1. **Area-level model** — trained on **real, open UK data**. Predicts the
   *under-75 cancer mortality rate* of every English upper-tier local authority
   from its risk-factor profile. Answers *"how heavy is the cancer burden in the
   kind of area someone grows up in?"*
2. **Individual-level model** — a personal 10-year cancer-risk calculator.
   Trained on **synthetic data calibrated to real published UK relative risks**,
   because individual patient data is access-controlled (see below).

> ## ⚠️ IMPORTANT — read before using
> This is an **educational / research demonstration, NOT a medical device and
> NOT a clinical tool.** It must **not** be used to make decisions about any real
> person's health. The individual model is trained on **synthetic data**; its
> predictions are illustrative, not diagnoses. Real cancer risk assessment
> requires validated clinical tools (e.g. QCancer, CanRisk/BOADICEA) and a
> qualified clinician. No real patient data is used or stored in this project.

---

## Why two models? The data reality

Genuinely **individual-level** UK cancer data (one row per patient, with their
diseases and outcomes) is **not freely downloadable** — it sits behind ethics
and data-access applications:

| Source | What it has | Access |
|---|---|---|
| [UK Biobank](https://www.ukbiobank.ac.uk/) | 500k people, genetics, lifestyle, cancer registry linkage | Application + fee |
| [CPRD](https://www.cprd.com/) | GP records, linked to cancer registry | Application + fee |
| [NHS England / NDRS](https://digital.nhs.uk/ndrs) | National cancer registration | Application (DARS) |

**Area-level** UK data, however, *is* fully open. So the area model uses real
data, and the individual model uses synthetic data whose effect sizes come from
the published literature — with a clean path to swap in real data later.

---

## Real data sources (area model)

All pulled live from the **OHID Fingertips API** (public, no key) for English
upper-tier local authorities. `data_fetch.py` downloads and caches them.

| Role | Indicator | Fingertips ID |
|---|---|---|
| **Outcome** | Under-75 mortality rate from cancer | 40501 |
| Feature | Deprivation score (IMD 2019) | 93553 |
| Feature | Overweight/obesity prevalence, adults | 93088 |
| Feature | Diabetes (QOF) prevalence | 241 |
| Feature | Smoking prevalence, adults 18+ | 92443 |
| Feature | Physically inactive adults | 93015 |
| Feature | Alcohol-related hospital admissions | 93764 |
| Feature | % population aged 65+ | 92310 |

- Fingertips: https://fingertips.phe.org.uk/
- English Indices of Deprivation 2019: https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019
- Cancer Research UK statistics: https://www.cancerresearchuk.org/health-professional/cancer-statistics

> **Note on the outcome:** all-cancer *incidence* is not published free at local-
> authority level, so the area model uses under-75 cancer *mortality* — the
> standard area-level cancer-burden indicator. It is a proxy for "how much cancer
> affects this area", not a direct incidence rate.

### Published risk factors behind the synthetic individuals
The synthetic cohort's cancer outcome is generated from a logistic model whose
odds ratios are *illustrative values drawn from* Cancer Research UK, ONS and
large cohort studies (see `config.ODDS_RATIOS`). They make the data structurally
realistic; they are **not** fitted to a specific dataset.

---

## Install & run

```bash
pip install -r requirements.txt
python run_all.py            # runs both models end to end
python run_all.py --force    # re-download the real area data first
```

Run pieces individually:
```bash
python data_fetch.py         # download + cache real UK area data
python area_model.py         # build area dataset + train area model
python synthetic_data.py     # generate synthetic individuals
python individual_model.py   # train the personal risk model
```

Score one person:
```python
from individual_model import predict_risk
predict_risk({
    "age": 68, "sex": "male", "ethnicity": "White", "imd_quintile": 1,
    "smoking": "current", "bmi_cat": "obese", "diabetes": 1,
    "alcohol": "heavy", "physically_inactive": 1, "family_history": 1,
})
# -> {'risk_10yr': ..., 'risk_pct': '...', 'band': 'HIGH'}
```

---

## Results (typical run)

**Area model** (~151 local authorities, 5-fold CV): **R² ≈ 0.69**, MAE ≈ 8
deaths/100k. Highest-burden areas surfaced are Blackpool, Stoke-on-Trent,
Middlesbrough, Kingston upon Hull — the well-known deprivation–cancer gradient.

**Individual model** (25% hold-out): **ROC AUC ≈ 0.84**, Brier ≈ 0.087
(well calibrated). The model recovers the input odds ratios closely
(e.g. current-smoker ≈ 1.56, family history ≈ 1.80, age ≈ 1.09/yr), confirming
it learns real signal.

Outputs written to `outputs/`:
- `area_pred_vs_actual.png`, `area_feature_importance.png`, `area_predictions.csv`
- `individual_roc.png`, `individual_calibration.png`,
  `individual_pred_vs_truth.png`, `individual_odds_ratios.csv`

---

## ⚠️ The ecological fallacy

The area model describes **areas, not individuals**. An area having high cancer
mortality does **not** mean a given resident is high-risk — inferring individual
risk from area-level associations is the *ecological fallacy*. Use the area model
for population/public-health insight and the individual model for per-person
estimates (and even then, see the disclaimer).

---

## Plugging in real individual data

`individual_model.py` reads the cohort via `synthetic_data.generate_cohort()`.
Once you have approved access (UK Biobank / CPRD), write a loader that returns a
DataFrame with the same columns —
`age, sex, ethnicity, imd_quintile, smoking, bmi_cat, diabetes, alcohol,
physically_inactive, family_history, cancer_10yr` — and point the model at it.
Nothing else in the pipeline changes.

---

## Project layout

```
config.py            paths, indicator IDs, published odds ratios
data_fetch.py        OHID Fingertips API client (real data, cached)
area_model.py        real-data area-level model
synthetic_data.py    synthetic individual cohort (real published risks)
individual_model.py  personal 10-year risk model + predict_risk()
run_all.py           end-to-end runner
data/  models/  outputs/   created on first run
```

## Limitations
- Area outcome is mortality, not incidence (incidence isn't free at LA level).
- Individual model is trained on synthetic data — directions/magnitudes are
  realistic but it is not validated on real patients.
- Ethnicity is only in the individual model; adding it area-side needs Census
  (Nomis) data joined on local-authority codes.
- Associations here are **not causal**.
