"""
INDIVIDUAL-level 10-year cancer-risk model.

Trains on the synthetic cohort (synthetic_data.py) and produces:
  * a calibrated risk classifier (logistic regression + gradient boosting),
  * discrimination/calibration metrics (ROC AUC, Brier score),
  * an odds-ratio table that recovers the known input effects (a sanity check
    that the pipeline learns real signal), and
  * predict_risk(): a function that scores a single person.

Swap generate_cohort() for a real-data loader (same columns) to retrain on
UK Biobank / CPRD once you have access.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import config
import synthetic_data

CAT = ["sex", "ethnicity", "smoking", "bmi_cat", "alcohol"]
NUM = ["age", "imd_quintile", "diabetes", "physically_inactive", "family_history"]
FEATURES = CAT + NUM
TARGET = "cancer_10yr"

# Reference categories (must match config.ODDS_RATIOS references).
CAT_DROP = ["female", "White", "never", "normal", "low"]


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("cat", OneHotEncoder(drop=CAT_DROP, sparse_output=False), CAT),
        ("num", "passthrough", NUM),
    ])


def _load() -> pd.DataFrame:
    path = config.DATA_PROCESSED / "synthetic_individuals.csv"
    if path.exists():
        return pd.read_csv(path)
    return synthetic_data.generate_cohort()


def train() -> dict:
    df = _load()
    X, y = df[FEATURES], df[TARGET].astype(int)
    true_prob = df["_true_prob"] if "_true_prob" in df else None

    Xtr, Xte, ytr, yte, *rest = train_test_split(
        X, y, *( [true_prob] if true_prob is not None else [] ),
        test_size=0.25, random_state=config.RANDOM_SEED, stratify=y,
    )
    tp_te = rest[1] if len(rest) >= 2 else None  # rest = [train_slice, test_slice]

    logistic = Pipeline([
        ("prep", _preprocessor()),
        ("clf", LogisticRegression(max_iter=2000)),
    ])
    gbm = Pipeline([
        ("prep", _preprocessor()),
        ("clf", GradientBoostingClassifier(random_state=config.RANDOM_SEED)),
    ])

    print("\nIndividual model - hold-out evaluation (25% test set):")
    results = {}
    for name, model in [("Logistic regression", logistic), ("Gradient boosting", gbm)]:
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yte, p)
        brier = brier_score_loss(yte, p)
        results[name] = {"model": model, "auc": auc, "brier": brier, "proba": p}
        print(f"  {name:20s}  ROC AUC = {auc:.3f}   Brier = {brier:.4f}")

    best_name = max(results, key=lambda k: results[k]["auc"])
    best = results[best_name]["model"]
    print(f"  -> selected for predict_risk(): {best_name}")

    joblib.dump({"model": best, "features": FEATURES}, config.MODELS / "individual_model.joblib")

    _plot_roc(yte, results)
    _plot_calibration(yte, results[best_name]["proba"], best_name)
    if tp_te is not None:
        _plot_vs_truth(tp_te.to_numpy(), results[best_name]["proba"], best_name)
    _odds_ratio_table(logistic)
    _print_personas(best)
    return results


def _plot_roc(yte, results):
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(yte, r["proba"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={r['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Individual risk model - ROC"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(config.OUTPUTS / "individual_roc.png", dpi=120)
    plt.close(fig)


def _plot_calibration(yte, proba, name):
    frac_pos, mean_pred = calibration_curve(yte, proba, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax.plot(mean_pred, frac_pos, "o-", label=name)
    ax.set_xlabel("Mean predicted risk"); ax.set_ylabel("Observed incidence")
    ax.set_title("Calibration (reliability) curve"); ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(config.OUTPUTS / "individual_calibration.png", dpi=120)
    plt.close(fig)


def _plot_vs_truth(true_prob, pred, name):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(true_prob, pred, s=6, alpha=0.2)
    ax.plot([0, 1], [0, 1], "r--", lw=1)
    ax.set_xlabel("True (data-generating) risk")
    ax.set_ylabel("Model-predicted risk")
    ax.set_title(f"Predicted vs true risk - {name}")
    fig.tight_layout(); fig.savefig(config.OUTPUTS / "individual_pred_vs_truth.png", dpi=120)
    plt.close(fig)


def _odds_ratio_table(logistic: Pipeline):
    """Recovered odds ratios vs the known inputs - a learn-real-signal check."""
    logistic.fit(*_load_xy())
    names = logistic.named_steps["prep"].get_feature_names_out()
    coefs = logistic.named_steps["clf"].coef_[0]
    rows = []
    for n, c in zip(names, coefs):
        clean = n.replace("cat__", "").replace("num__", "")
        rows.append((clean, float(np.exp(c))))
    table = pd.DataFrame(rows, columns=["feature", "odds_ratio_recovered"])
    table["odds_ratio_recovered"] = table["odds_ratio_recovered"].round(3)
    table.to_csv(config.OUTPUTS / "individual_odds_ratios.csv", index=False)
    print("\nRecovered odds ratios (illustrative effects the model learned):")
    print(table.to_string(index=False))


def _load_xy():
    df = _load()
    return df[FEATURES], df[TARGET].astype(int)


# --------------------------------------------------------------------------- #
# Scoring a single person
# --------------------------------------------------------------------------- #
def _band(p: float) -> str:
    return "LOW" if p < 0.08 else "MODERATE" if p < 0.18 else "HIGH"


def predict_risk(person: dict) -> dict:
    """Return 10-year cancer risk for one person.

    person example:
        {"age": 68, "sex": "male", "ethnicity": "White", "imd_quintile": 1,
         "smoking": "current", "bmi_cat": "obese", "diabetes": 1,
         "alcohol": "heavy", "physically_inactive": 1, "family_history": 1}
    """
    bundle = joblib.load(config.MODELS / "individual_model.joblib")
    row = pd.DataFrame([{k: person[k] for k in bundle["features"]}])
    p = float(bundle["model"].predict_proba(row)[:, 1][0])
    return {"risk_10yr": round(p, 4), "risk_pct": f"{p:.1%}", "band": _band(p)}


PERSONAS = {
    "Low-risk young adult": {
        "age": 30, "sex": "female", "ethnicity": "Asian", "imd_quintile": 5,
        "smoking": "never", "bmi_cat": "normal", "diabetes": 0,
        "alcohol": "low", "physically_inactive": 0, "family_history": 0},
    "Average middle-aged adult": {
        "age": 55, "sex": "male", "ethnicity": "White", "imd_quintile": 3,
        "smoking": "former", "bmi_cat": "overweight", "diabetes": 0,
        "alcohol": "moderate", "physically_inactive": 0, "family_history": 0},
    "High-risk older adult (deprived area)": {
        "age": 70, "sex": "male", "ethnicity": "White", "imd_quintile": 1,
        "smoking": "current", "bmi_cat": "obese", "diabetes": 1,
        "alcohol": "heavy", "physically_inactive": 1, "family_history": 1},
}


def _print_personas(model):
    print("\nExample individual predictions:")
    for label, person in PERSONAS.items():
        row = pd.DataFrame([person])[FEATURES]
        p = float(model.predict_proba(row)[:, 1][0])
        print(f"  {label:40s}  {p:5.1%}  [{_band(p)}]")


if __name__ == "__main__":
    train()
