"""
AREA-LEVEL cancer model (REAL UK open data).

Builds one analysis table — one row per English upper-tier local authority — by
merging the cached Fingertips indicators, then trains models to predict the
area's under-75 cancer mortality rate from its risk-factor profile
(deprivation, obesity, diabetes, smoking, inactivity, alcohol, age structure).

This answers: "given the kind of area someone grows up in, how heavy is the
cancer burden there?"  It is an ECOLOGICAL model — it describes areas, not
individuals (see the ecological-fallacy note in the README).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
import data_fetch

UTLA_AREA_TYPE = "Counties & UAs (from Apr 2023)"
# Preferred "headline" age bands when an indicator is split by age.
AGE_PRIORITY = [
    "All ages", "18+ yrs", "16+ yrs", "19+ yrs", "18-64 yrs",
    "65+ yrs", "<75 yrs", "40-79 yrs",
]


def _pick_headline(df: pd.DataFrame, friendly: str) -> pd.DataFrame:
    """Reduce a raw Fingertips CSV to one value per local authority."""
    d = df[(df["Area Type"] == UTLA_AREA_TYPE) & (df["Category Type"].isna())].copy()
    if (d["Sex"] == "Persons").any():
        d = d[d["Sex"] == "Persons"]

    ages = list(d["Age"].dropna().unique())
    if len(ages) > 1:
        def rank(a: str):
            coverage = d.loc[d["Age"] == a, "Area Code"].nunique()
            priority = AGE_PRIORITY.index(a) if a in AGE_PRIORITY else 999
            return (priority, -coverage)
        d = d[d["Age"] == sorted(ages, key=rank)[0]]

    d = d.dropna(subset=["Value"])
    d = d.sort_values("Time period Sortable").drop_duplicates("Area Code", keep="last")
    return d[["Area Code", "Area Name", "Value"]].rename(columns={"Value": friendly})


def build_area_dataset(force_fetch: bool = False) -> pd.DataFrame:
    raw = data_fetch.fetch_all(force=force_fetch)

    outcome = config.AREA_OUTCOME["name"]
    merged = _pick_headline(raw[outcome], outcome)
    for friendly in config.AREA_FEATURES.values():
        merged = merged.merge(
            _pick_headline(raw[friendly], friendly),
            on=["Area Code", "Area Name"], how="outer",
        )

    merged = merged[merged[outcome].notna()].reset_index(drop=True)
    out_path = config.DATA_PROCESSED / "area_dataset.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nArea dataset: {len(merged)} local authorities, "
          f"{len(config.AREA_FEATURES)} features -> {out_path}")
    return merged


def _evaluate(name, model, X, y, cv):
    r2 = cross_val_score(model, X, y, cv=cv, scoring="r2")
    mae = -cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")
    print(f"  {name:18s}  R2 = {r2.mean():.3f} (+/-{r2.std():.3f})   "
          f"MAE = {mae.mean():.1f} deaths/100k")
    return r2.mean()


def train_area_model(df: pd.DataFrame) -> None:
    import joblib

    target = config.AREA_OUTCOME["name"]
    feat_names = list(config.AREA_FEATURES.values())

    # Median-impute the handful of missing feature cells (a few small LAs).
    X = df[feat_names].apply(lambda c: c.fillna(c.median()))
    y = df[target].astype(float)
    cv = KFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)

    print("\nArea-level model - 5-fold cross-validation:")
    linear = Pipeline([("scale", StandardScaler()), ("lr", LinearRegression())])
    gbr = GradientBoostingRegressor(random_state=config.RANDOM_SEED)
    r2_lin = _evaluate("Linear regression", linear, X, y, cv)
    r2_gbr = _evaluate("Gradient boosting", gbr, X, y, cv)

    best_name, best = ("Gradient boosting", gbr) if r2_gbr >= r2_lin else ("Linear regression", linear)
    print(f"  -> selected: {best_name}")

    # Out-of-fold predictions for an honest predicted-vs-actual plot.
    oof = cross_val_predict(best, X, y, cv=cv)
    best.fit(X, y)
    joblib.dump({"model": best, "features": feat_names, "target": target},
                config.MODELS / "area_model.joblib")

    _plot_pred_vs_actual(y, oof, best_name)
    _plot_importance(best, X, y, feat_names)
    _save_rankings(df, target, best.predict(X))


def _plot_pred_vs_actual(y, pred, model_name):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y, pred, alpha=0.6, edgecolor="k", linewidth=0.3)
    lo, hi = min(y.min(), pred.min()), max(y.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax.set_xlabel("Actual under-75 cancer mortality (per 100k)")
    ax.set_ylabel("Predicted (cross-validated)")
    ax.set_title(f"Area model: predicted vs actual\n{model_name}, "
                 f"R2={r2_score(y, pred):.2f}")
    fig.tight_layout()
    fig.savefig(config.OUTPUTS / "area_pred_vs_actual.png", dpi=120)
    plt.close(fig)


def _plot_importance(model, X, y, feat_names):
    imp = permutation_importance(model, X, y, n_repeats=20,
                                 random_state=config.RANDOM_SEED)
    order = np.argsort(imp.importances_mean)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(np.array(feat_names)[order], imp.importances_mean[order],
            xerr=imp.importances_std[order], color="#2b8cbe")
    ax.set_xlabel("Permutation importance (drop in R2)")
    ax.set_title("What drives area cancer burden?")
    fig.tight_layout()
    fig.savefig(config.OUTPUTS / "area_feature_importance.png", dpi=120)
    plt.close(fig)


def _save_rankings(df, target, pred):
    out = df[["Area Name", target]].copy()
    out["predicted"] = pred
    out["residual"] = out[target] - out["predicted"]
    out = out.sort_values(target, ascending=False)
    out.to_csv(config.OUTPUTS / "area_predictions.csv", index=False)
    print("\nHighest-burden areas (actual under-75 cancer mortality / 100k):")
    print(out.head(8).to_string(index=False))


def main(force_fetch: bool = False) -> None:
    df = build_area_dataset(force_fetch=force_fetch)
    train_area_model(df)


if __name__ == "__main__":
    import sys
    main(force_fetch="--force" in sys.argv)
