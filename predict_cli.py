"""
Terminal cancer-risk calculator (zero dependencies beyond the project).

Run it:
    cd C:\\Users\\rehan\\healthcareML
    python predict_cli.py

It asks a few questions, then prints the estimated 10-year cancer risk and the
factors driving it. Press Enter to accept the [default] for any question.

NOTE: educational/research demo only, trained on synthetic data. NOT medical
advice. See README.md.
"""
from __future__ import annotations

import config
import individual_model

DEPRIVATION_HELP = "1=most deprived 20% ... 5=least deprived 20%"


def ask_choice(prompt: str, options: list[str], default: str) -> str:
    opts = "/".join(options)
    while True:
        raw = input(f"{prompt} ({opts}) [{default}]: ").strip().lower()
        if not raw:
            return default
        for o in options:
            if raw == o.lower():
                return o
        print(f"  please type one of: {opts}")


def ask_int(prompt: str, lo: int, hi: int, default: int) -> int:
    while True:
        raw = input(f"{prompt} ({lo}-{hi}) [{default}]: ").strip()
        if not raw:
            return default
        try:
            v = int(raw)
            if lo <= v <= hi:
                return v
        except ValueError:
            pass
        print(f"  please enter a whole number between {lo} and {hi}")


def ask_yesno(prompt: str, default: bool = False) -> int:
    d = "y" if default else "n"
    while True:
        raw = input(f"{prompt} (y/n) [{d}]: ").strip().lower()
        if not raw:
            return int(default)
        if raw in ("y", "yes"):
            return 1
        if raw in ("n", "no"):
            return 0
        print("  please type y or n")


def bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "underweight"
    if bmi < 25:
        return "normal"
    if bmi < 30:
        return "overweight"
    return "obese"


def main() -> None:
    if not (config.MODELS / "individual_model.joblib").exists():
        print("Training the model for the first time...")
        import synthetic_data
        synthetic_data.generate_cohort().to_csv(
            config.DATA_PROCESSED / "synthetic_individuals.csv", index=False)
        individual_model.train()

    print("\n" + "=" * 60)
    print(" UK Cancer Risk Calculator  (synthetic-data demo - NOT advice)")
    print("=" * 60)

    age = ask_int("Age", 18, 90, 50)
    sex = ask_choice("Sex", ["female", "male"], "female")
    ethnicity = ask_choice("Ethnicity", ["White", "Asian", "Black", "Mixed", "Other"], "White")
    print(f"  ({DEPRIVATION_HELP})")
    imd = ask_int("Area deprivation quintile", 1, 5, 3)
    smoking = ask_choice("Smoking status", ["never", "former", "current"], "never")
    height = ask_int("Height (cm)", 120, 220, 170)
    weight = ask_int("Weight (kg)", 35, 250, 75)
    bmi = weight / (height / 100) ** 2
    print(f"  -> BMI {bmi:.1f} ({bmi_category(bmi)})")
    alcohol = ask_choice("Alcohol intake", ["low", "moderate", "heavy"], "low")
    diabetes = ask_yesno("Diagnosed with diabetes?")
    inactive = ask_yesno("Physically inactive (under ~150 min/week)?")
    family = ask_yesno("Family history of cancer?")

    person = {
        "age": age, "sex": sex, "ethnicity": ethnicity, "imd_quintile": imd,
        "smoking": smoking, "bmi_cat": bmi_category(bmi), "diabetes": diabetes,
        "alcohol": alcohol, "physically_inactive": inactive, "family_history": family,
    }

    result = individual_model.predict_risk(person)
    drivers = individual_model.explain_risk(person)

    print("\n" + "-" * 60)
    print(f"  Estimated 10-year cancer risk:  {result['risk_pct']}   [{result['band']} risk]")
    print("-" * 60)
    if drivers:
        print("  What's driving this (vs a typical 50-year-old):")
        for d in drivers:
            direction = "increases" if d["multiplier"] > 1 else "reduces  "
            print(f"    {d['factor']:30s}  x{d['multiplier']:.2f}  ({direction} risk)")
    print("\n  Reminder: synthetic-data demonstration, not a diagnosis. "
          "Speak to a GP for real concerns.\n")


if __name__ == "__main__":
    main()
