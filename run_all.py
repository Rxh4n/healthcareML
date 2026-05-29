"""
Run the whole project end to end:

    python run_all.py            # use cached data if present
    python run_all.py --force    # re-download the real UK area data first

Produces models/ (saved models) and outputs/ (figures, metrics, predictions).
"""
from __future__ import annotations

import sys

import area_model
import individual_model
import synthetic_data


def main() -> None:
    force = "--force" in sys.argv

    print("=" * 70)
    print("1/2  AREA-LEVEL MODEL  (real UK open data, OHID Fingertips)")
    print("=" * 70)
    area_model.main(force_fetch=force)

    print("\n" + "=" * 70)
    print("2/2  INDIVIDUAL-LEVEL MODEL  (synthetic data, real published risks)")
    print("=" * 70)
    synthetic_data.generate_cohort().to_csv(
        area_model.config.DATA_PROCESSED / "synthetic_individuals.csv", index=False
    )
    individual_model.train()

    print("\nDone. See outputs/ for figures, metrics and predictions.")


if __name__ == "__main__":
    main()
