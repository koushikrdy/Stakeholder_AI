"""
One-off preprocessing: convert the raw cardio_train.csv (Kaggle "Cardiovascular
Disease dataset") into the column names and units run_experiment.py expects.

Usage:
    python scripts/prepare_cardio_train.py cardio_train.csv cohort_ready.csv
"""
import sys
import pandas as pd

def main(src_path: str, dst_path: str) -> None:
    raw = pd.read_csv(src_path, sep=";" if src_path.endswith(".csv") else ",")
    # cardio_train.csv is semicolon-separated on Kaggle; fall back to comma if needed.
    if raw.shape[1] == 1:
        raw = pd.read_csv(src_path, sep=",")

    out = pd.DataFrame()
    out["age"] = raw["age"] / 365.25  # days -> years
    # Confirm this against your data dictionary before trusting it:
    # cardio_train convention is 1 = women, 2 = men.
    out["male"] = (raw["gender"] == 2).astype(int)
    out["cholesterol_level"] = raw["cholesterol"].astype(float)
    out["systolic_bp"] = raw["ap_hi"].astype(float)
    out["glucose_level"] = raw["gluc"].astype(float)
    out["bmi"] = raw["weight"] / (raw["height"] / 100.0) ** 2
    out["smoker"] = raw["smoke"].astype(int)
    out["alcohol"] = raw["alco"].astype(int)
    out["active"] = raw["active"].astype(int)
    out["_event"] = raw["cardio"].astype(int)

    # Basic sanity filtering: the public dataset has some biologically
    # implausible ap_hi/ap_lo/height/weight entries worth dropping.
    before = len(out)
    out = out[(out["systolic_bp"].between(80, 220)) & (out["bmi"].between(14, 55))]
    out = out[(out["age"].between(25, 70))]
    print(f"kept {len(out)}/{before} rows after sanity filtering")

    out.to_csv(dst_path, index=False)
    print(f"wrote {dst_path}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
