"""Check test prediction coverage and key alignment against the baselines."""
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"


def main() -> None:
    predictions = pd.read_csv(PROC / "gru_test_preds.csv")
    baseline = pd.read_parquet(PROC / "baseline_preds.parquet")
    baseline = baseline.loc[baseline["split"].eq("test"), ["match_id", "point_idx", "progress"]]
    keys = ["match_id", "point_idx"]

    if predictions.duplicated(keys).any():
        raise SystemExit("GRU file has duplicate (match_id, point_idx) rows.")
    if baseline.duplicated(keys).any():
        raise SystemExit("Baseline file has duplicate test (match_id, point_idx) rows.")
    if predictions["gru"].isna().any() or not predictions["gru"].between(0, 1).all():
        raise SystemExit("GRU probabilities must be finite values between 0 and 1.")

    expected = pd.MultiIndex.from_frame(baseline[keys])
    actual = pd.MultiIndex.from_frame(predictions[keys])
    missing = expected.difference(actual)
    extra = actual.difference(expected)
    print(f"Baseline test points: {len(expected)} across {baseline['match_id'].nunique()} matches")
    print(f"GRU prediction points: {len(actual)} across {predictions['match_id'].nunique()} matches")
    print(f"Missing keys: {len(missing)}; extra keys: {len(extra)}")
    if len(missing) or len(extra):
        print("Sample missing keys:", list(missing[:10]))
        print("Sample extra keys:", list(extra[:10]))
        raise SystemExit("GRU prediction coverage does not match the baseline test set.")

    aligned = baseline.merge(predictions, on=keys, how="inner", validate="one_to_one")
    bins = [i / 10 for i in range(10)] + [math.nextafter(1.0, math.inf)]
    aligned["progress_bin"] = pd.cut(
        aligned["progress"], bins=bins, include_lowest=True, right=False,
        labels=[f"{i * 10}-{(i + 1) * 10}%" for i in range(10)],
    )
    print("Prediction coverage by match progress:")
    print(aligned.groupby("progress_bin", observed=False).size().to_string())
    print("All prediction keys align exactly with the test baseline.")


if __name__ == "__main__":
    main()
