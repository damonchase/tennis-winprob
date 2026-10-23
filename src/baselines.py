"""Score the two baselines (Elo-only and Markov chain) on the validation and test years.

  * elo_only : every point of a match gets the same pre-match Elo probability
  * markov   : the Markov-chain model, calibrated so its pre-match probability equals the Elo probability,
               then updated after every point from the score

Serve probabilities: base = the average share of points the server wins (per slam, TRAINING years only);
player 1 / player 2 get base +/- d, with d chosen so the model's pre-match win probability matches Elo.

Outputs: data/processed/baseline_preds.parquet (or .csv.gz), reports/baseline_results.csv, 2 figures.

Usage (from the repo root):
    uv run src/baselines.py
"""
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import by_progress, match_bootstrap, metrics_table  # noqa: E402
from markov import MarkovMatch, solve_serve_probs  # noqa: E402
from states import invalid_state_mask  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "processed"
REPORTS = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "reports"
FIG = REPORTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)
STATE_COLS = ["p1_sets", "p2_sets", "p1_games", "p2_games", "p1_pts", "p2_pts", "server", "is_tiebreak"]
MODELS = ["elo_only", "markov"]
_solved = {}


def load_points() -> pd.DataFrame:
    parquet = PROC / "points_men.parquet"
    return pd.read_parquet(parquet) if parquet.exists() else pd.read_csv(PROC / "points_men.csv.gz")


def serve_probs(p_elo: float, base: float):
    key = (round(p_elo, 3), round(base, 4))
    if key not in _solved:
        _solved[key] = solve_serve_probs(key[0], key[1])
    return _solved[key]


def predict_split(points: pd.DataFrame, matches: pd.DataFrame, base_by_slam: dict) -> pd.DataFrame:
    parts = []
    info = matches.set_index("match_id")
    for match_id, rows in points.groupby("match_id", sort=False):
        p_elo = float(info.loc[match_id, "p_elo_blend"])
        slam = info.loc[match_id, "slam"]
        p1s, p2s = serve_probs(p_elo, base_by_slam.get(slam, base_by_slam["_overall"]))
        model = MarkovMatch(p1s, p2s)
        states = zip(*(rows[c].to_numpy() for c in STATE_COLS))
        p_markov = [model.win_prob(int(s1), int(s2), int(g1), int(g2), int(a), int(b), int(srv), bool(tb))
                    for s1, s2, g1, g2, a, b, srv, tb in states]
        out = rows[["match_id", "split", "point_idx", "p1_wins"]].copy()
        out["progress"] = out["point_idx"] / len(rows)
        out["elo_only"] = p_elo
        out["markov"] = p_markov
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    matches = pd.read_csv(PROC / "matches_men_elo.csv")
    points = load_points()
    points = points[points["match_id"].isin(matches["match_id"])]
    bad = set(points.loc[invalid_state_mask(points), "match_id"])
    if bad:
        print(f"WARNING: {len(bad)} matches contain impossible states and are skipped. "
              "Re-run src/build_dataset.py (and src/elo.py) to remove them properly.")
        points = points[~points["match_id"].isin(bad)]

    train = points[points["split"] == "train"]
    base_by_slam = (train["point_winner"] == train["server"]).groupby(train["slam"]).mean().to_dict()
    base_by_slam["_overall"] = float((train["point_winner"] == train["server"]).mean())   # fallback for unseen slams
    print("Average share of points won by the server, training years:",
          {k: round(v, 3) for k, v in base_by_slam.items()})

    start = time.time()
    preds = []
    for split in ("val", "test"):
        part = points[points["split"] == split]
        print(f"Predicting {split}: {part['match_id'].nunique()} matches, {len(part)} points ...")
        preds.append(predict_split(part, matches[matches["split"] == split], base_by_slam))
    preds = pd.concat(preds, ignore_index=True)
    print(f"Done in {time.time() - start:.0f}s")

    try:
        preds.to_parquet(PROC / "baseline_preds.parquet", index=False)
    except ImportError:
        preds.to_csv(PROC / "baseline_preds.csv.gz", index=False)

    results = []
    for split in ("val", "test"):
        df = preds[preds["split"] == split]
        print(f"\n================ {split.upper()}: {df['match_id'].nunique()} matches, {len(df)} points ================")
        table = metrics_table(df, MODELS)
        est, ci, boots = match_bootstrap(df, MODELS)
        table["log_loss_95%_CI"] = [f"[{ci[0, i]:.4f}, {ci[1, i]:.4f}]" for i in range(len(MODELS))]
        print(table.to_string())
        diff = boots[:, 1] - boots[:, 0]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        verdict = "Markov is better (interval is below 0)" if hi < 0 else (
            "Markov is worse (interval is above 0)" if lo > 0 else "no clear difference (interval includes 0)")
        print(f"\nPaired bootstrap over matches, log loss (markov - elo_only): {diff.mean():+.4f}  "
              f"95% CI [{lo:+.4f}, {hi:+.4f}]  -> {verdict}")
        print("\nLog loss by how far into the match we are:")
        print(by_progress(df, MODELS).to_string())
        t = table.copy()
        t.insert(0, "split", split)
        results.append(t)
    pd.concat(results).to_csv(REPORTS / "baseline_results.csv")

    test = preds[preds["split"] == "test"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    prog = by_progress(test, MODELS)
    for c in MODELS:
        ax.plot(range(len(prog)), prog[c].values, marker="o", label=c)
    ax.set_xticks(range(len(prog))); ax.set_xticklabels(prog.index, rotation=45)
    ax.set_xlabel("How far into the match"); ax.set_ylabel("Log loss (lower is better)")
    ax.set_title("Test years: baselines by match progress"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "07_baselines_by_progress.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    bins = np.linspace(0, 1, 11)
    for c in MODELS:
        idx = np.minimum((test[c].to_numpy() * 10).astype(int), 9)
        grp = pd.DataFrame({"p": test[c].to_numpy(), "y": test["p1_wins"].to_numpy(), "b": idx}).groupby("b")
        ax.plot(grp["p"].mean(), grp["y"].mean(), marker="o", label=c)
    ax.set_xlabel("Predicted probability that player 1 wins"); ax.set_ylabel("Actual win rate")
    ax.set_title("Test years: calibration"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "08_baselines_calibration.png", dpi=150); plt.close(fig)
    print(f"\nSaved results to {REPORTS / 'baseline_results.csv'} and 2 figures to {FIG}")


if __name__ == "__main__":
    main()