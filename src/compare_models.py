"""Compare the GRU with the baselines on the TEST years and find out where it helps.

Reads
  * data/processed/baseline_preds.(parquet|csv.gz)   written by src/baselines.py (elo_only, markov)
  * your GRU test predictions: a CSV/parquet with columns  match_id, point_idx, gru
        gru = probability that player 1 wins, i.e. the SAME numbers you used for your reported metrics
        (after temperature scaling), one row per test point, with padding rows removed
  * data/processed/points_men.* and matches_men_elo.csv   (set number, tiebreak flag, slam)

Prints (and saves to reports/):
  1. ROC-AUC, Brier, log loss, calibration error for every model, with 95% bootstrap intervals
  2. Paired bootstrap differences: GRU minus Markov, and GRU minus Elo-only
  3. Where the GRU helps: the same comparison sliced by match progress, set, tiebreak, Elo-favorite strength, slam

All intervals come from resampling MATCHES (points inside a match are strongly correlated).

Usage (from the repo root):
    uv run src/compare_models.py data/processed/gru_test_preds.csv
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import brier, ece, log_loss, point_losses  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "data" / "processed"
REPORTS = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "reports"
FIG = REPORTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)
MODELS = ["gru", "markov", "elo_only"]
N_BOOT, N_BOOT_AUC, SEED = 1000, 300, 0


def read_table(path: Path) -> pd.DataFrame:
    name = str(path)
    if name.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def find(stem: str) -> Path:
    for ext in (".parquet", ".csv.gz", ".csv"):
        if (PROC / f"{stem}{ext}").exists():
            return PROC / f"{stem}{ext}"
    raise SystemExit(f"Could not find {stem}.* in {PROC}. Run the earlier steps first.")


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """ROC-AUC via ranks (same value as sklearn's roc_auc_score)."""
    ranks = pd.Series(p).rank(method="average").to_numpy()
    n_pos = float(y.sum())
    n_neg = len(y) - n_pos
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def verdict(lo: float, hi: float, lower_is_better: bool = True) -> str:
    """Interpret the interval of (GRU minus other)."""
    if lo > 0:
        return "GRU worse" if lower_is_better else "GRU better"
    if hi < 0:
        return "GRU better" if lower_is_better else "GRU worse"
    return "no clear difference"


def load_merged(gru_path: Path) -> pd.DataFrame:
    gru = read_table(gru_path)
    missing = {"match_id", "point_idx", "gru"} - set(gru.columns)
    if missing:
        raise SystemExit(f"{gru_path} must have columns match_id, point_idx, gru. Missing: {sorted(missing)}. Found: {gru.columns.tolist()}")
    base = read_table(find("baseline_preds"))
    base = base[base["split"] == "test"]
    df = base.merge(gru[["match_id", "point_idx", "gru"]], on=["match_id", "point_idx"], how="inner")
    print(f"Test points in baseline file: {len(base)}   in GRU file: {len(gru)}   matched: {len(df)}")
    if len(df) < 0.99 * min(len(base), len(gru)):
        print("WARNING: fewer than 99% of points matched. Check that match_id and point_idx mean the same thing in both files.")
    points = read_table(find("points_men"))[["match_id", "point_idx", "set_no", "is_tiebreak"]]
    matches = pd.read_csv(PROC / "matches_men_elo.csv")[["match_id", "slam"]]
    df = df.merge(points, on=["match_id", "point_idx"], how="left").merge(matches, on="match_id", how="left")
    return df.sort_values(["match_id", "point_idx"]).reset_index(drop=True)


class Bootstrap:
    """Resample matches with replacement; the same resamples are used for every model (paired)."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.codes, uniq = pd.factorize(df["match_id"])
        self.m = len(uniq)
        rng = np.random.default_rng(SEED)
        self.draws = rng.integers(0, self.m, size=(N_BOOT, self.m))
        y = df["p1_wins"].to_numpy()
        self.loss = {"brier": {c: (df[c].to_numpy() - y) ** 2 for c in MODELS},
                     "log_loss": {c: point_losses(y, df[c].to_numpy()) for c in MODELS}}

    def sums(self, kind: str, model: str, cat: pd.Categorical):
        """Per-match sums and counts for each slice: arrays of shape (matches, slices)."""
        k = len(cat.categories)
        keep = cat.codes >= 0
        idx = self.codes[keep] * k + cat.codes[keep]
        n = np.bincount(idx, minlength=self.m * k).reshape(self.m, k).astype(float)
        s = np.bincount(idx, weights=self.loss[kind][model][keep], minlength=self.m * k).reshape(self.m, k)
        return s, n

    def table(self, kind: str, cat: pd.Categorical, a: str = "gru", b: str = "markov") -> pd.DataFrame:
        sa, n = self.sums(kind, a, cat)
        sb, _ = self.sums(kind, b, cat)
        with np.errstate(divide="ignore", invalid="ignore"):   # a slice can be empty in some resamples
            diff = (sa[self.draws].sum(axis=1) - sb[self.draws].sum(axis=1)) / n[self.draws].sum(axis=1)
        lo, hi = np.nanpercentile(diff, [2.5, 97.5], axis=0)
        rows = []
        for j, name in enumerate(cat.categories):
            if n[:, j].sum() == 0:
                continue
            va, vb = sa[:, j].sum() / n[:, j].sum(), sb[:, j].sum() / n[:, j].sum()
            rows.append({"slice": name, "points": int(n[:, j].sum()), a: round(va, 4), b: round(vb, 4),
                         f"{a}-{b}": round(va - vb, 4), "95% CI": f"[{lo[j]:+.4f}, {hi[j]:+.4f}]",
                         "verdict": verdict(lo[j], hi[j])})
        return pd.DataFrame(rows)

    def auc_draws(self, model: str) -> np.ndarray:
        y, p = self.df["p1_wins"].to_numpy(), self.df[model].to_numpy()
        counts = np.bincount(self.codes, minlength=self.m)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        pieces = [np.arange(s, s + c) for s, c in zip(starts, counts)]   # rows are sorted by match
        out = np.empty(N_BOOT_AUC)
        for i in range(N_BOOT_AUC):
            idx = np.concatenate([pieces[j] for j in self.draws[i]])
            out[i] = auc(y[idx], p[idx])
        return out


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: uv run src/compare_models.py path/to/gru_test_preds.csv")
    df = load_merged(Path(sys.argv[1]))
    y = df["p1_wins"].to_numpy()
    print(f"\n=== TEST: {df['match_id'].nunique()} matches, {len(df)} points ===")

    boot = Bootstrap(df)
    whole = pd.Categorical(["all"] * len(df), categories=["all"])
    print(f"Computing bootstrap intervals ({N_BOOT} resamples; ROC-AUC uses {N_BOOT_AUC}; this takes a minute) ...")
    auc_boots = {m: boot.auc_draws(m) for m in MODELS}

    rows = {}
    for m in MODELS:
        p = df[m].to_numpy()
        ci = {}
        for kind in ("brier", "log_loss"):
            s, n = boot.sums(kind, m, whole)
            v = (s[boot.draws].sum(axis=1) / n[boot.draws].sum(axis=1))[:, 0]
            ci[kind] = np.percentile(v, [2.5, 97.5])
        rows[m] = {"roc_auc": round(auc(y, p), 4), "roc_auc_95%CI": f"[{np.percentile(auc_boots[m], 2.5):.4f}, {np.percentile(auc_boots[m], 97.5):.4f}]",
                   "brier": round(brier(y, p), 4), "brier_95%CI": f"[{ci['brier'][0]:.4f}, {ci['brier'][1]:.4f}]",
                   "log_loss": round(log_loss(y, p), 4), "log_loss_95%CI": f"[{ci['log_loss'][0]:.4f}, {ci['log_loss'][1]:.4f}]",
                   "ece": round(ece(y, p), 4)}
    overall = pd.DataFrame(rows).T
    print("\n--- 1. Overall test metrics (use these in your resume; the GRU row should match your reported numbers) ---")
    print(overall.to_string())

    print("\n--- 2. Paired bootstrap differences (GRU minus other). Brier and log loss: negative = GRU better. AUC: positive = GRU better ---")
    diffs = []
    for other in ("markov", "elo_only"):
        for kind in ("brier", "log_loss"):
            t = boot.table(kind, whole, "gru", other).iloc[0]
            diffs.append({"comparison": f"gru - {other}", "metric": kind, "difference": t[f"gru-{other}"], "95% CI": t["95% CI"], "verdict": t["verdict"]})
        d = auc_boots["gru"] - auc_boots[other]
        lo, hi = np.percentile(d, [2.5, 97.5])
        diffs.append({"comparison": f"gru - {other}", "metric": "roc_auc", "difference": round(auc(y, df['gru'].to_numpy()) - auc(y, df[other].to_numpy()), 4),
                      "95% CI": f"[{lo:+.4f}, {hi:+.4f}]", "verdict": verdict(lo, hi, lower_is_better=False)})
    diffs = pd.DataFrame(diffs)
    print(diffs.to_string(index=False))

    # ---- slices ----
    progress = pd.Categorical(pd.cut(df["progress"], np.linspace(0, 1, 11), include_lowest=True, right=False,
                                     labels=[f"{i * 10}-{i * 10 + 10}%" for i in range(10)]))
    fav = np.maximum(df["elo_only"], 1 - df["elo_only"])
    favorite = pd.Categorical(pd.cut(fav, [0.5, 0.6, 0.7, 0.8, 1.0001], right=False,
                                     labels=["toss-up (50-60%)", "moderate (60-70%)", "strong (70-80%)", "heavy (80%+)"]))
    set_cat = pd.Categorical(df["set_no"].map(lambda v: f"set {int(v)}" if pd.notna(v) else None),
                             categories=[f"set {i}" for i in range(1, 6)])
    tb_cat = pd.Categorical(df["is_tiebreak"].map({0: "normal game", 1: "tiebreak"}), categories=["normal game", "tiebreak"])
    slam_cat = pd.Categorical(df["slam"], categories=sorted(df["slam"].dropna().unique()))
    slices = {"match progress": progress, "set": set_cat, "game type": tb_cat,
              "pre-match Elo favorite strength": favorite, "tournament": slam_cat}

    print("\n--- 3. Where does the GRU help? Brier score by slice, GRU vs Markov (negative difference = GRU better) ---")
    saved = []
    for name, cat in slices.items():
        t = boot.table("brier", cat, "gru", "markov")
        print(f"\n[{name}]")
        print(t.to_string(index=False))
        t.insert(0, "dimension", name)
        saved.append(t)
    pd.concat(saved).to_csv(REPORTS / "where_gru_helps.csv", index=False)
    pd.concat([overall.assign(table="overall").reset_index(names="model")]).to_csv(REPORTS / "model_comparison.csv", index=False)
    diffs.to_csv(REPORTS / "model_differences.csv", index=False)

    prog = boot.table("brier", progress, "gru", "markov")
    prog_elo = boot.table("brier", progress, "gru", "elo_only")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = range(len(prog))
    ax.plot(x, prog["gru"], marker="o", label="GRU")
    ax.plot(x, prog["markov"], marker="o", label="Markov chain")
    ax.plot(x, prog_elo["elo_only"], marker="o", label="Elo only")
    ax.set_xticks(list(x)); ax.set_xticklabels(prog["slice"], rotation=45)
    ax.set_xlabel("How far into the match"); ax.set_ylabel("Brier score (lower is better)")
    ax.set_title("Test years: where the GRU helps"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "09_where_gru_helps.png", dpi=150); plt.close(fig)
    print(f"\nSaved tables to {REPORTS} and the figure to {FIG / '09_where_gru_helps.png'}")


if __name__ == "__main__":
    main()