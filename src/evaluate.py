"""Metrics shared by every model, so all models are scored the same way.

Predictions live in a table with one row per point and these columns:
    match_id, p1_wins (0/1 label), progress (fraction of the match played, 0-1), and one column per model.
"""
import numpy as np
import pandas as pd

EPS = 1e-6


def point_losses(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Log loss of each prediction: confident wrong answers are punished heavily."""
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def log_loss(y, p) -> float:
    return float(point_losses(np.asarray(y), np.asarray(p)).mean())


def brier(y, p) -> float:
    return float(((np.asarray(p) - np.asarray(y)) ** 2).mean())


def ece(y, p, n_bins: int = 10) -> float:
    """Expected calibration error: average gap between predicted probability and actual win rate."""
    y, p = np.asarray(y), np.asarray(p)
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            total += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(total)


def metrics_table(df: pd.DataFrame, cols) -> pd.DataFrame:
    y = df["p1_wins"].to_numpy()
    rows = {c: {"log_loss": log_loss(y, df[c]), "brier": brier(y, df[c]), "ece": ece(y, df[c])} for c in cols}
    return pd.DataFrame(rows).T.round(4)


def by_progress(df: pd.DataFrame, cols, n_bins: int = 10) -> pd.DataFrame:
    """Log loss per model, by how far into the match the point is (0-10%, 10-20%, ...)."""
    y = df["p1_wins"].to_numpy()
    bins = np.minimum((df["progress"].to_numpy() * n_bins).astype(int), n_bins - 1)
    out = {}
    for c in cols:
        losses = point_losses(y, df[c].to_numpy())
        out[c] = pd.Series(losses).groupby(bins).mean()
    table = pd.DataFrame(out)
    table.index = [f"{int(i * 100 / n_bins)}-{int((i + 1) * 100 / n_bins)}%" for i in table.index]
    return table.round(4)


def match_bootstrap(df: pd.DataFrame, cols, n_boot: int = 1000, seed: int = 0):
    """Bootstrap over MATCHES (points inside a match are strongly correlated).

    Returns (estimates, ci, boots): boots[i, j] is model j's log loss in resample i,
    so a paired difference is just boots[:, j] - boots[:, k].
    """
    y = df["p1_wins"].to_numpy()
    per_point = pd.DataFrame({"match_id": df["match_id"].to_numpy()})
    for c in cols:
        per_point[c] = point_losses(y, df[c].to_numpy())
    grouped = per_point.groupby("match_id")
    sums, counts = grouped[list(cols)].sum().to_numpy(), grouped.size().to_numpy()
    rng = np.random.default_rng(seed)
    m = len(counts)
    boots = np.empty((n_boot, len(cols)))
    for i in range(n_boot):
        idx = rng.integers(0, m, m)
        boots[i] = sums[idx].sum(axis=0) / counts[idx].sum()
    estimates = sums.sum(axis=0) / counts.sum()
    ci = np.percentile(boots, [2.5, 97.5], axis=0)
    return estimates, ci, boots