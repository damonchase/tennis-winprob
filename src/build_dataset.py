"""Build the men's point-level dataset from the Grand Slam files (roadmap step 1.4).

What it does
  * keeps men's singles only (match ids 1xxx, or 'MS...' in some files)
  * skips files that lack set/game counters (the 2018-2021 Australian/French Open files)
  * drops the placeholder row (PointNumber 0) at the start of every match
  * derives each match's winner from the set winners and drops unfinished matches
  * turns the raw "state AFTER the point" columns into the state BEFORE each point

Raw columns hold the state AFTER a point, so for point t we use point t-1's state plus the
server of point t. A point's own result is never used as one of its own inputs.

Outputs (data/processed/): points_men.parquet (or .csv.gz if pyarrow is missing) and matches_men.csv

Usage (from the repo root):
    uv run src/build_dataset.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "raw" / "slam"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "data" / "processed"

SETS_TO_WIN = 3  # men's slam matches are best-of-five
REQUIRED = ["match_id", "PointNumber", "SetNo", "GameNo", "PointServer", "PointWinner",
            "P1Score", "P2Score", "P1GamesWon", "P2GamesWon", "SetWinner"]
NUMERIC = ["PointNumber", "SetNo", "GameNo", "PointServer", "PointWinner",
           "P1GamesWon", "P2GamesWon", "SetWinner"]
OUTCOME_COLS = ["P1Ace", "P2Ace", "P1DoubleFault", "P2DoubleFault",
                "P1Winner", "P2Winner", "P1UnfErr", "P2UnfErr"]  # results of the CURRENT point
NORMAL = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4}


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, encoding_errors="replace")


def is_mens(match_id) -> bool:
    tail = str(match_id).split("-")[-1].upper()
    if tail.startswith("MS"):
        return True
    return tail.isdigit() and int(tail) // 1000 == 1


def score_text(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    return out.replace("nan", "0")


def tiebreak_like(s: pd.Series) -> pd.Series:
    """True for plain numbers that normal game scores never use (1, 2, 3, 5, 6, 7, ...)."""
    return ~s.isin(list(NORMAL)) & s.str.fullmatch(r"\d+")


def to_points(raw: pd.Series, in_tiebreak: pd.Series) -> pd.Series:
    normal = raw.map(NORMAL)
    numeric = pd.to_numeric(raw, errors="coerce")
    return pd.Series(np.where(in_tiebreak, numeric, normal), index=raw.index)


def process_file(key: str):
    info = {"file": key, "status": "ok", "mens_matches": 0, "invalid": 0,
            "unfinished": 0, "unparsed": 0, "kept": 0}
    pts = read(RAW / f"{key}-points.csv")

    missing = [c for c in REQUIRED if c not in pts.columns]
    if missing:
        info["status"] = f"skipped (missing: {', '.join(missing)})"
        return None, info
    weak = [c for c in ("SetWinner", "P1GamesWon", "P2GamesWon") if pts[c].notna().mean() < 0.95]
    if weak:
        info["status"] = f"skipped (mostly empty: {', '.join(weak)})"
        return None, info

    for col in NUMERIC:
        pts[col] = pd.to_numeric(pts[col], errors="coerce")
    pts = pts.dropna(subset=["match_id", "PointNumber"])
    pts = pts[pts["match_id"].map(is_mens)]
    pts = pts[pts["PointNumber"] > 0].copy()          # drop the placeholder row
    info["mens_matches"] = pts["match_id"].nunique()
    pts["SetWinner"] = pts["SetWinner"].fillna(0)

    # matches with invalid server / winner / counters are dropped whole
    bad_row = (pts[["PointServer", "PointWinner", "SetNo", "GameNo", "P1GamesWon", "P2GamesWon"]].isna().any(axis=1)
               | ~pts["PointServer"].isin([1, 2]) | ~pts["PointWinner"].isin([1, 2]))
    bad = set(pts.loc[bad_row, "match_id"])
    info["invalid"] = len(bad)
    pts = pts[~pts["match_id"].isin(bad)].sort_values(["match_id", "PointNumber"]).reset_index(drop=True)

    # winner = player with more set wins; a match is finished once someone has won 3 sets
    pts["_s1"] = (pts["SetWinner"] == 1).astype(int)
    pts["_s2"] = (pts["SetWinner"] == 2).astype(int)
    totals = pts.groupby("match_id", sort=False)[["_s1", "_s2"]].sum()
    finished = totals.max(axis=1) >= SETS_TO_WIN
    info["unfinished"] = int((~finished).sum())
    p1_wins = (totals["_s1"] > totals["_s2"]).astype(int)
    pts = pts[pts["match_id"].isin(totals.index[finished])].reset_index(drop=True)
    pts["p1_wins"] = pts["match_id"].map(p1_wins)

    # ---- state BEFORE each point = state AFTER the previous point ----
    g = pts.groupby("match_id", sort=False)
    pts["p1_sets"] = g["_s1"].cumsum() - pts["_s1"]
    pts["p2_sets"] = g["_s2"].cumsum() - pts["_s2"]
    new_set = pts["SetNo"] != g["SetNo"].shift(1)                   # also True on a match's first point
    new_game = new_set | (pts["GameNo"] != g["GameNo"].shift(1))
    pts["p1_games"] = g["P1GamesWon"].shift(1).fillna(0).where(~new_set, 0)
    pts["p2_games"] = g["P2GamesWon"].shift(1).fillna(0).where(~new_set, 0)

    raw1, raw2 = score_text(pts["P1Score"]), score_text(pts["P2Score"])
    pre1 = raw1.groupby(pts["match_id"], sort=False).shift(1).fillna("0").where(~new_game, "0")
    pre2 = raw2.groupby(pts["match_id"], sort=False).shift(1).fillna("0").where(~new_game, "0")

    # a set has a tiebreak if any score in it looks like a tiebreak count
    set_has_tb = (tiebreak_like(raw1) | tiebreak_like(raw2)).groupby(
        [pts["match_id"], pts["SetNo"]]).transform("any")
    in_tb = set_has_tb & (pts["p1_games"] == 6) & (pts["p2_games"] == 6)
    pts["p1_pts"] = to_points(pre1, in_tb)
    pts["p2_pts"] = to_points(pre2, in_tb)

    unparsed = set(pts.loc[pts["p1_pts"].isna() | pts["p2_pts"].isna(), "match_id"])
    info["unparsed"] = len(unparsed)
    pts = pts[~pts["match_id"].isin(unparsed)].copy()
    in_tb = in_tb.loc[pts.index]

    out = pd.DataFrame({
        "match_id": pts["match_id"],
        "year": int(key[:4]),
        "slam": key.split("-")[1],
        "point_idx": pts.groupby("match_id", sort=False).cumcount(),
        "point_number": pts["PointNumber"].astype(int),
        "set_no": pts["SetNo"].astype(int),
        "game_no": pts["GameNo"].astype(int),
        "p1_sets": pts["p1_sets"].astype(int), "p2_sets": pts["p2_sets"].astype(int),
        "p1_games": pts["p1_games"].astype(int), "p2_games": pts["p2_games"].astype(int),
        "p1_pts": pts["p1_pts"].astype(int), "p2_pts": pts["p2_pts"].astype(int),
        "server": pts["PointServer"].astype(int),
        "is_tiebreak": in_tb.astype(int),
        "point_winner": pts["PointWinner"].astype(int),
        "p1_wins": pts["p1_wins"].astype(int),
    })
    for col in OUTCOME_COLS:   # results of the current point; shift before using them as features
        out[col] = pd.to_numeric(pts[col], errors="coerce") if col in pts else np.nan

    names = read(RAW / f"{key}-matches.csv")
    keep_cols = [c for c in ("match_id", "player1", "player2", "player1id", "player2id") if c in names.columns]
    info["kept"] = out["match_id"].nunique()
    return (out, names[keep_cols]), info


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames, name_frames, infos = [], [], []
    for path in sorted(RAW.glob("*-points.csv")):
        key = path.name.replace("-points.csv", "")
        result, info = process_file(key)
        infos.append(info)
        if result is not None:
            frames.append(result[0])
            name_frames.append(result[1])

    report = pd.DataFrame(infos)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 300)
    print("=== Per-file report (men's matches only) ===")
    print(report.to_string(index=False))

    points = pd.concat(frames, ignore_index=True)
    points["split"] = np.where(points["year"] <= 2021, "train", np.where(points["year"] == 2022, "val", "test"))

    matches = points.groupby("match_id").agg(year=("year", "first"), slam=("slam", "first"),
                                             n_points=("point_idx", "size"), p1_wins=("p1_wins", "first"),
                                             split=("split", "first")).reset_index()
    names = pd.concat(name_frames).drop_duplicates("match_id")
    matches = matches.merge(names, on="match_id", how="left")

    print("\n=== Final dataset ===")
    print(f"matches: {len(matches)}   points: {len(points)}")
    print(matches.groupby("split").agg(matches=("match_id", "size"), points=("n_points", "sum")).to_string())
    print("\nP(player 1 wins), should be near 0.5 before we balance it:", round(matches["p1_wins"].mean(), 3))
    print("matches by slam:", matches["slam"].value_counts().to_dict())

    non_tb = points[points["is_tiebreak"] == 0]
    one_server = (non_tb.groupby(["match_id", "set_no", "game_no"])["server"].nunique() == 1).mean()
    print(f"\nSanity check: games (outside tiebreaks) with exactly one server: {one_server:.1%}  (expect ~100%)")

    cols = ["point_idx", "set_no", "game_no", "p1_sets", "p2_sets", "p1_games", "p2_games",
            "p1_pts", "p2_pts", "server", "is_tiebreak", "point_winner"]
    first = points["match_id"].iloc[0]
    sample = points[points["match_id"] == first][cols]
    print(f"\n=== Sample match {first}: first 14 rows (state BEFORE each point) ===")
    print(sample.head(14).to_string(index=False))
    print("... last 3 rows")
    print(sample.tail(3).to_string(index=False))

    matches.to_csv(OUT / "matches_men.csv", index=False)
    try:
        points.to_parquet(OUT / "points_men.parquet", index=False)
        print(f"\nSaved {OUT / 'points_men.parquet'} and matches_men.csv")
    except ImportError:
        points.to_csv(OUT / "points_men.csv.gz", index=False)
        print(f"\npyarrow not installed (run: uv add pyarrow); saved points_men.csv.gz instead")


if __name__ == "__main__":
    main()