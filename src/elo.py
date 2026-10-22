"""Build leakage-free Elo ratings from ATP results and attach them to the slam matches.

For every Grand Slam we freeze a snapshot of all ratings BEFORE the tournament starts
(so a match never influences its own rating, nor any other match in the same event),
then look up both players in the snapshot.

Two ratings are kept: an overall Elo and a surface-specific Elo.
Inputs : data/raw/atp/atp_matches_YYYY.csv and data/processed/matches_men.csv
Output : data/processed/matches_men_elo.csv

Usage (from the repo root):
    uv run src/elo.py
"""
import difflib
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ATP_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "raw" / "atp"
PROC = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "data" / "processed"

START = 1500.0
ATP_REQUIRED = ["tourney_id", "tourney_name", "tourney_date", "surface", "tourney_level",
                "match_num", "round", "winner_name", "loser_name", "score"]
ROUND_RANK = {"Q1": -3, "Q2": -2, "Q3": -1, "R128": 1, "R64": 2, "R32": 3, "R16": 4,
              "RR": 3, "QF": 5, "SF": 6, "BR": 6, "F": 7}
SLAM_SURFACE = {"ausopen": "Hard", "usopen": "Hard", "frenchopen": "Clay", "wimbledon": "Grass"}

# If a slam name has no ATP match, add it here: normalized slam name -> normalized ATP name.
NAME_FIXES = {
    "holger vitus nodskov rune": "holger rune",
}


def normalize(name) -> str:
    """'Jo-Wilfried Tsonga' / 'Rafael Nádal' -> 'jo wilfried tsonga' / 'rafael nadal'."""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    s = s.replace("'", "").replace("\u2019", "")      # O'Connell -> oconnell
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return NAME_FIXES.get(s, s)


def slam_key(tourney_name: str, level: str):
    if level != "G":
        return None
    n = str(tourney_name).lower()
    if "australian" in n:
        return "ausopen"
    if "roland" in n or "french" in n:
        return "frenchopen"
    if "wimbledon" in n:
        return "wimbledon"
    if "us open" in n or "u.s. open" in n:
        return "usopen"
    return None


def expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def k_factor(n_matches: int) -> float:
    return 250.0 / (n_matches + 5) ** 0.4   # bigger updates for players with few matches


def apply_result(table: dict, winner, loser) -> None:
    rw, nw = table.get(winner, [START, 0])
    rl, nl = table.get(loser, [START, 0])
    e = expected(rw, rl)
    table[winner] = [rw + k_factor(nw) * (1 - e), nw + 1]
    table[loser] = [rl - k_factor(nl) * (1 - e), nl + 1]


def load_atp() -> pd.DataFrame:
    files = sorted(ATP_DIR.glob("atp_matches_[0-9][0-9][0-9][0-9].csv"))
    if not files:
        raise SystemExit(f"No ATP files found in {ATP_DIR}. Run src/data.py first.")
    frames = [pd.read_csv(f, low_memory=False, encoding_errors="replace") for f in files]
    atp = pd.concat(frames, ignore_index=True)
    missing = [c for c in ATP_REQUIRED if c not in atp.columns]
    if missing:
        raise SystemExit(f"ATP files are missing columns {missing}. Columns found: {atp.columns.tolist()}")
    atp = atp.dropna(subset=["winner_name", "loser_name", "tourney_date"]).copy()
    score = atp["score"].astype(str).str.upper()
    atp = atp[~(score.str.contains("W/O") | score.str.contains("DEF"))]      # walkovers and defaults
    atp = atp[atp["tourney_level"] != "D"]                                    # Davis Cup
    atp["tourney_date"] = pd.to_numeric(atp["tourney_date"], errors="coerce")
    atp = atp.dropna(subset=["tourney_date"])
    atp["w"] = atp["winner_name"].map(normalize)
    atp["l"] = atp["loser_name"].map(normalize)
    atp["rank"] = atp["round"].map(ROUND_RANK).fillna(0)
    atp["match_num"] = pd.to_numeric(atp["match_num"], errors="coerce").fillna(0)
    return atp


def build_snapshots(atp: pd.DataFrame) -> dict:
    """Run through all results in date order; freeze ratings just before each slam starts."""
    overall, surface, snapshots, event_names = {}, {}, {}, {}
    for (date, _tid), t in atp.groupby(["tourney_date", "tourney_id"], sort=True):
        key = slam_key(t["tourney_name"].iloc[0], t["tourney_level"].iloc[0])
        if key is not None:
            snapshots[(int(date) // 10000, key)] = (
                {k: tuple(v) for k, v in overall.items()},
                {k: tuple(v) for k, v in surface.items()},
            )
            event_names[(int(date) // 10000, key)] = set(t["w"]) | set(t["l"])
        for row in t.sort_values(["rank", "match_num"]).itertuples():
            apply_result(overall, row.w, row.l)
            surf = row.surface if isinstance(row.surface, str) else "Unknown"
            apply_result(surface, (row.w, surf), (row.l, surf))
    return snapshots, event_names


def attach_elo(matches: pd.DataFrame, snapshots: dict, known_names: set):
    rows, unmatched, no_snapshot = [], [], 0
    for m in matches.itertuples():
        p1, p2 = normalize(m.player1), normalize(m.player2)
        if p1 not in known_names or p2 not in known_names:
            for raw, norm in ((m.player1, p1), (m.player2, p2)):
                if norm not in known_names:
                    unmatched.append((raw, norm, int(m.year), m.slam))
            continue
        snap = snapshots.get((int(m.year), m.slam))
        if snap is None:
            no_snapshot += 1
            continue
        overall, surface = snap
        surf = SLAM_SURFACE[m.slam]
        r1, n1 = overall.get(p1, (START, 0))
        r2, n2 = overall.get(p2, (START, 0))
        s1, _ = surface.get((p1, surf), (START, 0))
        s2, _ = surface.get((p2, surf), (START, 0))
        rows.append({"match_id": m.match_id, "elo1": r1, "elo2": r2, "selo1": s1, "selo2": s2,
                     "n1": n1, "n2": n2})
    return pd.DataFrame(rows), unmatched, no_snapshot


def suggest_fixes(mentions, event_names, claimed, min_ratio=0.75, margin=0.08):
    """For each unmatched slam name, look for a similar ATP name among players in the SAME
    tournament that no other slam name has claimed. Returns (auto_fixes, review_lines)."""
    per_name = {}
    for raw, norm, year, slam in mentions:
        pool = sorted(event_names.get((year, slam), set()) - claimed.get((year, slam), set()))
        scored = sorted(((difflib.SequenceMatcher(None, norm, c).ratio(), c) for c in pool), reverse=True)[:3]
        per_name.setdefault(norm, {"raw": raw, "count": 0, "options": []})
        per_name[norm]["count"] += 1
        per_name[norm]["options"].append(scored)
    auto, review = {}, []
    for norm, info in per_name.items():
        votes = Counter(o[0][1] for o in info["options"] if o)
        best_name = votes.most_common(1)[0][0] if votes else None
        shown = [(round(r, 2), c) for r, c in (info["options"][0] if info["options"] else [])]
        if best_name is not None:
            ratios = [o for o in info["options"] if o and o[0][1] == best_name][0]
            best_ratio = ratios[0][0]
            runner_up = ratios[1][0] if len(ratios) > 1 else 0.0
            if best_ratio >= min_ratio and best_ratio - runner_up >= margin:
                auto[norm] = (best_name, round(best_ratio, 2), info["count"])
                continue
        review.append((info["raw"], info["count"], shown))
    return auto, review


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"accuracy": round(float(((p > 0.5) == (y == 1)).mean()), 3),
            "log_loss": round(float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()), 4),
            "brier": round(float(((p - y) ** 2).mean()), 4)}


def main() -> None:
    matches = pd.read_csv(PROC / "matches_men.csv")
    atp = load_atp()
    print(f"ATP results used for Elo: {len(atp)} matches, {atp['tourney_date'].min():.0f} to {atp['tourney_date'].max():.0f}")
    known = set(atp["w"]) | set(atp["l"])
    snapshots, event_names = build_snapshots(atp)
    print(f"Slam snapshots built: {len(snapshots)}")

    elo, unmatched, no_snapshot = attach_elo(matches, snapshots, known)
    if unmatched:
        claimed = {}
        for m in matches.itertuples():
            ev = (int(m.year), m.slam)
            for nm in (normalize(m.player1), normalize(m.player2)):
                if nm in event_names.get(ev, ()):
                    claimed.setdefault(ev, set()).add(nm)
        auto, review = suggest_fixes(unmatched, event_names, claimed)
        if auto:
            print("\nAutomatic name fixes applied (check that these are the same player):")
            for norm, (target, ratio, count) in sorted(auto.items(), key=lambda kv: -kv[1][2]):
                print(f"  {norm!r} -> {target!r}   (similarity {ratio}, {count} matches)")
            NAME_FIXES.update({k: v[0] for k, v in auto.items()})
            elo, unmatched, no_snapshot = attach_elo(matches, snapshots, known)
        if review:
            print("\nStill unmatched. Closest unclaimed names in that tournament's draw are shown; if one is right,")
            print("add it to NAME_FIXES in src/elo.py as  'slam name (lowercase, no accents)': 'atp name',")
            for raw, count, shown in sorted(review, key=lambda r: -r[1]):
                print(f"  {raw!r}: {count} matches   closest: {shown}")
    out = matches.merge(elo, on="match_id", how="inner")
    out["logit_overall"] = (out["elo1"] - out["elo2"]) / 400.0 * math.log(10)
    out["logit_blend"] = ((out["elo1"] + out["selo1"]) / 2 - (out["elo2"] + out["selo2"]) / 2) / 400.0 * math.log(10)
    out["p_elo_overall"] = 1 / (1 + np.exp(-out["logit_overall"]))
    out["p_elo_blend"] = 1 / (1 + np.exp(-out["logit_blend"]))

    print(f"\nMatches in: {len(matches)}   with Elo: {len(out)}   dropped: {len(matches) - len(out)} "
          f"(unknown names: {len(unmatched)} mentions, missing snapshot: {no_snapshot})")

    print("\n=== Pre-match Elo only (every point gets the same probability) ===")
    for split in ["train", "val", "test"]:
        part = out[out["split"] == split]
        y = part["p1_wins"].to_numpy()
        print(f"{split:5s} n={len(part):4d}  overall: {metrics(y, part['p_elo_overall'].to_numpy())}")
        print(f"{'':5s}          blend:   {metrics(y, part['p_elo_blend'].to_numpy())}")
    print(f"\nFor reference, always predicting 50%: log_loss 0.6931, brier 0.25")
    print(f"Matches where a player had fewer than 10 prior tour matches: "
          f"{int(((out['n1'] < 10) | (out['n2'] < 10)).sum())}")

    out.to_csv(PROC / "matches_men_elo.csv", index=False)
    print(f"\nSaved {PROC / 'matches_men_elo.csv'}")


if __name__ == "__main__":
    main()