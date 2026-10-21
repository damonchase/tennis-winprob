"""Download Jeff Sackmann's open tennis data into data/raw/.

Data (c) Jeff Sackmann, CC BY-NC-SA 4.0 -- https://tennisabstract.com
Do not commit the downloaded files; run this script instead.

Jeff Sackmann's original GitHub repos (tennis_slam_pointbypoint, tennis_atp) were
removed from GitHub in 2026, so this script reads a community archival mirror that
keeps the same file names:
    https://github.com/Aneeshers/tennis-sackmann-archive
(slam point-by-point snapshot: Oct 2024; ATP/WTA snapshots: Jun 2026)

The script asks GitHub which files the mirror actually contains instead of
guessing file names, and records where the data came from in data/raw/SOURCE.txt.

Usage (from the repo root):
    python src/data.py
"""
import re
import time
from collections import Counter
from datetime import date
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

OWNER = "Aneeshers"
REPO = "tennis-sackmann-archive"

SLAM_DIR = "slam_pointbypoint/"
ATP_DIR = "atp/"

SLAMS = ["ausopen", "frenchopen", "wimbledon", "usopen"]
SLAM_YEARS = range(2011, 2025)   # point-by-point data we plan to use
ATP_YEARS = range(2000, 2025)    # tour-level results for Elo (years of burn-in before 2011)

# Singles files only: the name ends right after "points" or "matches"
# (doubles/mixed files have extra suffixes and are skipped).
SLAM_RE = re.compile(rf"(?:^|/)(\d{{4}})-({'|'.join(SLAMS)})-(points|matches)\.csv$")
ATP_RE = re.compile(r"(?:^|/)atp_matches_(\d{4})\.csv$")
DOC_NAMES = {"data_dictionary.txt", "matches_data_dictionary.txt", "UPSTREAM_README.md"}


def get_json(url: str, **kwargs):
    resp = requests.get(url, timeout=60, **kwargs)
    if resp.status_code == 403:
        raise SystemExit("GitHub API rate limit hit. Wait ~1 hour or try a different network.")
    resp.raise_for_status()
    return resp.json()


def list_repo_files():
    """Return (branch, commit_sha, [file paths]) for the mirror repo."""
    base = f"https://api.github.com/repos/{OWNER}/{REPO}"
    branch = get_json(base)["default_branch"]
    try:
        sha = get_json(f"{base}/commits/{branch}")["sha"]
    except requests.RequestException:
        sha = "unknown"
    tree = get_json(f"{base}/git/trees/{branch}", params={"recursive": "1"})
    if tree.get("truncated"):
        print("  warning: the file list was truncated")
    paths = [t["path"] for t in tree["tree"] if t["type"] == "blob"]
    return branch, sha, paths


def download(url: str, dest: Path, retries: int = 3) -> str:
    """Download url to dest unless it already exists. Returns a status string."""
    if dest.exists() and dest.stat().st_size > 0:
        return "cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code == 404:
                return "missing"
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            return "downloaded"
        except requests.RequestException as err:
            print(f"  retry {attempt + 1} for {url}: {err}")
            time.sleep(2 * (attempt + 1))
    return "failed"


def raw_url(branch: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{OWNER}/{REPO}/{branch}/{path}"


def main() -> None:
    branch, sha, paths = list_repo_files()
    print(f"{OWNER}/{REPO}: branch '{branch}', commit {sha[:7]}, {len(paths)} files")

    jobs = []  # (url, destination)

    # ---- Grand Slam point-by-point -------------------------------------------------
    found = {}
    for path in paths:
        if path.startswith(SLAM_DIR):
            m = SLAM_RE.search(path)
            if m:
                found[(int(m.group(1)), m.group(2), m.group(3))] = path
    wanted = {(y, s, k) for y in SLAM_YEARS for s in SLAMS for k in ("points", "matches")}
    extra_years = sorted({y for (y, _, _) in found} - set(SLAM_YEARS))
    missing_in_repo = sorted(wanted - set(found))
    for key in sorted(wanted & set(found)):
        jobs.append((raw_url(branch, found[key]), RAW / "slam" / Path(found[key]).name))

    # ---- ATP tour-level match results ---------------------------------------------
    for path in paths:
        if path.startswith(ATP_DIR):
            m = ATP_RE.search(path)
            if m and int(m.group(1)) in ATP_YEARS:
                jobs.append((raw_url(branch, path), RAW / "atp" / Path(path).name))

    # ---- Documentation files (data dictionaries, original READMEs) -----------------
    for path in paths:
        name = Path(path).name
        if name in DOC_NAMES and (path.startswith(SLAM_DIR) or path.startswith(ATP_DIR)):
            sub = "slam" if path.startswith(SLAM_DIR) else "atp"
            jobs.append((raw_url(branch, path), RAW / sub / name))

    # ---- Download ------------------------------------------------------------------
    results = Counter()
    problems = []
    for i, (url, dest) in enumerate(jobs, start=1):
        status = download(url, dest)
        results[status] += 1
        if status in ("missing", "failed"):
            problems.append((status, url))
        if status == "downloaded":
            print(f"[{i}/{len(jobs)}] downloaded {dest.parent.name}/{dest.name}")
            time.sleep(0.2)  # be polite to GitHub

    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "SOURCE.txt").write_text(
        f"Downloaded {date.today()} from https://github.com/{OWNER}/{REPO} "
        f"(branch {branch}, commit {sha}).\n"
        "Original data by Jeff Sackmann (Tennis Abstract), CC BY-NC-SA 4.0.\n"
    )

    print("\nSummary:", dict(results))
    if extra_years:
        print(f"Slam years in the archive that we are NOT downloading: {extra_years}")
    if missing_in_repo:
        print("\nExpected slam files that are not in the archive:")
        for year, slam, kind in missing_in_repo:
            print(f"  {year}-{slam}-{kind}.csv")
    if problems:
        print("\nCheck these:")
        for status, url in problems:
            print(f"  {status}: {url}")


if __name__ == "__main__":
    main()