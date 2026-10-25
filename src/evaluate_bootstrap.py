import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timezone
from pathlib import Path
import shutil
from sklearn.metrics import log_loss

from models import TennisSequenceDataset, tennis_collate_fn, TennisGRUModel
from features import build_features

ROOT = Path(__file__).resolve().parents[1]

def run_bootstrap():
    print("Loading data, baselines, and trained GRU model...")
    raw_df = pd.read_parquet(ROOT / "data" / "processed" / "points_men.parquet")
    baseline_df = pd.read_parquet(ROOT / "data" / "processed" / "baseline_preds.parquet")
    
    eval_raw = raw_df[raw_df["split"] == "test"].copy()
    
    # Load trained GRU model and scaler stats
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(ROOT / "models" / "scaler_stats.pkl", "rb") as f:
        scaler_stats = pickle.load(f)

    with open(ROOT / "models" / "temperature.pkl", "rb") as f:
        temperature = pickle.load(f)
        
    eval_feature_df, _ = build_features(eval_raw, is_train=False, scaler_stats=scaler_stats)
    eval_dataset = TennisSequenceDataset(eval_feature_df, eval_raw["p1_wins"], eval_raw["match_id"])
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False, collate_fn=tennis_collate_fn)
    
    input_dim = eval_feature_df.shape[1]
    model = TennisGRUModel(input_dim=input_dim, hidden_dim=64).to(device)
    model.load_state_dict(torch.load(ROOT / "models" / "tennis_gru.pth", map_location=device, weights_only=True))
    model.eval()
    
    # Initialize dictionary for match-level log loss and lists for point-by-point CSV
    point_predictions = []
    dataset_match_ids = eval_dataset.match_ids
    source_match_ids = eval_raw["match_id"].drop_duplicates().tolist()
    if dataset_match_ids != source_match_ids:
        raise RuntimeError("Dataset match IDs no longer match the source match order.")
    batch_idx_offset = 0

    with torch.no_grad():
        for batch_features, _, mask in eval_loader:
            batch_features = batch_features.to(device)
            mask = mask.to(device)
            
            # Forward pass outputs logits of shape (batch_size, seq_len)
            preds = model(batch_features, mask)
            
            # Apply temperature scaling and sigmoid
            calibrated_preds = preds / temperature
            probs = torch.sigmoid(calibrated_preds).cpu().numpy()
            lengths = mask.sum(dim=1).long().cpu().tolist()
            
            batch_size = probs.shape[0]
            for i in range(batch_size):
                m_id = dataset_match_ids[batch_idx_offset + i]
                valid_len = lengths[i]
                match_points = eval_raw.loc[eval_raw["match_id"].eq(m_id)].sort_values("point_idx")
                if len(match_points) != valid_len or not np.array_equal(
                    match_points["point_idx"].to_numpy(), np.arange(valid_len)
                ):
                    raise RuntimeError(f"Prediction sequence does not align with points for match {m_id}.")
                point_predictions.append(pd.DataFrame({
                    "match_id": m_id,
                    "point_idx": match_points["point_idx"].to_numpy(),
                    "gru": probs[i, :valid_len],
                }))
                    
            batch_idx_offset += batch_size

    out = pd.concat(point_predictions, ignore_index=True)
    baseline_test = baseline_df.loc[baseline_df["split"].eq("test"), ["match_id", "point_idx"]]
    if baseline_test.duplicated(["match_id", "point_idx"]).any():
        raise RuntimeError("Baseline test predictions contain duplicate point keys.")
    if out.duplicated(["match_id", "point_idx"]).any():
        raise RuntimeError("GRU predictions contain duplicate point keys.")
    expected_keys = pd.MultiIndex.from_frame(baseline_test)
    actual_keys = pd.MultiIndex.from_frame(out[["match_id", "point_idx"]])
    if len(expected_keys) != len(actual_keys) or not expected_keys.isin(actual_keys).all() or not actual_keys.isin(expected_keys).all():
        missing = expected_keys.difference(actual_keys)
        extra = actual_keys.difference(expected_keys)
        raise RuntimeError(f"Prediction keys differ from baseline: {len(missing)} missing, {len(extra)} extra.")

    output_path = ROOT / "data" / "processed" / "gru_test_preds.csv"
    if output_path.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup_path = output_path.with_name(f"{output_path.stem}.backup.{stamp}{output_path.suffix}")
        shutil.copy2(output_path, backup_path)
        print(f"Backed up existing predictions to {backup_path}")
    out.sort_values(["match_id", "point_idx"]).to_csv(output_path, index=False)
    print(f"Saved {len(out)} pointwise test predictions for {out['match_id'].nunique()} matches to {output_path}")

    # Compare final-point predictions using the same test labels and matches.
    eval_df = baseline_df[baseline_df["split"] == "test"]
    match_groups = eval_df.groupby("match_id")
    match_ids = list(match_groups.groups.keys())
    n_matches = len(match_ids)
    
    markov_losses = []
    gru_losses = []
    
    for match_id in match_ids:
        group = match_groups.get_group(match_id)
        y_true = group["p1_wins"].iloc[-1]
        
        # Markov loss
        p_markov = float(group["markov"].iloc[-1])
        p_markov = np.clip(p_markov, 1e-6, 1 - 1e-6)
        markov_losses.append(log_loss([y_true], [p_markov], labels=[0, 1]))
        
        # GRU loss (looked up from dictionary)
        p_gru = float(out.loc[out["match_id"].eq(match_id)].sort_values("point_idx")["gru"].iloc[-1])
        p_gru = np.clip(p_gru, 1e-6, 1 - 1e-6)
        gru_losses.append(log_loss([y_true], [p_gru], labels=[0, 1]))

    markov_losses = np.array(markov_losses)
    gru_losses = np.array(gru_losses)

    # Paired bootstrap loop over matches (1000 iterations)
    rng = np.random.default_rng(42)
    diffs = []
    
    for _ in range(1000):
        idx = rng.integers(0, n_matches, n_matches)
        mean_diff = (markov_losses[idx] - gru_losses[idx]).mean()
        diffs.append(mean_diff)

    # Compute 95% Confidence Interval
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"\nPaired bootstrap 95% CI (test, Markov - GRU final-point log loss): [{lo:+.4f}, {hi:+.4f}]")

if __name__ == "__main__":
    run_bootstrap()