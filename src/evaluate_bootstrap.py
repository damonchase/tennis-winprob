import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import log_loss

from models import TennisSequenceDataset, tennis_collate_fn, TennisGRUModel
from features import build_features

def run_bootstrap():
    print("Loading data, baselines, and trained GRU model...")
    raw_df = pd.read_parquet("../data/processed/points_men.parquet")
    baseline_df = pd.read_parquet("../data/processed/baseline_preds.parquet")
    
    eval_raw = raw_df[raw_df["split"] == "test"]
    
    # Load trained GRU model and scaler stats
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open("../models/scaler_stats.pkl", "rb") as f:
        scaler_stats = pickle.load(f)

    with open("../models/temperature.pkl", "rb") as f:
        temperature = pickle.load(f)
        
    eval_feature_df, _ = build_features(eval_raw, is_train=False, scaler_stats=scaler_stats)
    eval_dataset = TennisSequenceDataset(eval_feature_df, eval_raw["p1_wins"], eval_raw["match_id"])
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False, collate_fn=tennis_collate_fn)
    
    input_dim = eval_feature_df.shape[1]
    model = TennisGRUModel(input_dim=input_dim, hidden_dim=64).to(device)
    model.load_state_dict(torch.load("../models/tennis_gru.pth", map_location=device))
    model.eval()
    
    # Initialize dictionary for match-level log loss and lists for point-by-point CSV
    gru_preds_dict = {}
    test_match_ids = []
    test_point_idx = []
    test_probs = []

    dataset_match_ids = list(eval_raw["match_id"].unique())
    batch_idx_offset = 0

    with torch.no_grad():
        for batch_features, batch_labels, mask in eval_loader:
            batch_features = batch_features.to(device)
            mask = mask.to(device)
            
            # Forward pass outputs logits of shape (batch_size, seq_len)
            preds = model(batch_features, mask)
            
            # Apply temperature scaling and sigmoid
            calibrated_preds = preds / temperature
            probs = torch.sigmoid(calibrated_preds).cpu().numpy()
            mask_np = mask.cpu().numpy()
            
            batch_size = probs.shape[0]
            for i in range(batch_size):
                m_id = dataset_match_ids[batch_idx_offset + i]
                valid_len = int(mask_np[i].sum())
                
                for t in range(valid_len):
                    test_match_ids.append(m_id)
                    test_point_idx.append(t)
                    test_probs.append(float(probs[i, t]))
                
                # Store the final valid point's probability for match-level evaluation
                gru_preds_dict[m_id] = float(probs[i, valid_len - 1])
                    
            batch_idx_offset += batch_size

    # Save point-by-point test predictions to CSV
    import os
    os.makedirs("../data/processed", exist_ok=True)
    out = pd.DataFrame({
        "match_id": test_match_ids,
        "point_idx": test_point_idx,
        "gru": test_probs,
    })
    out.to_csv("../data/processed/gru_test_preds.csv", index=False)
    print("Saved point-by-point GRU test predictions to ../data/processed/gru_test_preds.csv")        # ----------------------------
    # Filter test split for baselines
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
        p_gru = gru_preds_dict[match_id]
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
    print(f"\nPaired bootstrap 95% CI (test, Markov - GRU log loss): [{lo:+.4f}, {hi:+.4f}]")

if __name__ == "__main__":
    run_bootstrap()