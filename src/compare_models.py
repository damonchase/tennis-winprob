import torch
from torch.utils.data import DataLoader
import pandas as pd
import pickle
from sklearn.metrics import roc_auc_score, brier_score_loss

from models import TennisSequenceDataset, tennis_collate_fn, TennisGRUModel
from features import build_features

def compare_models():
    print("Loading test data and baselines (GRU calibrated on validation data)...")
    raw_df = pd.read_parquet("../data/processed/points_men.parquet")
    baseline_df = pd.read_parquet("../data/processed/baseline_preds.parquet")
    
    eval_raw = raw_df[raw_df["split"] == "test"]
    
    # 2. Load trained GRU model and scaler stats
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
    
    # 3. Get GRU predictions at the final timestep of each match
    gru_probs = []
    gru_labels = []
    
    with torch.no_grad():
        for batch_features, batch_labels, mask in eval_loader:
            batch_features = batch_features.to(device)
            mask = mask.to(device)
            
            preds = model(batch_features, mask)
            
            # TODO: Extract final valid timestep for each match in the batch
            # Hint: match_lengths = mask.sum(dim=1).long() - 1
            match_lengths = mask.sum(dim=1).long() - 1
            final_preds = preds[torch.arange(preds.size(0), device=device), match_lengths]
            probs = torch.sigmoid(final_preds / temperature)
            
            gru_probs.extend(probs.cpu().numpy())
            gru_labels.extend(batch_labels.numpy())
            
    # TODO: Compute ROC-AUC and Brier score for the GRU model
    gru_auc = roc_auc_score(gru_labels, gru_probs)
    gru_brier = brier_score_loss(gru_labels, gru_probs)
    
    # 4. Extract baseline predictions for validation matches at their final point
    eval_baselines = baseline_df[baseline_df["split"] == "test"].groupby("match_id").last()
    
    elo_probs = eval_baselines["elo_only"].to_numpy()
    markov_probs = eval_baselines["markov"].to_numpy()
    
    # TODO: Gather ground-truth labels corresponding to the baseline match IDs
    labels_by_match = eval_raw.groupby("match_id")["p1_wins"].first()
    baseline_labels = labels_by_match.reindex(eval_baselines.index).to_numpy()
    
    # TODO: Compute metrics for elo_only and markov baselines
    elo_auc = roc_auc_score(baseline_labels, elo_probs)
    elo_brier = brier_score_loss(baseline_labels, elo_probs)
    
    markov_auc = roc_auc_score(baseline_labels, markov_probs)
    markov_brier = brier_score_loss(baseline_labels, markov_probs)

    # 5. Print out the final comparison table
    print("\n" + "=" * 55)
    print(f"{'Model':<22} | {'ROC-AUC':<10} | {'Brier Score':<12}")
    print("-" * 55)
    print(f"{'Elo-Only Baseline':<22} | {elo_auc:.4f}     | {elo_brier:.4f}")
    print(f"{'Markov Chain Baseline':<22} | {markov_auc:.4f}     | {markov_brier:.4f}")
    print(f"{'GRU Sequence Model':<22} | {gru_auc:.4f}     | {gru_brier:.4f}")
    print("=" * 55)

if __name__ == "__main__":
    compare_models()