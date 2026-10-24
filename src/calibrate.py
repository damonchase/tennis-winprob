import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import pickle

from models import TennisSequenceDataset, tennis_collate_fn, TennisGRUModel
from features import build_features

def calibrate_model():
    print("Loading validation data and trained GRU model...")
    raw_df = pd.read_parquet("../data/processed/points_men.parquet")
    val_raw = raw_df[raw_df["split"] == "val"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open("../models/scaler_stats.pkl", "rb") as f:
        scaler_stats = pickle.load(f)
        
    val_feature_df, _ = build_features(val_raw, is_train=False, scaler_stats=scaler_stats)
    val_dataset = TennisSequenceDataset(val_feature_df, val_raw["p1_wins"], val_raw["match_id"])
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=tennis_collate_fn)
    
    input_dim = val_feature_df.shape[1]
    model = TennisGRUModel(input_dim=input_dim, hidden_dim=64).to(device)
    model.load_state_dict(torch.load("../models/tennis_gru.pth", map_location=device))
    model.eval()
    
    # 1. Collect all uncalibrated logits and true labels at the final timestep for each match
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch_features, batch_labels, mask in val_loader:
            batch_features = batch_features.to(device)
            mask = mask.to(device)
            
            # TODO: Get model raw logits (BEFORE sigmoid)
            preds = model(batch_features, mask)
            match_lengths = mask.sum(dim=1).long() - 1
            final_logits = preds[torch.arange(preds.size(0), device=device), match_lengths]
            
            all_logits.append(final_logits.cpu())
            all_labels.append(batch_labels)
            
    logits_tensor = torch.cat(all_logits).to(device)
    labels_tensor = torch.cat(all_labels).to(device)
    
    # 2. Define Temperature parameter T and LBFGS optimizer
    T = nn.Parameter(torch.ones(1, device=device) * 1.5)
    optimizer = torch.optim.LBFGS([T], lr=0.1, max_iter=50)
    
    # 3. Optimize T using BCEWithLogitsLoss closure
    criterion = nn.BCEWithLogitsLoss()
    
    def closure():
        optimizer.zero_grad()
        # TODO: Scale logits by T and compute loss
        scaled_logits = logits_tensor / T
        loss = criterion(scaled_logits, labels_tensor)
        loss.backward()
        return loss
    
    optimizer.step(closure)
    print(f"Optimal Temperature T found: {T.item():.4f}")
    
    # 4. Save the optimal temperature parameter
    with open("../models/temperature.pkl", "wb") as f:
        pickle.dump(T.item(), f)

if __name__ == "__main__":
    calibrate_model()