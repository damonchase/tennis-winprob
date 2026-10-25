import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import pickle
from pathlib import Path

from models import TennisSequenceDataset, tennis_collate_fn, TennisGRUModel
from features import build_features

ROOT = Path(__file__).resolve().parents[1]

def calibrate_model():
    print("Loading validation data and trained GRU model...")
    raw_df = pd.read_parquet(ROOT / "data" / "processed" / "points_men.parquet")
    val_raw = raw_df[raw_df["split"] == "val"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(ROOT / "models" / "scaler_stats.pkl", "rb") as f:
        scaler_stats = pickle.load(f)
        
    val_feature_df, _ = build_features(val_raw, is_train=False, scaler_stats=scaler_stats)
    val_dataset = TennisSequenceDataset(val_feature_df, val_raw["p1_wins"], val_raw["match_id"])
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=tennis_collate_fn)
    
    input_dim = val_feature_df.shape[1]
    model = TennisGRUModel(input_dim=input_dim, hidden_dim=64).to(device)
    model.load_state_dict(torch.load(ROOT / "models" / "tennis_gru.pth", map_location=device, weights_only=True))
    model.eval()
    
    # Collect predictions and labels for every real validation point.
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch_features, batch_labels, mask in val_loader:
            batch_features = batch_features.to(device)
            mask = mask.to(device)
            
            preds = model(batch_features, mask)
            valid = mask.bool()
            point_labels = batch_labels.unsqueeze(1).expand_as(preds)
            all_logits.append(preds[valid].cpu())
            all_labels.append(point_labels[valid].cpu())
            
    logits_tensor = torch.cat(all_logits).to(device)
    labels_tensor = torch.cat(all_labels).to(device)

    log_temperature = nn.Parameter(torch.zeros((), device=device))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50)
    criterion = nn.BCEWithLogitsLoss()
    
    def closure():
        optimizer.zero_grad()
        scaled_logits = logits_tensor / log_temperature.exp()
        loss = criterion(scaled_logits, labels_tensor)
        loss.backward()
        return loss
    
    optimizer.step(closure)
    temperature = float(log_temperature.exp().item())
    print(f"Validation pointwise temperature found: {temperature:.4f}")
    
    with open(ROOT / "models" / "temperature.pkl", "wb") as f:
        pickle.dump(temperature, f)

if __name__ == "__main__":
    calibrate_model()