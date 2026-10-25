import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import pickle
import random
from pathlib import Path
from sklearn.metrics import roc_auc_score, brier_score_loss

from models import TennisSequenceDataset, tennis_collate_fn, TennisGRUModel
from features import build_features

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def masked_point_loss(logits, labels, mask):
    targets = labels.unsqueeze(1).expand_as(logits)
    losses = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (losses * mask).sum() / mask.sum()


def train_epoch(model, dataloader, optimizer, device):
    """
    Trains the TennisGRUModel for one epoch.
    """
    model.train()  # Put model in training mode
    total_loss = 0.0
    total_points = 0
    
    for batch_features, batch_labels, mask in dataloader:
        # 1. Move tensors to the correct device (CPU or GPU)
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)
        mask = mask.to(device)
        
        # 2. Zero out the gradients from the previous step
        optimizer.zero_grad()
        
        # 3. Forward pass: get predictions from the model
        preds = model(batch_features, mask)
        
        loss = masked_point_loss(preds, batch_labels, mask)
        
        # 6. Backward pass: compute gradients
        loss.backward()
        
        # 7. Optimizer step: update model weights
        optimizer.step()
        
        valid_points = int(mask.sum().item())
        total_loss += loss.item() * valid_points
        total_points += valid_points

    return total_loss / total_points

def evaluate_model(model, dataloader, device):
    """
    Evaluates the model on validation data, computing loss, accuracy, ROC-AUC, and Brier score.
    """
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_points = 0
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_features, batch_labels, mask in dataloader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            mask = mask.to(device)
            
            preds = model(batch_features, mask)
            
            loss = masked_point_loss(preds, batch_labels, mask)
            valid_points = int(mask.sum().item())
            total_loss += loss.item() * valid_points
            total_points += valid_points

            valid = mask.bool()
            valid_logits = preds[valid]
            valid_labels = batch_labels.unsqueeze(1).expand_as(preds)[valid]
            probs = torch.sigmoid(valid_logits)
            predicted_labels = (probs >= 0.5).float()
            
            correct_predictions += (predicted_labels == valid_labels).sum().item()
            
            # Collect for advanced metrics
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(valid_labels.cpu().numpy())
            
    avg_loss = total_loss / total_points
    accuracy = correct_predictions / total_points if total_points > 0 else 0.0
    
    # Compute ROC-AUC and Brier score
    roc_auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    brier = brier_score_loss(all_labels, all_probs)
    
    return avg_loss, accuracy, roc_auc, brier

if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 1. Set device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Load the processed point-by-point dataset
    print("Loading data...")
    raw_df = pd.read_parquet(ROOT / "data" / "processed" / "points_men.parquet")
    
    # 3. Filter into train and validation splits using the 'split' column
    print("Splitting datasets...")
    train_raw = raw_df[raw_df["split"] == "train"]
    val_raw = raw_df[raw_df["split"] == "val"]
    
    # 5. Build features (fit scaler on train data, apply to validation data)
    print("Building features...")
    train_feature_df, scaler_stats = build_features(train_raw, is_train=True)
    val_feature_df, _ = build_features(val_raw, is_train=False, scaler_stats=scaler_stats)
    
    # 6. Instantiate datasets
    train_dataset = TennisSequenceDataset(train_feature_df, train_raw["p1_wins"], train_raw["match_id"])
    val_dataset = TennisSequenceDataset(val_feature_df, val_raw["p1_wins"], val_raw["match_id"])
    
    # 7. Create DataLoaders
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=tennis_collate_fn,
                              generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=tennis_collate_fn)
    
    # 8. Initialize model using the shape of your feature DataFrame
    input_dim = train_feature_df.shape[1]
    hidden_dim = 64
    model = TennisGRUModel(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    
    # 9. Loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 10. Run the training loop over multiple epochs
    num_epochs = 10
    print("Starting training loop...")
    best_val_loss = float("inf")
    best_state = None
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        val_loss, val_acc, val_roc, val_brier = evaluate_model(model, val_loader, device)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Val ROC-AUC: {val_roc:.4f} | "
              f"Val Brier: {val_brier:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training produced no validation checkpoint.")
    model_dir = ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, model_dir / "tennis_gru.pth")
    with open(model_dir / "scaler_stats.pkl", "wb") as f:
        pickle.dump(scaler_stats, f)
        
    print("Model and scaler saved successfully to 'models/' directory!")