import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import pickle
import os
from sklearn.metrics import roc_auc_score, brier_score_loss

from models import TennisSequenceDataset, tennis_collate_fn, TennisGRUModel
from features import build_features

def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Trains the TennisGRUModel for one epoch.
    """
    model.train()  # Put model in training mode
    total_loss = 0.0
    
    for batch_features, batch_labels, mask in dataloader:
        # 1. Move tensors to the correct device (CPU or GPU)
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)
        mask = mask.to(device)
        
        # 2. Zero out the gradients from the previous step
        optimizer.zero_grad()
        
        # 3. Forward pass: get predictions from the model
        preds = model(batch_features, mask)
        
        # 4. Extract the prediction at the *final valid timestep* for each match using the mask
        match_lengths = mask.sum(dim=1).long() - 1
        final_preds = preds[torch.arange(preds.size(0), device=device), match_lengths]
        
        # 5. Compute the binary cross-entropy loss between final predictions and true match labels
        loss = criterion(final_preds, batch_labels)
        
        # 6. Backward pass: compute gradients
        loss.backward()
        
        # 7. Optimizer step: update model weights
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluates the model on validation data, computing loss, accuracy, ROC-AUC, and Brier score.
    """
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_matches = 0
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_features, batch_labels, mask in dataloader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            mask = mask.to(device)
            
            preds = model(batch_features, mask)
            
            match_lengths = mask.sum(dim=1).long() - 1
            final_preds = preds[torch.arange(preds.size(0), device=device), match_lengths]
            
            loss = criterion(final_preds, batch_labels)
            total_loss += loss.item()
            
            probs = torch.sigmoid(final_preds)
            predicted_labels = (probs >= 0.5).float()
            
            correct_predictions += (predicted_labels == batch_labels).sum().item()
            total_matches += batch_labels.size(0)
            
            # Collect for advanced metrics
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_matches if total_matches > 0 else 0.0
    
    # Compute ROC-AUC and Brier score
    roc_auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    brier = brier_score_loss(all_labels, all_probs)
    
    return avg_loss, accuracy, roc_auc, brier

if __name__ == "__main__":
    # 1. Set device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Load the processed point-by-point dataset
    print("Loading data...")
    raw_df = pd.read_parquet("../data/processed/points_men.parquet")
    
    # 3. Derive match-level outcome label (1 if Player 1 won more sets, 0 otherwise)
    print("Deriving match-level labels...")
    last_points = raw_df.groupby("match_id").last()
    match_winners = (last_points["p1_sets"] > last_points["p2_sets"]).astype(float)
    raw_df["p1_win"] = raw_df["match_id"].map(match_winners)
    
    # 4. Filter into train and validation splits using the 'split' column
    print("Splitting datasets...")
    train_raw = raw_df[raw_df["split"] == "train"]
    val_raw = raw_df[raw_df["split"] == "val"]
    
    # 5. Build features (fit scaler on train data, apply to validation data)
    print("Building features...")
    train_feature_df, scaler_stats = build_features(train_raw, is_train=True)
    val_feature_df, _ = build_features(val_raw, is_train=False, scaler_stats=scaler_stats)
    
    # 6. Instantiate datasets
    train_dataset = TennisSequenceDataset(train_feature_df, train_raw["p1_win"], train_raw["match_id"])
    val_dataset = TennisSequenceDataset(val_feature_df, val_raw["p1_win"], val_raw["match_id"])
    
    # 7. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=tennis_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=tennis_collate_fn)
    
    # 8. Initialize model using the shape of your feature DataFrame
    input_dim = train_feature_df.shape[1]
    hidden_dim = 64
    model = TennisGRUModel(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    
    # 9. Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 10. Run the training loop over multiple epochs
    num_epochs = 5
    print("Starting training loop...")
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Unpack all 4 values returned by evaluate_model
        val_loss, val_acc, val_roc, val_brier = evaluate_model(model, val_loader, criterion, device)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Val ROC-AUC: {val_roc:.4f} | "
              f"Val Brier: {val_brier:.4f}")    
        
    # After training loop finishes:
    os.makedirs("../models", exist_ok=True)
    
    # Save model weights
    torch.save(model.state_dict(), "../models/tennis_gru.pth")
    
    # Save scaler stats so we can use the exact same mean/std during inference
    with open("../models/scaler_stats.pkl", "wb") as f:
        pickle.dump(scaler_stats, f)
        
    print("Model and scaler saved successfully to 'models/' directory!")