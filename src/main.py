import torch
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

from src.models import TennisGRUModel
from src.features import build_features

app = FastAPI(
    title="Tennis Win Probability API",
    description="Inference service for real-time tennis match outcome prediction using a calibrated GRU model.",
    version="1.0.0"
)

# Global model artifacts
model = None
scaler_stats = None
temperature = 1.0
input_dim = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.on_event("startup")
@app.on_event("startup")
def load_artifacts():
    global model, scaler_stats, temperature, input_dim
    print("Loading model artifacts and configuration...")
    
    ROOT_DIR = Path(__file__).resolve().parent.parent
    
    scaler_path = ROOT_DIR / "models" / "scaler_stats.pkl"
    temp_path = ROOT_DIR / "models" / "temperature.pkl"
    model_path = ROOT_DIR / "models" / "tennis_gru.pth"
    
    with open(scaler_path, "rb") as f:
        scaler_stats = pickle.load(f)
        
    with open(temp_path, "rb") as f:
        temperature = pickle.load(f)
        
    # Load model state dict first to extract exact input dimension safely
    model_state = torch.load(model_path, map_location=device)
    input_dim = model_state['gru.weight_ih_l0'].shape[1]
    
    model = TennisGRUModel(input_dim=input_dim, hidden_dim=64).to(device)
    model.load_state_dict(model_state)
    model.eval()
    print("Model loaded successfully!")

class MatchPayload(BaseModel):
    points: List[Dict[str, Any]]

@app.post("/predict")
def predict_match(payload: MatchPayload):
    if not payload.points:
        raise HTTPException(status_code=400, detail="Point sequence payload cannot be empty.")
        
    try:
        # 1. Convert incoming JSON point records into a DataFrame
        df = pd.DataFrame(payload.points)
        
        # 2. Build inference features using saved scaler stats
        feature_df, _ = build_features(df, is_train=False, scaler_stats=scaler_stats)
        
        # 3. Format into batch tensors: shape (1, seq_len, input_dim)
        features_tensor = torch.tensor(feature_df.values, dtype=torch.float32).unsqueeze(0).to(device)
        mask = torch.ones((1, features_tensor.size(1)), dtype=torch.float32).to(device)
        
        # 4. Run model inference
        with torch.no_grad():
            preds = model(features_tensor, mask)
            match_length = mask.sum(dim=1).long() - 1
            final_logit = preds[0, match_length]
            
            # Apply temperature scaling and sigmoid calibration
            calibrated_logit = final_logit / temperature
            p1_prob = torch.sigmoid(calibrated_logit).item()
            
        return {
            "p1_win_probability": round(p1_prob, 4),
            "p2_win_probability": round(1.0 - p1_prob, 4),
            "points_evaluated": len(df)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "healthy", "device": str(device), "model_loaded": model is not None}