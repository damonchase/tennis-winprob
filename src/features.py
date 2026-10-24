import pandas as pd
from typing import Optional

def compute_running_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes running in-match statistics for each player up to the current point.
    CRITICAL: Must not use the current point's outcome or future points.
    """
    stats = pd.DataFrame(index=df.index)
    
    # 1. Did the server win this point?
    server_won = df["server"] == df["point_winner"]
    server_won = server_won.astype(int)
    
    # 2. Track cumulative stats per match, shifted by 1 so point t doesn't see point t's outcome
    temp_df = pd.DataFrame({
        "match_id": df["match_id"],
        "server_won": server_won,
        "serve_played": 1
    })
    
    # Group by match_id and shift within each match so no previous match can leak in.
    grouped = temp_df.groupby("match_id", sort=False)[["server_won", "serve_played"]].cumsum()
    grouped = grouped.groupby(temp_df["match_id"], sort=False).shift(1, fill_value=0)
    
    # 3. Compute smoothed running serve win rate (e.g., adding pseudocounts: 2 wins out of 3 points at ~64% tour average)
    # This prevents division by zero and extreme 0% or 100% rates on the very first point of a match.
    smoothed_wins = grouped["server_won"] + 1.92  # roughly 0.64 * 3
    smoothed_total = grouped["serve_played"] + 3.0
    
    stats["server_running_win_rate"] = smoothed_wins / smoothed_total
    
    return stats

def build_features(df: pd.DataFrame, is_train: bool = True, scaler_stats=None) -> tuple[pd.DataFrame, Optional[dict]]:    
    """
    Builds the final feature matrix for model training or inference.
    
    Parameters:
    - df: Raw point-by-point DataFrame from build_dataset.py.
    - is_train: If True, compute mean/std for scaling. If False, apply stored scaler_stats.
    - scaler_stats: Dict containing mean and std from training data (if is_train=False).
    
    Returns:
    - feature_df: DataFrame containing model-ready features.
    - scaler_stats: The fitted scaling parameters (to be reused during validation/test/serving).
    """
    features = pd.DataFrame(index=df.index)
    
    # 1. Score-state features
    features["p1_sets"] = df["p1_sets"]
    features["p2_sets"] = df["p2_sets"]
    features["set_no"] = df["set_no"]
    features["p1_games"] = df["p1_games"]
    features["p2_games"] = df["p2_games"]
    features["p1_pts"] = df["p1_pts"]
    features["p2_pts"] = df["p2_pts"]
    features["is_tiebreak"] = df["is_tiebreak"]
    features["server"] = df["server"]
    
    # 2. Pre-match Elo context
    if "elo_diff_logit" in df.columns:
        features["elo_diff_logit"] = df["elo_diff_logit"]
    
    # 3. Bring in our running stats
    running_stats = compute_running_stats(df)
    for col in running_stats.columns:
        features[col] = running_stats[col]
    
    # 4. Standardization of continuous features (e.g., elo_diff_logit)
    # TODO: We need to write the scaling logic here so that train mean/std are used.
    # 4. Standardization of continuous features
    continuous_cols = ["elo_diff_logit", "server_running_win_rate"]
    
    if is_train:
        scaler_stats = {}
        for col in continuous_cols:
            if col in features.columns:
                mean_val = features[col].mean()
                std_val = features[col].std()
                # Handle edge case where std might be 0 or NaN
                if pd.isna(std_val) or std_val == 0:
                    std_val = 1.0
                scaler_stats[col] = {"mean": mean_val, "std": std_val}
                features[col] = (features[col] - mean_val) / std_val
    else:
        if scaler_stats is not None:
            for col, stats in scaler_stats.items():
                if col in features.columns:
                    mean_val = stats["mean"]
                    std_val = stats["std"]
                    features[col] = (features[col] - mean_val) / std_val
    
    return features, scaler_stats