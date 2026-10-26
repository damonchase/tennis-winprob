import json

import streamlit as st
import pandas as pd
import requests

st.title("🎾 Tennis Live Win Probability")

@st.cache_data
def load_metadata():
    return pd.read_csv("github-data/matches_men.csv")

matches_df = load_metadata()

# Select match
match_id = st.selectbox("Select Match ID", matches_df["match_id"].tolist())
match_row = matches_df[matches_df["match_id"] == match_id].iloc[0]

st.markdown(f"**{match_row['player1']}** vs **{match_row['player2']}** *({match_row['slam'].upper()} {match_row['year']})*")

# Load points for selected match
points_df = pd.read_parquet("github-data/points_men.parquet")
match_points = points_df[points_df["match_id"] == match_id]

# Point progression slider
n_points = st.slider("Evaluate up to point number:", 1, len(match_points), min(20, len(match_points)))

if st.button("Calculate Win Probability", type="primary"):
    subset = match_points.head(n_points)
    payload = json.dumps({"points": subset.to_dict(orient="records")})
    
    try:
        response = requests.post(
            "https://tennis-winprob-api-6840016701.us-central1.run.app/predict",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        if response.status_code == 200:
            data = response.json()
            p1_prob = data["p1_win_probability"]
            p2_prob = data["p2_win_probability"]
            
            col1, col2 = st.columns(2)
            col1.metric(label=match_row['player1'], value=f"{p1_prob:.1%}")
            col2.metric(label=match_row['player2'], value=f"{p2_prob:.1%}")
        else:
            st.error(f"API Error: {response.text}")
    except Exception as e:
        st.error(f"Failed to connect to FastAPI server at port 8000: {e}")