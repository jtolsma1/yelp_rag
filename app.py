import os
from pathlib import Path
import pandas as pd
import streamlit as st
import src.config as config

DATA_PATH = os.path.join(config.DATA_DIR_PROC,"summaries.parquet")

st.set_page_config(page_title="Yelp Restaurant Review Summaries", layout="wide")
st.title("Yelp Restaurant Review Summaries")

@st.cache_data
def load_summaries(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.reset_index().rename(columns = {"index":"restaurant_name"})
    # Basic safety: ensure expected columns exist
    expected = {"restaurant_name", "food", "service", "ambiance"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in summaries file: {missing}")
    # Clean up names for UX
    df["restaurant_name"] = df["restaurant_name"].astype("string").str.strip()
    df = df.dropna(subset=["restaurant_name"]).drop_duplicates("restaurant_name")
    df = df.sort_values("restaurant_name").reset_index(drop=True)
    return df

df = load_summaries(DATA_PATH)

# --- Restaurant selector ---
restaurant = st.selectbox(
    "Choose a restaurant",
    df["restaurant_name"].tolist(),
    index=0 if len(df) else None,
)

if df.empty:
    st.warning("No restaurants found in the summaries file.")
    st.stop()

row = df.loc[df["restaurant_name"] == restaurant].iloc[0]

st.subheader(row["restaurant_name"])

# --- Topic tabs ---
tab_food, tab_service, tab_ambiance = st.tabs(["Food", "Service", "Ambiance"])

with tab_food:
    st.markdown(row["food"] if pd.notna(row["food"]) else "No food summary available.")

with tab_service:
    st.markdown(row["service"] if pd.notna(row["service"]) else "No service summary available.")

with tab_ambiance:
    st.markdown(row["ambiance"] if pd.notna(row["ambiance"]) else "No ambiance summary available.")
