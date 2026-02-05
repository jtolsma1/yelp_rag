import os
from pathlib import Path
import pandas as pd
import streamlit as st
from run_pipeline import YelpRAGPipelineRunner
import src.config as config

DATA_PATH = os.path.join(config.DATA_DIR_PROC,"summaries.parquet")

st.set_page_config(page_title="Yelp Restaurant Review Summaries", layout="wide")
st.title("Yelp Restaurant Review Summaries")

st.markdown(
"""
This dashboard summarizes a sample of Yelp customer reviews using a
retrieval-augmented generation (RAG) pipeline.

Select a restaurant below to see concise summaries of what customers
say about **food**, **service**, and **ambiance**.
"""
)

# --- Restaurant selector ---

if "seed" not in st.session_state:
    st.session_state.seed = 5
if "randomize" not in st.session_state:
    st.session_state.randomize = False

col1, col2 = st.columns([2,1])

with col1:
    seed_val = st.number_input(
        "Seed (integer)",
        min_value = 0,
        step = 1,
        value = int(st.session_state.seed),
        disabled = st.session_state.randomize,
        help = "Random restaurant selections will be reproduced when given the same seed value."
    )
with col2:
    randomize = st.checkbox(
        "Randomize",
        value = st.session_state.randomize,
        help = "if checked, seed is ignored and a truly random selection of restaurants is summarized."
    )

st.session_state.seed = int(seed_val)
st.session_state.randomize = bool(randomize)

random_state = None if st.session_state.randomize else st.session_state.seed

st.caption(f"Effective random state: `{random_state}`")

status_box = st.empty()
progress_bar = st.progress(0)

def make_status_cb():
    def status_cb(event:dict):
        msg = event.get("message","")
        restaurant_no = event.get("restaurant_no")
        total = event.get("total")

        if msg:
            status_box.info(msg)
        
        if restaurant_no is not None and total:
            progress_bar.progress(int((restaurant_no / total)*100))
    
    return status_cb

if st.button("Run pipeline (test status)"):
    status_box.info("Starting pipeline...")
    progress_bar.progress(0)
    cb = make_status_cb()
    
    cb({"message": "Callback is wired ✅", "restaurant_no": 0, "total": 10})

    runner = YelpRAGPipelineRunner()
    runner.run_pipeline(random_state=random_state,status_cb=cb)

    status_box.success("Pipeline finished.")
    progress_bar.progress(100)

@st.cache_data
def load_summaries(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
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

try:
    df = load_summaries(DATA_PATH)
except:
    st.warning("Run the pipeline to generate summaries.")


restaurant = st.selectbox(
    "Restaurant",
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
