import os
from pathlib import Path
import pandas as pd
import streamlit as st
from run_pipeline import YelpRAGPipelineRunner
from src import config


# ------------------------------------
# Basic config
# ------------------------------------

DATA_PATH = os.path.join(config.DATA_DIR_PROC,"summaries.parquet")
DATA_PATH = Path(DATA_PATH)

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


# ------------------------------------
# Initialize input-output objects
# ------------------------------------

if "df" not in st.session_state:
    st.session_state.df = None

if "n_restaurants" not in st.session_state:
    st.session_state.n_restaurants = config.N_RESTAURANTS

if "seed" not in st.session_state:
    st.session_state.seed = 1

if "randomize" not in st.session_state:
    st.session_state.randomize = False

col1, col2, col3  = st.columns([2,2,1])

with col1:
    seed_val = st.number_input(
        "Seed (integer)",
        min_value = 1,
        step = 1,
        value = int(st.session_state.seed),
        disabled = st.session_state.randomize,
        help = "Random restaurant selections will be reproduced when given the same seed value."
    )

with col2:
    n_restaurants_val = st.number_input(
        "Number of Restaurants",
        min_value = 2,
        max_value = 25,
        step = 1,
        value = int(st.session_state.n_restaurants),
        help = "Input the number of restaurants to appear in the summary output."
    )

with col3:
    randomize = st.checkbox(
        "Randomize",
        value = st.session_state.randomize,
        help = "if checked, seed is ignored and a truly random selection of restaurants is summarized."
    )

st.session_state.seed = int(seed_val)
st.session_state.n_restaurants = int(n_restaurants_val)
st.session_state.randomize = bool(randomize)

random_state = None if st.session_state.randomize else st.session_state.seed

st.caption(f"Effective random state: `{random_state}`")

status_box = st.empty()
progress_bar = st.progress(0)


# ------------------------------------
# Enable messaging from inside the RAG
# ------------------------------------

def make_status_cb():
    "Obtain message emissions from the summarization pipeline for serving in the streamlit app"
    def status_cb(event:dict):
        msg = event.get("message","")
        restaurant_no = event.get("restaurant_no")
        total = event.get("total")

        if msg:
            status_box.info(msg)

        if restaurant_no is not None and total:
            progress_bar.progress(int((restaurant_no / total)*100))

    return status_cb


# ------------------------------------
# Create data caching function with error handling
# ------------------------------------

@st.cache_data
def load_summaries(path: Path, mtime_ns: int) -> pd.DataFrame:
    """
    Load LLM-generated summaries from parquet file;
    Clean up restaurant names and handle mising column errors.
    @param path: file path to retrieve summaries parquet file
    @param mtime_ns: nanosecond timestamp of data retrieval to avoid pulling from old caches
    @return: cleaned summaries dataframe
    """
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


# ------------------------------------
# Introduce execution button
# ------------------------------------

if st.button("Run RAG Pipeline"):

    progress_bar.progress(0)
    cb = make_status_cb()

    cb({"message": "Retrieving Yelp reviews...", "restaurant_no": 0, "total": 10})

    runner = YelpRAGPipelineRunner()
    runner.run_pipeline(random_state=random_state,n_restaurants = n_restaurants_val,status_cb=cb)

    mtime_ns = DATA_PATH.stat().st_mtime_ns
    df = load_summaries(DATA_PATH,mtime_ns)

    st.session_state.df = df


# ------------------------------------
# Serve RAG summaries using topic tabs
# ------------------------------------

df = st.session_state.df

if df is None:
    status_box.info("Click the button to generate review summaries.")
else:
    options = sorted(df["restaurant_name"].unique().tolist())
    selected = st.selectbox("Select a restaurant to view review summaries",options)
    filtered = df[df["restaurant_name"] == selected].iloc[0].copy()

    tab_food,tab_service,tab_ambiance = st.tabs(["Food","Service","Ambiance"])
    with tab_food:
        st.markdown(filtered["food"] if pd.notna(filtered["food"]) else "No food summary available.")

    with tab_service:
        st.markdown(filtered["service"] if pd.notna(filtered["service"]) else "No service summary available.")

    with tab_ambiance:
        st.markdown(filtered["ambiance"] if pd.notna(filtered["ambiance"]) else "No ambiance summary available.")
