from pathlib import Path
import os

# pathing
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR_RAW = os.path.join(PROJECT_DIR,"data","raw")
DATA_DIR_SAMP = os.path.join(PROJECT_DIR,"data","sampled")
DATA_DIR_PROC = os.path.join(PROJECT_DIR,"data","processed")
INDEX_DIR = PROJECT_DIR / "indexes"
CACHE_DIR = PROJECT_DIR / "cache"
INDEX_EXTENSIONS = (".faiss",".parquet")

# dataset import and setup
N_IMPORT_ROWS = 500000
MIN_REVIEWS = 100
N_RESTAURANTS = 2
RANDOM_STATE = 5 # set to None for random selection

# column definitions
COL_BUSINESS_CATEGORY = "categories"
COL_RESTAURANT_ID = "business_id"
COL_RESTAURANT_NAME = "name"
COL_REVIEW_ID = "review_id"
COL_DATE = "date"
COL_STARS = "stars_reviews"
COL_TEXT = "text"

# cleaning & chunking params
MIN_REVIEW_CHARS = 30 # characters
CHUNK_CHARS = 1000
OVERLAP_CHARS = 200
MIN_CHUNK_CHARS = 230

# model params
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_BATCH_SIZE = 64
EMBED_DEVICE = "mps"
NORMALIZE_EMBEDDINGS = True

# indexing settings
INDEX_METRIC = "cosine"
TOP_K_PER_TOPIC = 25
MAX_CHUNKS_PER_TOPIC = 12
MAX_CHUNKS_PER_REVIEW = 1

def index_path(restaurant_id: str) -> Path:
    return INDEX_DIR / f"{restaurant_id}.faiss"

def meta_path(restaurant_id: str) -> Path:
    return INDEX_DIR / f"{restaurant_id}_meta.parquet"

# topics
TOPICS = {
    "food":"food taste flavor menu dishes portion fresh spicy presentation",
    "service":"service staff wait time hostess server rude friendly attentive",
    "ambiance":"ambiance atmosphere decor music lighting seating noise vibe"
}

# local LLM
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_BASE_URL = "http://localhost:11434/api/generate"
TEMPERATURE = 0.4
MAX_OUTPUT_TOKENS = 600

# caching & logging
ENABLE_CACHE = True
CACHE_VER = "v1"
LOG_LEVEL = "INFO"

# saving objects
METADATA_COLS = {
    "chunk_id_col":"chunk_id",
    "business_id_col":"business_id",
    "restaurant_name_col":"restaurant_name",
    "stars_col":"stars",
    "review_id_col":"review_id",
    "chunk_col":"chunk"
}
PARQUET_ENGINE = "pyarrow"