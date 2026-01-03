from pathlib import Path
import os

# pathing
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR_RAW = os.path.join(PROJECT_DIR,"data","raw")
DATA_DIR_SAMP = os.path.join(PROJECT_DIR,"data","sampled")
DATA_DIR_PROC = os.path.join(PROJECT_DIR,"data","processed")
INDEX_DIR = PROJECT_DIR / "indexes"
CACHE_DIR = PROJECT_DIR / "cache"

# column definitions
COL_RESTAURANT_ID = ""
COL_RESTAURANT_NAME = ""
COL_REVIEW_ID = ""
COL_DATE = ""
COL_STARS = ""
COL_TEXT = ""

# chunking params
CHUNK_MAX_TOKENS = 280
CHUNK_OVERLAP_TOKENS = 60
MIN_TOKENS_TO_CHUNK = 320
MIN_CHUNK_TOKENS = 30

# model params
EMBEDDING_MODEL_NAME = ""
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

TOPIC_ORDER = {"food","service","ambiance"}

# local LLM
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_BASE_URL = "http://localhost:11434"
TEMPERATURE = 0.2
MAX_OUTPUT_TOKENS = 600

# caching & logging
ENABLE_CACHE = True
CACHE_VER = "v1"
LOG_LEVEL = "INFO"