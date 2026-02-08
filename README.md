# Yelp RAG: Restaurant Review Summarization

An end-to-end **retrieval-augmented generation (RAG)** system that summarizes restaurant reviews by topic (**food, service, ambiance**) using semantic search and a local LLM.

The project is built as a **production-style ML system**, not a notebook demo: data processing, retrieval, summarization, and UI are cleanly separated and fully reproducible.

---

### What This Project Demonstrates

- Practical RAG architecture without managed vector databases  
- Local semantic search using FAISS
- Topic-aware retrieval to reduce LLM hallucination  
- Config-driven, testable Python pipeline  
- Local LLM inference with Ollama 
- Lightweight Streamlit dashboard for exploration  

---

### High-Level Flow
```
Raw reviews
   → cleaning & chunking
   → embeddings via HuggingFace
   → FAISS indices on a per-restaurant basis
   → topic-based retrieval
   → LLM summarization using Ollama
   → parquet summaries
   → Streamlit dashboard
```

### Repo Structure
```
yelp_rag/
├── app.py                  # Streamlit dashboard
├── run_pipeline.py         # Pipeline entry point
├── src/
│   ├── config.py           # Central configuration
│   ├── cleaning.py         # Review cleaning & chunking
│   ├── embeddings.py       # Embedding generation
│   ├── retrieval.py        # FAISS similarity search
│   ├── summarization.py    # LLM prompts + Ollama calls
│   ├── data_io.py          # Parquet & filesystem utilities
│   ├── emit_util.py        # Pipeline status emission
│   └── __init__.py
```

**Note:** pipeline-generated artifacts (FAISS indices, metadata, summaries) are intentionally excluded from version control and 
are rebuilt by the pipeline for each execution.

---

### Running a Local Instance of This Project

#### 1. Clone the Repo
```
https://github.com/jtolsma1/yelp_rag.git
```

#### 2. Install requirements
Create a new virtual environment if desired (not shown), then
execute these commands in terminal to install dependencies:
```
cd /local/path/to/repo/yelp_rag/

pip install -r requirements.txt
```

#### 3. Run Ollama

**3a.** Install Ollama on local setup at https://ollama.com/download.<br>
**3b.** In terminal, download an LLM to run locally (will likely require multiple GB of free disk space); for example:<br>
```
ollama pull llama3.1:8b
```
**3c**. In `src/config.py` in this repo, input the name of the downloaded Ollama model into the `OLLAMA_MODEL` parameter.<br>
**3d (optional).** Configure other LLM parameters in `src/config.py`, such as the Ollama URL and temperature, if necessary.<br>
**3e.** Run Ollama on local setup; expect to see this UI upon opening, although the UI is not needed to run the code in this repo:<br><br>
<img src="images/ollama_ui.png" width=400 align="center"></img>

**Note:** If Ollama is not running, the summarization step will fail and throw an HTTP Connection Error; 
start or restart Ollama to resolve:<br><br>
<img src="images/ollama_missing_error.png" width = 900 align="center"></img>

#### 4. Run the Dashboard
In the terminal, run:
```
streamlit run app.py
```

---

### Why I Made This
* To teach myself vector search, LLM summarization, and Streamlit
* To practice productionizing ML pipelines