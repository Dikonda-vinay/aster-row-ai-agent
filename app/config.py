"""Central configuration for the Aster & Row support agent.

All tunable knobs live here so the rest of the code stays readable.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Paths -------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
KB_DIR = ROOT_DIR / "knowledge-base"
DATA_DIR = ROOT_DIR / "data"
ORDERS_PATH = DATA_DIR / "orders.json"
INDEX_CACHE_PATH = ROOT_DIR / ".cache" / "kb_index.pkl"

# --- LLM -----------------------------------------------------------------
# Using Groq's free tier (OpenAI-compatible API, no billing required).
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "llama-3.3-70b-versatile")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1024"))

# --- Embeddings ------------------------------------------------------------
# Local sentence-transformers model. No external API key required, and
# results are deterministic given the same model version, which matters
# for reproducible eval runs.
EMBEDDING_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)

# --- Retrieval ----------------------------------------------------------
TOP_K = int(os.environ.get("RETRIEVAL_TOP_K", "6"))
# Minimum cosine similarity for a chunk to be considered relevant at all.
# Below this we treat the corpus as not containing an answer.
MIN_RELEVANCE_SCORE = float(os.environ.get("MIN_RELEVANCE_SCORE", "0.28"))

# --- Session --------------------------------------------------------------
MAX_HISTORY_TURNS = int(os.environ.get("MAX_HISTORY_TURNS", "12"))

# --- Logging --------------------------------------------------------------
LOG_DIR = ROOT_DIR / "logs"
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
