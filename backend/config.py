# config.py - every setting, threshold, path and model tier name lives here.
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Paths
DATA_DIR = ROOT / "data"
QUERIES_CSV = DATA_DIR / "queries.csv"
PRICES_YAML = DATA_DIR / "prices.yaml"
MODEL_RUNS_CSV = DATA_DIR / "model_runs.csv"
ROUTER_PKL = DATA_DIR / "router.pkl"
DB_PATH = DATA_DIR / "app.db"
CHROMA_DIR = DATA_DIR / "chroma"

# Mock mode: MOCK_LLM=1 -> fake answers, no API calls, no spend
MOCK_LLM = os.getenv("MOCK_LLM", "0") == "1"
MOCK_ACCURACY = {"small": 0.60, "medium": 0.80, "large": 0.95}

# Model tiers, cheapest first (names map to litellm models in prices.yaml)
TIERS = ["small", "medium", "large"]

# LLM call settings
MAX_OUTPUT_TOKENS = 512
REQUEST_TIMEOUT_S = 60
SAMPLE_TEMPERATURE = 0.7      # for the self-consistency re-samples (step 7)

# Budget: once today's spend reaches this, force the small model
DAILY_BUDGET = 1.00           # in the currency of prices.yaml

# Cache (steps 2 + 4)
CACHE_SIM_THRESHOLD = 0.92
CACHE_TOP_K = 3

# Router (step 5)
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
KNN_K = 5
TAU_SMALL = 0.70              # defaults; train.py overwrites these in router.pkl
TAU_MEDIUM = 0.70
QUALITY_TARGET = 0.95         # router must reach 95% of always-large accuracy on tune

# Escalation (step 7)
MAX_ESCALATIONS = 2
