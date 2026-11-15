"""Model IDs and settings. Everything can be overridden with environment variables."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = Path(os.environ.get("LEMONLENS_CACHE", ROOT / "data" / "cache"))


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal .env reader (KEY=VALUE per line); existing env vars win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


load_dotenv()

# Latest Cohere models (v2 API).
CHAT_MODEL = os.environ.get("LEMONLENS_CHAT_MODEL", "command-a-plus-05-2026")
# Embed 5 Pro and Fast share one embedding space: index with Pro, query with Fast.
EMBED_DOC_MODEL = os.environ.get("LEMONLENS_EMBED_DOC_MODEL", "embed-v5.0-pro")
EMBED_QUERY_MODEL = os.environ.get("LEMONLENS_EMBED_QUERY_MODEL", "embed-v5.0-fast")
RERANK_MODEL = os.environ.get("LEMONLENS_RERANK_MODEL", "rerank-v5.0-pro")

# Matryoshka output size: one of 256, 512, 768, 1024, 1536, 2048.
EMBED_DIM = int(os.environ.get("LEMONLENS_EMBED_DIM", "1024"))

# Retrieval: vector shortlist size, then how many survive reranking.
CANDIDATES = int(os.environ.get("LEMONLENS_CANDIDATES", "50"))
TOP_K = int(os.environ.get("LEMONLENS_TOP_K", "8"))

# Agent: thinking-token budget per step (0 disables thinking) and max tool rounds.
THINKING_BUDGET = int(os.environ.get("LEMONLENS_THINKING_BUDGET", "2000"))
MAX_AGENT_STEPS = int(os.environ.get("LEMONLENS_MAX_STEPS", "6"))

# Cap on complaints embedded per session (most recent are kept).
MAX_COMPLAINTS = int(os.environ.get("LEMONLENS_MAX_COMPLAINTS", "4000"))


PLACEHOLDER_KEY = "your-cohere-api-key-here"  # the value shipped in .env.example


def api_key() -> str | None:
    """The configured Cohere key, or None if unset or still the placeholder."""
    key = (os.environ.get("COHERE_API_KEY") or os.environ.get("CO_API_KEY") or "").strip()
    return None if not key or key == PLACEHOLDER_KEY else key
