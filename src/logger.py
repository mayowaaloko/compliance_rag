import logging
import sys
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# ---- FORMATTER ----------------------------------------------
formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---- FILE HANDLER -------------------------------------------
# Writes all logs (DEBUG and above) to a file
file_handler = logging.FileHandler(LOG_DIR / "compliance_rag.log", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

# ---- CONSOLE HANDLER ----------------------------------------
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# ---- ROOT LOGGER --------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[file_handler, console_handler],
)

# Suppress noisy third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("qdrant_client").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Call this in every module: logger = get_logger(__name__)"""
    return logging.getLogger(name)
