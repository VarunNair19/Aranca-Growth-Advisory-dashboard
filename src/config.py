import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
import os
import warnings

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings(
    "ignore",
    message=r".*huggingface_hub.*cache-system uses symlinks.*",
)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
LOGS_DIR = ROOT_DIR / "logs"
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Free LLM API Keys (Google AI Studio + optional Groq fallback)
    gemini_api_key: str = ""
    groq_api_key: str = ""

    # Models
    primary_llm_model: str = "gemini-3.8-flash"
    fast_gate_model: str = "gemini-3.5-flash-lite"
    fallback_llm_model: str = "openai/gpt-oss-120b"
    embedding_model: str = "all-MiniLM-L6-v2"

    # File Paths
    db_path: Path = DATA_DIR / "intel.db"
    aranca_profile_path: Path = DATA_DIR / "aranca_profile.json"
    competitors_path: Path = DATA_DIR / "competitors.json"

    # Retention & Ingestion Windows
    baseline_lookback_days: int = 90
    default_retention_days: int = 180  # 6 months default cleanup cutoff

    # Circuit Breakers & Rate-Limit Safeguards
    max_articles_per_run: int = 60
    max_llm_calls_per_run: int = 35
    llm_call_delay_sec: float = 2.0
    scrape_delay_sec: float = 1.5
    dedupe_similarity_threshold: float = 0.85
    max_validation_retries: int = 2
    max_upload_bytes: int = 10 * 1024 * 1024  # 10 MB

def setup_logging(level: int = logging.INFO) -> None:
    """Configures console and rotating file logging once across the application."""
    root_logger = logging.getLogger("aranca_intel")
    if root_logger.handlers:
        return

    root_logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        LOGS_DIR / "aranca_intel.log",
        maxBytes=5 * 1024 * 1024,  # 5 MB per log file
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Returns a namespaced child logger under 'aranca_intel'."""
    setup_logging()
    clean_name = name.removeprefix("src.")
    return logging.getLogger(f"aranca_intel.{clean_name}")

settings = Settings()