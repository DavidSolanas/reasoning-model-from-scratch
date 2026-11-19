"""Filesystem locations shared across chapters."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
QWEN3_DIR = MODELS_DIR / "qwen3"
