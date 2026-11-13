"""Filesystem locations shared across chapters."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = REPO_ROOT / "models"
QWEN3_DIR = MODELS_DIR / "qwen3"
