"""Project paths and runtime configuration."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DOC_DIR = PROJECT_ROOT / "docs"
PUBLIC_DIR = PROJECT_ROOT / "public_datasets"
REPORT_DIR = PROJECT_ROOT / "reports"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

for directory in [DATA_DIR, DOC_DIR, PUBLIC_DIR, REPORT_DIR, ARTIFACT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


RANDOM_STATE = 42
USE_LOCAL_LLM = os.getenv("MW_USE_LLM", "1").strip().lower() not in {"0", "false", "no"}
LOCAL_LLM_MODEL_ID = os.getenv("MW_LLM_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
EMBED_MODEL_ID = os.getenv("MW_EMBED_MODEL_ID", "BAAI/bge-small-en-v1.5")
