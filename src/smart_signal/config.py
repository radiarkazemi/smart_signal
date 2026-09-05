from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "goldnet.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    cfg["_path"] = str(cfg_path)
    cfg["_root"] = str(ROOT)
    return cfg


def env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name, default)
    return value if value not in {"", None} else default


def data_dir() -> Path:
    path = ROOT / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def artifacts_dir() -> Path:
    path = ROOT / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir() -> Path:
    path = ROOT / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoint_path(cfg: dict[str, Any] | None = None) -> Path:
    override = env("SMART_SIGNAL_CHECKPOINT")
    if override:
        return Path(override)
    return models_dir() / "goldnet.pt"
