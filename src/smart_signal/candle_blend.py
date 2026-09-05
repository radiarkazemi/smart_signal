"""Blend GoldNet candle probs with a structure-only ICT/MTF logistic prior.

The logistic is trained on candle+ICT+nested-MTF columns only (no leakage).
Alpha is selected on an inner validation split; default artifact ships alpha=0.35.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

_BLEND: dict[str, Any] | None = None


def _default_paths() -> list[Path]:
    from smart_signal.config import artifacts_dir, models_dir

    return [
        models_dir() / "candle_structure_blend.joblib",
        artifacts_dir() / "candle_structure_blend.joblib",
        Path("models/candle_structure_blend.joblib"),
        Path("artifacts/candle_structure_blend.joblib"),
    ]


def load_candle_blend(path: Path | None = None) -> dict[str, Any] | None:
    global _BLEND
    if path is None and _BLEND is not None:
        return _BLEND
    try:
        import joblib
    except Exception:
        return None
    candidates = [path] if path is not None else _default_paths()
    for p in candidates:
        if p is None:
            continue
        p = Path(p)
        if not p.exists():
            continue
        payload = joblib.load(p)
        _BLEND = payload
        return payload
    return None


def structure_candle_probs(row: dict[str, Any] | Any, blend: dict[str, Any] | None = None) -> np.ndarray | None:
    """Return 3-class probs from the structure logistic for one feature row."""
    blend = blend or load_candle_blend()
    if blend is None:
        return None
    cols = blend["cols"]
    vals = []
    for c in cols:
        try:
            v = float(row[c]) if not hasattr(row, "get") else float(row.get(c, 0.0) or 0.0)
        except Exception:
            v = 0.0
        if not np.isfinite(v):
            v = 0.0
        vals.append(v)
    x = np.asarray(vals, dtype=np.float64).reshape(1, -1)
    x = blend["scaler"].transform(x)
    proba = blend["clf"].predict_proba(x)[0]
    out = np.zeros(3, dtype=np.float64)
    for j, c in enumerate(blend["clf"].classes_):
        out[int(c)] = float(proba[j])
    s = out.sum()
    return out / s if s > 0 else np.array([1 / 3, 1 / 3, 1 / 3])


def blend_candle_probs(
    net_probs: np.ndarray,
    row: dict[str, Any] | Any,
    *,
    alpha: float | None = None,
    blend: dict[str, Any] | None = None,
) -> np.ndarray:
    """Mix network candle probs with structure prior. alpha=0 => network only."""
    p = np.asarray(net_probs, dtype=np.float64).reshape(-1).copy()
    if p.size != 3:
        return p
    p = np.clip(p, 1e-8, 1.0)
    p /= p.sum()
    blend = blend or load_candle_blend()
    if blend is None:
        return p
    a = float(blend.get("alpha", 0.35) if alpha is None else alpha)
    a = float(np.clip(a, 0.0, 1.0))
    if a <= 1e-9:
        return p
    s = structure_candle_probs(row, blend=blend)
    if s is None:
        return p
    out = (1.0 - a) * p + a * s
    out = np.clip(out, 1e-8, 1.0)
    return out / out.sum()
