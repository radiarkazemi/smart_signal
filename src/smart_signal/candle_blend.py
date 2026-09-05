"""Blend GoldNet candle probs with a structure-only ICT/MTF logistic prior.

Portable artifact stores scaler + logistic coefficients (no sklearn model pickle),
so live servers remain compatible across scikit-learn versions.
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
        # Accept legacy sklearn-pickle artifacts by extracting coefficients.
        if "coef" not in payload and "clf" in payload:
            clf = payload["clf"]
            sc = payload["scaler"]
            payload = {
                "cols": payload["cols"],
                "alpha": float(payload.get("alpha", 0.35)),
                "classes": [int(c) for c in clf.classes_],
                "coef": np.asarray(clf.coef_, dtype=np.float64),
                "intercept": np.asarray(clf.intercept_, dtype=np.float64),
                "scaler_mean": np.asarray(sc.mean_, dtype=np.float64),
                "scaler_scale": np.asarray(sc.scale_, dtype=np.float64),
            }
        _BLEND = payload
        return payload
    return None


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.maximum(e.sum(), 1e-12)


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
    x = np.asarray(vals, dtype=np.float64)
    mean = np.asarray(blend["scaler_mean"], dtype=np.float64)
    scale = np.asarray(blend["scaler_scale"], dtype=np.float64)
    scale = np.where(scale <= 1e-12, 1.0, scale)
    x = (x - mean) / scale
    coef = np.asarray(blend["coef"], dtype=np.float64)
    intercept = np.asarray(blend["intercept"], dtype=np.float64)
    classes = [int(c) for c in blend["classes"]]
    if coef.ndim == 1:
        coef = coef.reshape(1, -1)
    # sklearn multinomial: logits = X @ coef.T + intercept
    logits = coef @ x + intercept
    if logits.size == 1 and len(classes) == 2:
        # binary fallback
        p1 = 1.0 / (1.0 + np.exp(-float(logits[0])))
        probs = {classes[0]: 1.0 - p1, classes[1]: p1}
    else:
        sm = _softmax(logits)
        probs = {classes[i]: float(sm[i]) for i in range(len(classes))}
    out = np.array([probs.get(0, 0.0), probs.get(1, 0.0), probs.get(2, 0.0)], dtype=np.float64)
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
