from __future__ import annotations

import copy

import numpy as np

from smart_signal.config import load_config
from smart_signal.policy import apply_trade_cooldown, decide_direction


def test_decide_direction_holds_when_edge_is_weak():
    cfg = load_config()
    probs = np.array([0.34, 0.33, 0.33], dtype=np.float64)
    cls, conf, diag = decide_direction(probs, cfg=cfg, expected_log_return=0.001)
    assert cls == 1
    assert diag["edge"] < float(cfg["inference"]["min_edge"])


def test_decide_direction_takes_strong_buy():
    cfg = load_config()
    probs = np.array([0.15, 0.20, 0.65], dtype=np.float64)
    cls, conf, diag = decide_direction(probs, cfg=cfg, expected_log_return=0.002)
    assert cls == 2
    assert conf >= float(cfg["inference"]["min_confidence"])
    assert diag["edge"] >= float(cfg["inference"]["min_edge"])


def test_decide_direction_rejects_return_disagreement_when_enabled():
    cfg = copy.deepcopy(load_config())
    cfg["inference"]["require_return_align"] = True
    cfg["inference"]["return_align_min_abs"] = 0.0
    probs = np.array([0.15, 0.20, 0.65], dtype=np.float64)
    cls, _, _ = decide_direction(probs, cfg=cfg, expected_log_return=-0.002)
    assert cls == 1


def test_trade_cooldown_skips_overlapping_entries():
    rows = [
        {"signal": "BUY", "price": 1.0},
        {"signal": "SELL", "price": 1.0},
        {"signal": "BUY", "price": 1.0},
    ]
    out = apply_trade_cooldown(rows, cooldown_bars=2)
    assert out[0]["signal"] == "BUY"
    assert out[1]["signal"] == "HOLD"
    assert out[2]["signal"] == "BUY"
