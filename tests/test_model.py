from __future__ import annotations

import torch

from smart_signal.config import load_config
from smart_signal.data.dataset import MTFGoldDataset, build_timeframes, label_signal_frame, valid_indices
from smart_signal.data.synthetic import synthetic_gold
from smart_signal.models.goldnet import build_goldnet, count_parameters


def _frames(n: int = 1600):
    cfg = load_config()
    cfg = {
        **cfg,
        "timeframes": {
            "15m": {"lookback": 32},
            "1h": {"lookback": 16},
            "4h": {"lookback": 12},
            "1d": {"lookback": 8},
        },
    }
    raw = synthetic_gold(n, seed=11)
    frames = build_timeframes(base_15m=raw)
    return label_signal_frame(frames, cfg), cfg


def test_model_forward_shapes():
    frames, cfg = _frames(1800)
    idx = valid_indices(frames, cfg)
    assert len(idx) > 20
    ds = MTFGoldDataset(frames, cfg, idx[:8])
    batch = {k: torch.stack([ds[i][k] for i in range(4)]) for k in ds[0]}
    model = build_goldnet(cfg)
    out = model(batch, explain=True)
    assert out["dir_logits"].shape == (4, 3)
    assert out["y_ret"].shape == (4,)
    assert out["y_vol"].shape == (4,)
    assert out["vsn_15m"].shape[0] == 4
    assert count_parameters(model) > 100_000


def test_windows_match_lookback():
    frames, cfg = _frames(1600)
    idx = valid_indices(frames, cfg)
    item = MTFGoldDataset(frames, cfg, idx[:3])[0]
    assert item["x_15m"].shape[0] == cfg["timeframes"]["15m"]["lookback"]
    assert item["x_1h"].shape[0] == cfg["timeframes"]["1h"]["lookback"]
    assert item["x_4h"].shape[0] == cfg["timeframes"]["4h"]["lookback"]
    assert item["x_1d"].shape[0] == cfg["timeframes"]["1d"]["lookback"]
    assert item["x_15m"].shape[1] == len(cfg["features"])
    assert item["y_dir"] in (0, 1, 2)
