from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from smart_signal.features.candle_structure import refine_path_olhc_from_1m
from smart_signal.labels.next_candle import decode_next_ohlc, next_candle_labels
from smart_signal.models.losses import multi_task_loss


def test_decode_next_ohlc_keeps_close_inside_range():
    hi, lo, cl = decode_next_ohlc(2000.0, 5.0, 1.2, 0.8, 0.75)
    assert lo <= cl <= hi
    assert hi > lo


def test_next_candle_labels_atr_scaled():
    n = 40
    df = pd.DataFrame(
        {
            "open": np.linspace(2000, 2020, n),
            "high": np.linspace(2001, 2022, n),
            "low": np.linspace(1999, 2018, n),
            "close": np.linspace(2000.5, 2021, n),
            "atr": np.full(n, 2.0),
        }
    )
    out = next_candle_labels(df)
    assert set(["y_candle", "y_up", "y_dn", "y_close_loc"]).issubset(out.columns)
    assert (out["y_up"].iloc[:-1] >= 0).all()
    assert (out["y_dn"].iloc[:-1] >= 0).all()
    assert ((out["y_close_loc"].iloc[:-1] >= 0) & (out["y_close_loc"].iloc[:-1] <= 1)).all()


def test_refine_path_olhc_from_1m_detects_low_first():
    # Parent 15m bar 10:00, high=10, low=0; 1m hits low then high.
    parent = pd.DataFrame(
        {
            "time": pd.to_datetime(["2024-01-01T10:00:00Z", "2024-01-01T10:15:00Z"]),
            "high": [10.0, 11.0],
            "low": [0.0, 1.0],
            "path_olhc": [0.0, 0.0],
        }
    )
    child = pd.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "2024-01-01T10:00:00Z",
                    "2024-01-01T10:01:00Z",
                    "2024-01-01T10:02:00Z",
                    "2024-01-01T10:03:00Z",
                ]
            ),
            "high": [1.0, 2.0, 10.0, 10.0],
            "low": [0.5, 0.0, 1.0, 1.0],
        }
    )
    out = refine_path_olhc_from_1m(parent, child)
    assert float(out.loc[0, "path_olhc"]) == 1.0


def test_structure_path_prior_nudges_bullish_geometry():
    from smart_signal.structure_prior import calibrate_candle_probs, calibrate_path

    up, dn, loc = calibrate_path(
        0.8, 0.8, 0.5, ms_bias=0.9, htf_trend_align=0.8, premium_discount=-0.5, strength=0.3
    )
    assert loc > 0.5
    assert up >= dn
    # Candle prior off by default (strength=0) — probs unchanged.
    base = np.array([0.3, 0.4, 0.3], dtype=np.float64)
    out = calibrate_candle_probs(base, ms_bias=0.9, htf_trend_align=0.8, premium_discount=-0.5)
    assert np.allclose(out, base / base.sum())


def test_consistency_loss_runs():
    b = 8
    outputs = {
        "dir_logits": torch.randn(b, 3),
        "candle_logits": torch.randn(b, 3),
        "y_ret": torch.randn(b),
        "y_vol": torch.rand(b),
        "y_up": torch.rand(b) + 0.1,
        "y_dn": torch.rand(b) + 0.1,
        "y_close_loc": torch.rand(b),
    }
    batch = {
        "y_dir": torch.randint(0, 3, (b,)),
        "y_candle": torch.randint(0, 3, (b,)),
        "y_ret": torch.randn(b),
        "y_vol": torch.rand(b),
        "y_up": torch.rand(b),
        "y_dn": torch.rand(b),
        "y_close_loc": torch.rand(b),
        "y_struct": torch.randint(0, 3, (b,)),
        "atr": torch.rand(b) + 1.0,
        "close": torch.full((b,), 2000.0),
    }
    loss, parts = multi_task_loss(
        outputs,
        batch,
        class_weight=None,
        gamma=2.0,
        return_w=0.3,
        vol_w=0.1,
        candle_w=0.4,
        path_w=0.3,
        consistency_w=0.2,
        price_w=0.2,
        struct_w=0.1,
    )
    assert torch.isfinite(loss)
    assert "consistency" in parts
    assert "price" in parts
    assert "struct" in parts

def test_blend_candle_probs_mixes_structure_prior():
    import numpy as np
    from smart_signal.candle_blend import blend_candle_probs, load_candle_blend

    blend = load_candle_blend()
    assert blend is not None
    row = {c: 0.0 for c in blend["cols"]}
    row["bull_bear"] = 1.0
    row["ms_bias"] = 0.8
    row["htf_trend_align"] = 0.7
    row["premium_discount"] = -0.4
    row["path_olhc"] = 1.0
    base = np.array([0.34, 0.33, 0.33], dtype=np.float64)
    out = blend_candle_probs(base, row, alpha=0.35, blend=blend)
    assert out.shape == (3,)
    assert abs(out.sum() - 1.0) < 1e-6
    # With bullish structure, blend should not decrease bull vs bear share vs flat base.
    assert out[2] - out[0] >= base[2] - base[0] - 1e-6
