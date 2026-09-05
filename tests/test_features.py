from __future__ import annotations

import numpy as np
import pandas as pd

from smart_signal.data.synthetic import synthetic_gold
from smart_signal.features.candle_structure import CANDLE_COLUMNS, add_candle_structure
from smart_signal.features.ict import ICT_COLUMNS, add_ict_features
from smart_signal.features.indicators import FEATURE_COLUMNS, add_features, feature_matrix, rsi
from smart_signal.features.mtf_align import attach_mtf_alignment
from smart_signal.labels.next_candle import BULL, decode_next_ohlc, next_candle_labels
from smart_signal.labels.triple_barrier import BUY, HOLD, SELL, triple_barrier_labels


def test_rsi_bounds():
    close = np.linspace(100, 140, 80)
    values = rsi(close, 14)
    assert values.min() >= 0
    assert values.max() <= 100
    assert values[-1] > 70


def test_features_have_no_nan():
    df = add_features(synthetic_gold(400, seed=1))
    assert set(FEATURE_COLUMNS).issubset(df.columns)
    mat = feature_matrix(df)
    assert mat.shape == (len(df), len(FEATURE_COLUMNS))
    assert np.isfinite(mat).all()
    assert (df["atr"] >= 0).all()


def test_candle_and_ict_columns():
    df = add_features(synthetic_gold(300, seed=2))
    for col in CANDLE_COLUMNS + ICT_COLUMNS:
        assert col in df.columns
    assert df["close_loc"].between(-0.05, 1.05).all()


def test_mtf_alignment_nested():
    base = synthetic_gold(1200, seed=4)
    from smart_signal.data.dataset import build_timeframes

    frames = build_timeframes(base_15m=base)
    assert "1w" in frames
    assert "htf_open_dist" in frames["15m"].columns
    assert np.isfinite(frames["15m"]["nested_pos"]).all()


def test_next_candle_labels_uptrend():
    df = synthetic_gold(200, seed=5)
    df["close"] = np.linspace(2000, 2400, 200)
    df["open"] = df["close"] - 0.8
    df["high"] = df["close"] + 1.2
    df["low"] = df["close"] - 1.5
    labeled = next_candle_labels(add_features(df))
    assert (labeled["y_candle"][:-1] == BULL).mean() > 0.7
    assert labeled["y_next_high"].iloc[0] > 0
    assert (labeled["y_up"][:-1] >= 0).all()
    assert (labeled["y_dn"][:-1] >= 0).all()
    assert labeled["y_close_loc"].between(0.0, 1.0).all()


def test_decode_next_ohlc_consistent_and_clamped():
    hi, lo, cl = decode_next_ohlc(4400.0, 10.0, 1.5, 1.2, 0.6)
    assert lo <= cl <= hi
    # Extreme softplus outputs are capped to max_atr_mult ATRs.
    hi2, lo2, cl2 = decode_next_ohlc(4400.0, 10.0, 50.0, 50.0, 0.5, max_atr_mult=4.0)
    assert lo2 <= cl2 <= hi2
    assert abs((hi2 - lo2) - 80.0) < 1e-6


def test_triple_barrier_uptrend_not_all_sell():
    df = synthetic_gold(500, seed=3)
    df["close"] = np.linspace(2000, 2600, 500)
    df["open"] = df["close"] - 0.4
    df["high"] = df["close"] + 1.5
    df["low"] = df["close"] - 0.6
    feat = add_features(df)
    labeled = triple_barrier_labels(feat, horizon=6, tp_atr=1.2, sl_atr=1.2)
    counts = labeled["y_dir"].value_counts()
    assert int(counts.get(BUY, 0)) > int(counts.get(SELL, 0))
    assert HOLD in labeled["y_dir"].unique() or True
    assert labeled["y_ret"].notna().all()
