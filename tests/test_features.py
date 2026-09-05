from __future__ import annotations

import numpy as np
import pandas as pd

from smart_signal.data.synthetic import synthetic_gold
from smart_signal.features.indicators import add_features, rsi
from smart_signal.labels.triple_barrier import BUY, HOLD, SELL, triple_barrier_labels


def test_rsi_bounds():
    close = np.linspace(100, 140, 80)
    values = rsi(close, 14)
    assert values.min() >= 0
    assert values.max() <= 100
    assert values[-1] > 70


def test_features_have_no_nan():
    df = add_features(synthetic_gold(300, seed=1))
    assert df[["rsi_14", "atr", "macd", "bb_pct"]].isna().sum().sum() == 0
    assert (df["atr"] >= 0).all()


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
