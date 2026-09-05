from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_gold(n: int = 4000, start: float = 2300.0, seed: int = 7) -> pd.DataFrame:
    """GBM + mean-reverting noise with sessions — for tests and smoke training."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / (252 * 96)
    rets = rng.normal(0.00002, 0.0011, size=n)
    # mild autocorrelation + occasional jumps
    for i in range(1, n):
        rets[i] += 0.18 * rets[i - 1]
    jumps = rng.choice([0.0, 1.0], size=n, p=[0.992, 0.008]) * rng.normal(0, 0.006, size=n)
    close = start * np.exp(np.cumsum(rets + jumps))
    noise = np.abs(rng.normal(0.0, 0.35, size=n))
    high = close + noise
    low = close - noise * rng.uniform(0.4, 1.2, size=n)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    idx = pd.date_range("2024-01-02", periods=n, freq="15min", tz="UTC")
    df = pd.DataFrame(
        {
            "time": idx,
            "open": open_,
            "high": np.maximum.reduce([high, open_, close]),
            "low": np.minimum.reduce([low, open_, close]),
            "close": close,
            "volume": rng.integers(50, 400, size=n).astype(np.float64),
        }
    )
    return df
