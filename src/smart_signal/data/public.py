from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from smart_signal.config import data_dir
from smart_signal.data.ohlcv import bars_from_yahoo_chart, save_parquet

YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
DEFAULT_SYMBOL = "GC=F"
TIMEOUT = 45.0


def fetch_yahoo_chart(
    symbol: str = DEFAULT_SYMBOL,
    interval: str = "1h",
    range_: str = "2y",
) -> pd.DataFrame:
    url = YAHOO_CHART.format(symbol=symbol)
    params = {"interval": interval, "range": range_}
    headers = {"User-Agent": "Mozilla/5.0 smart-signal/0.1"}
    with httpx.Client(timeout=TIMEOUT, headers=headers, follow_redirects=True) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        payload = resp.json()
    return bars_from_yahoo_chart(payload)


def load_yahoo_cache(path: Path) -> pd.DataFrame:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return bars_from_yahoo_chart(payload)


def cache_public_gold(frames: dict[str, pd.DataFrame]) -> dict[str, Path]:
    out_dir = data_dir() / "public"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for tf, df in frames.items():
        path = out_dir / f"gc_{tf}.parquet"
        save_parquet(df, path)
        written[tf] = path
    return written
