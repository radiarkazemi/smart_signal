from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
import pandas as pd

from smart_signal.config import data_dir, env
from smart_signal.data.ohlcv import bars_from_records, jsonl_to_frame, save_parquet

DEFAULT_API_BASE = "http://185.222.163.116/crypto-api"
DEFAULT_BARS_URL = "http://185.222.163.116/trh-api/bars"
TIMEOUT = 45.0


def api_base() -> str:
    return (env("SMART_SIGNAL_API_BASE") or DEFAULT_API_BASE).rstrip("/")


def bars_url() -> str:
    return env("SMART_SIGNAL_BARS_URL") or DEFAULT_BARS_URL


def fetch_last_price(symbol: str = "xauusd", timeframe: str = "1m") -> dict[str, Any]:
    url = f"{api_base()}/prices/{symbol}/?timeframe={timeframe}"
    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.json()


def fetch_history_page(
    symbol: str,
    timeframe: str,
    limit: int = 2000,
    before: str | None = None,
) -> list[dict[str, Any]]:
    url = f"{api_base()}/prices/{symbol}/history/?timeframe={timeframe}&limit={limit}"
    if before:
        url += f"&before={quote(before)}"
    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.get(url)
        resp.raise_for_status()
        payload = resp.json()
    return payload.get("results") or []


def fetch_history(
    symbol: str = "xauusd",
    timeframe: str = "1m",
    max_pages: int = 12,
    limit: int = 2000,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    before: str | None = None
    for _ in range(max_pages):
        try:
            page = fetch_history_page(symbol, timeframe, limit=limit, before=before)
        except (httpx.HTTPError, json.JSONDecodeError):
            break
        if not page:
            break
        new = 0
        for item in page:
            iid = str(item.get("id"))
            if iid in seen:
                continue
            seen.add(iid)
            rows.append(item)
            new += 1
        if len(page) < limit or new == 0:
            break
        before = str(page[0].get("id"))
    return bars_from_records(rows)


def fetch_live_1m_bars(limit: int = 2000) -> pd.DataFrame:
    url = f"{bars_url()}?limit={limit}"
    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.get(url)
        resp.raise_for_status()
        payload = resp.json()
    bars = payload.get("bars") or []
    rows = []
    for bar in bars:
        if not bar or len(bar) < 5:
            continue
        rows.append(
            {
                "time": bar[0],
                "open": bar[1],
                "high": bar[2],
                "low": bar[3],
                "close": bar[4],
                "volume": bar[5] if len(bar) > 5 else 0.0,
            }
        )
    return bars_from_records(rows)


def load_cached_jsonl(timeframe: str) -> pd.DataFrame:
    names = {
        "1m": "xauusd_1m.jsonl",
        "1h": "xauusd_1h.jsonl",
        "1d": "xauusd.jsonl",
    }
    path = data_dir() / "forexcom" / names[timeframe]
    if not path.exists():
        return pd.DataFrame()
    return jsonl_to_frame(path)


def cache_forexcom_dump(src_1m: Path, src_1h: Path | None = None, src_1d: Path | None = None) -> dict[str, Path]:
    out_dir = data_dir() / "forexcom"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    df_1m = jsonl_to_frame(src_1m)
    p_1m = out_dir / "xauusd_1m.parquet"
    save_parquet(df_1m, p_1m)
    written["1m"] = p_1m
    if src_1h and Path(src_1h).exists():
        p = out_dir / "xauusd_1h.parquet"
        save_parquet(jsonl_to_frame(src_1h), p)
        written["1h"] = p
    if src_1d and Path(src_1d).exists():
        p = out_dir / "xauusd_1d.parquet"
        save_parquet(jsonl_to_frame(src_1d), p)
        written["1d"] = p
    return written
