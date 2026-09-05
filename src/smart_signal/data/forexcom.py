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
TIMEOUT = 8.0


def api_base() -> str:
    return (env("SMART_SIGNAL_API_BASE") or DEFAULT_API_BASE).rstrip("/")


def bars_url() -> str:
    return env("SMART_SIGNAL_BARS_URL") or DEFAULT_BARS_URL


def _headers() -> dict[str, str]:
    key = env("SMART_SIGNAL_API_KEY")
    return {"X-API-Key": key} if key else {}


def fetch_last_price(symbol: str = "xauusd", timeframe: str = "1m") -> dict[str, Any]:
    url = f"{api_base()}/prices/{symbol}/?timeframe={timeframe}"
    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.get(url, headers=_headers())
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
        resp = client.get(url, headers=_headers())
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


def fetch_mongo_1m(limit: int = 4000) -> pd.DataFrame:
    uri = env("SMART_SIGNAL_MONGO_URI")
    if not uri:
        return pd.DataFrame()
    try:
        from pymongo import MongoClient
    except ImportError:
        return pd.DataFrame()
    client = MongoClient(uri, serverSelectionTimeoutMS=4000)
    docs = (
        client["historical_data"]["xauusd_1m"]
        .find({}, {"data": 1})
        .sort("_id", -1)
        .limit(limit)
    )
    rows = []
    for doc in docs:
        data = doc.get("data") or {}
        if data.get("close") is None:
            continue
        rows.append(
            {
                "time": data.get("time"),
                "open": data.get("open"),
                "high": data.get("high"),
                "low": data.get("low"),
                "close": data.get("close"),
                "volume": data.get("volume") or 0.0,
            }
        )
    return bars_from_records(rows)


def fetch_quote() -> dict[str, Any]:
    try:
        quote = fetch_last_price("xauusd", "1m")
        if quote.get("price") is not None:
            return quote
    except Exception:
        pass
    bars = fetch_live_1m_bars(limit=80)
    if bars.empty:
        mongo = fetch_mongo_1m(limit=5)
        if mongo.empty:
            raise RuntimeError("No live XAUUSD quote from API, bars, or Mongo")
        bars = mongo
    last = bars.iloc[-1]
    return {
        "symbol": "xauusd",
        "exchange": "forexcom",
        "timeframe": "1m",
        "price": float(last["close"]),
        "open": float(last["open"]),
        "high": float(last["high"]),
        "low": float(last["low"]),
        "volume": float(last.get("volume") or 0.0),
        "time": pd.Timestamp(last["time"]).isoformat(),
    }


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
