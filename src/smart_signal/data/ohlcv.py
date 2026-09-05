from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable

import pandas as pd

OHLCV_COLS = ["time", "open", "high", "low", "close", "volume"]

TF_RULES = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1D",
    "1w": "W-MON",
}


def _to_utc(ts: Any) -> pd.Timestamp:
    if isinstance(ts, pd.Timestamp):
        t = ts
    elif isinstance(ts, datetime):
        t = pd.Timestamp(ts)
    elif isinstance(ts, (int, float)):
        value = float(ts)
        if value > 1e12:
            value /= 1000.0
        t = pd.Timestamp(value, unit="s", tz="UTC")
    else:
        t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t


def bars_from_records(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    records = []
    for row in rows:
        time_val = row.get("time", row.get("t", row.get("id")))
        if time_val is None:
            continue
        try:
            rec = {
                "time": _to_utc(time_val),
                "open": float(row["open"] if "open" in row else row["o"]),
                "high": float(row["high"] if "high" in row else row["h"]),
                "low": float(row["low"] if "low" in row else row["l"]),
                "close": float(row["close"] if "close" in row else row["c"]),
                "volume": float(row.get("volume", row.get("v", 0.0)) or 0.0),
            }
        except (TypeError, ValueError, KeyError):
            continue
        if rec["high"] < rec["low"]:
            rec["high"], rec["low"] = rec["low"], rec["high"]
        records.append(rec)
    if not records:
        return pd.DataFrame(columns=OHLCV_COLS)
    df = pd.DataFrame.from_records(records)
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.sort_values("time").drop_duplicates("time", keep="last")
    return df.reset_index(drop=True)


def bars_from_yahoo_chart(payload: dict[str, Any]) -> pd.DataFrame:
    result = (payload.get("chart") or {}).get("result") or []
    if not result:
        # simplified cache: {"ts": [...], "quote": {...}}
        if "ts" in payload and "quote" in payload:
            ts = payload["ts"]
            quote = payload["quote"]
        else:
            return pd.DataFrame(columns=OHLCV_COLS)
    else:
        block = result[0]
        ts = block.get("timestamp") or []
        quote = ((block.get("indicators") or {}).get("quote") or [{}])[0]
    rows = []
    opens = quote.get("open") or []
    highs = quote.get("high") or []
    lows = quote.get("low") or []
    closes = quote.get("close") or []
    vols = quote.get("volume") or [0.0] * len(ts)
    for i, t in enumerate(ts):
        o, h, l, c = (
            _num(opens, i),
            _num(highs, i),
            _num(lows, i),
            _num(closes, i),
        )
        if None in (o, h, l, c):
            continue
        rows.append(
            {
                "time": t,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": _num(vols, i) or 0.0,
            }
        )
    return bars_from_records(rows)


def _num(seq: list[Any], i: int) -> float | None:
    if i >= len(seq) or seq[i] is None:
        return None
    try:
        return float(seq[i])
    except (TypeError, ValueError):
        return None


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    rule = TF_RULES[timeframe]
    indexed = df.set_index("time").sort_index()
    out = indexed.resample(rule, label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    out = out.dropna(subset=["open", "high", "low", "close"]).reset_index()
    return out


def jsonl_to_frame(path) -> pd.DataFrame:
    import json
    from pathlib import Path

    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return bars_from_records(rows)


def save_parquet(df: pd.DataFrame, path) -> None:
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_parquet(path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    return df
