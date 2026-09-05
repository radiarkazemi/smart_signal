from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from smart_signal.config import artifacts_dir


def _log_path():
    return artifacts_dir() / "signal_log.jsonl"


def _latest_path():
    return artifacts_dir() / "latest_signal.json"


def append_signal(sig: dict[str, Any]) -> dict[str, Any]:
    record = dict(sig)
    record["logged_at"] = datetime.now(timezone.utc).isoformat()
    latest = _latest_path()
    prev = {}
    if latest.exists():
        try:
            prev = json.loads(latest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prev = {}
    changed = (
        prev.get("signal") != record.get("signal")
        or prev.get("as_of") != record.get("as_of")
        or abs(float(prev.get("price") or 0) - float(record.get("price") or 0)) > 0.05
    )
    if changed:
        with _log_path().open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    latest.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return record


def latest_signal() -> dict[str, Any] | None:
    path = _latest_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def signal_history(limit: int = 40) -> list[dict[str, Any]]:
    path = _log_path()
    if not path.exists():
        latest = latest_signal()
        return [latest] if latest else []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines()[-max(limit, 1) :]:
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    rows.reverse()
    return rows[:limit]
