"""Append-only spend ledger so every paid API call is accounted for."""

from __future__ import annotations

import json
import time

from .config import LEDGER


def record(service: str, usd: float, **meta) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "service": service,
                            "usd": round(usd, 5), **meta}) + "\n")


def rows() -> list[dict]:
    if not LEDGER.exists():
        return []
    return [json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()]


def spent(service: str) -> float:
    return sum(row["usd"] for row in rows() if row["service"] == service)
