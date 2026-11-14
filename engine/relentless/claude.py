"""Claude vision calls used by the engine (screening, labeling and the variant check)."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import TypeVar

import anthropic
from pydantic import BaseModel

from . import ledger
from .config import CLAUDE_FALLBACK, CLAUDE_LONG_PROMPT, CLAUDE_MODEL, CLAUDE_PRICES, secret

T = TypeVar("T", bound=BaseModel)

_client: anthropic.Anthropic | None = None


def client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        # Explicit key + base URL so shell-level ANTHROPIC_* proxy settings are ignored.
        _client = anthropic.Anthropic(api_key=secret("ANTHROPIC_API_KEY"), base_url="https://api.anthropic.com")
    return _client


def image_block(path: Path) -> dict:
    return jpeg_block(path.read_bytes())


def jpeg_block(data: bytes) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": base64.standard_b64encode(data).decode()}}


def cost(model: str, usage) -> float:
    """USD for one response, from the price table (falls back to the configured model's prices)."""
    p_in, p_out, p_read = CLAUDE_PRICES.get(model) or CLAUDE_PRICES.get(CLAUDE_MODEL) or (4.0, 20.0, 0.20)
    fresh = usage.input_tokens or 0
    read = getattr(usage, "cache_read_input_tokens", 0) or 0
    usd = (fresh * p_in + (usage.output_tokens or 0) * p_out + read * p_read) / 1e6
    limit, factor = CLAUDE_LONG_PROMPT.get(model, (None, 1.0))
    prompt = fresh + read + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
    return usd * factor if limit and prompt > limit else usd


def ask_full(content: list[dict], schema: type[T], *, purpose: str, effort: str = "medium",
             system: str | None = None, model: str | None = None) -> tuple[T, dict]:
    """One structured call. Records its cost in the spend ledger and returns (parsed output, usage info)."""
    model = model or CLAUDE_MODEL
    request = {"model": model, "max_tokens": 16000, "thinking": {"type": "adaptive"},
               "output_config": {"effort": effort}, "output_format": schema,
               "messages": [{"role": "user", "content": content}]}
    if system:
        request["system"] = system
    if model in CLAUDE_FALLBACK:
        # If the safety classifiers decline the request, the API re-runs it on another model.
        resp = client().beta.messages.parse(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **request)
    else:
        resp = client().messages.parse(**request)
    u = resp.usage
    usd = cost(resp.model, u)
    ledger.record("anthropic", usd, purpose=purpose, model=resp.model,
                  input_tokens=u.input_tokens, output_tokens=u.output_tokens)
    if resp.stop_reason == "refusal":
        category = getattr(getattr(resp, "stop_details", None), "category", None)
        raise RuntimeError(f"Claude declined this request (safety category: {category or 'not given'})")
    if resp.parsed_output is None:
        raise RuntimeError(f"Claude returned no usable output (stop_reason={resp.stop_reason})")
    return resp.parsed_output, {"usd": round(usd, 5), "model": resp.model,
                                "input_tokens": u.input_tokens, "output_tokens": u.output_tokens}


def ask(content: list[dict], schema: type[T], *, purpose: str, effort: str = "medium",
        system: str | None = None, model: str | None = None) -> T:
    return ask_full(content, schema, purpose=purpose, effort=effort, system=system, model=model)[0]
