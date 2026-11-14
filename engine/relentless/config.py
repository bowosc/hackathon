"""Paths, keys and price tables shared by every engine stage."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values

REPO = Path(__file__).resolve().parents[2]
# Runs, uploads, the cache and the spend ledger live here. RELENTLESS_DATA moves them, for
# example onto a mounted volume when the engine runs on a server.
DATA = Path(os.environ.get("RELENTLESS_DATA") or REPO / "data")
RUNS = DATA / "runs"
MULTIPLY_OUT = DATA / "multiply"
LEDGER = DATA / "spend-ledger.jsonl"
MODELS = REPO / "engine" / "models"  # downloaded weights, git-ignored

# Read keys from the repo .env only. The shell may carry ANTHROPIC_* variables from
# other tools (proxies, CLIs) that must not leak into the engine.
_ENV = dotenv_values(REPO / ".env")


def secret(name: str) -> str:
    value = _ENV.get(name) or os.environ.get(f"RELENTLESS_{name}")
    if not value:
        raise SystemExit(f"Missing {name}: add it to {REPO / '.env'}")
    return value


def setting(name: str, default: str) -> str:
    """Non-secret override: RELENTLESS_<NAME> in the shell, then <NAME> in .env, then the default."""
    return os.environ.get(f"RELENTLESS_{name}") or _ENV.get(name) or default


# The Claude model behind screening, labeling and the variant check. Haiku 5.5 costs 1/40 of
# Opus 5.5 per token. To use another model, set CLAUDE_MODEL in .env (e.g. claude-opus-5-5).
CLAUDE_MODEL = setting("CLAUDE_MODEL", "claude-haiku-5-5")

# USD per 1M tokens (input, output, cache read), from Anthropic's price list on 2026-11-14.
CLAUDE_PRICES = {
    "claude-opus-5-5": (4.0, 20.0, 0.20),
    "claude-sonnet-5-5": (2.0, 10.0, 0.10),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
}
# Haiku 5.5 bills a request at 5x these prices when its prompt is over 100,000 tokens
# (model: (prompt tokens, factor)). Engine prompts are about 10,000 tokens.
CLAUDE_LONG_PROMPT = {"claude-haiku-5-5": (100_000, 5.0)}
# Models for which the API can re-run a request its safety classifiers decline on another
# model (server-side fallback). Haiku 5.5 has no such fallback.
CLAUDE_FALLBACK = {"claude-opus-5-5", "claude-sonnet-5-5"}

# fal.ai video edit models: USD per second of output video (from fal's pricing API, 2026-11-13).
FAL_MODELS = {
    "gemini": {"endpoint": "google/gemini-omni-flash/v1.2/edit", "usd_per_s": 0.0273, "max_s": 15},
    "flux": {"endpoint": "blackforestlabs/flux-3/edit-video", "usd_per_s": 0.03, "max_s": 15},
}

# Hard cap on total fal spend recorded in the ledger. Override with RELENTLESS_FAL_BUDGET_USD.
FAL_BUDGET_USD = float(setting("FAL_BUDGET_USD", "3.0"))

# Pose + person model for stages 2-3 (Ultralytics downloads it into MODELS on first use).
POSE_WEIGHTS = setting("POSE_WEIGHTS", "yolo26s-pose.pt")

# Upscaler weights (Real-ESRGAN, BSD-3-Clause), fetched into MODELS on first use.
UPSCALE_MODELS = {
    "general": {
        "file": "realesr-general-x4v3.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
    },
    "x4plus": {
        "file": "RealESRGAN_x4plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    },
}


_threads_capped = False


def device() -> str:
    """Best available torch device: Apple GPU, then CUDA, then CPU. DEVICE (e.g. cpu) overrides it.

    Every stage that uses torch asks for its device first, so this is also where THREADS is applied
    to torch, in whichever process loads it."""
    import torch

    global _threads_capped
    threads = int(setting("THREADS", "0"))
    if threads > 0 and not _threads_capped:
        torch.set_num_threads(threads)
        _threads_capped = True
    forced = setting("DEVICE", "")
    if forced:
        return forced
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def limit_threads() -> None:
    """THREADS caps the CPU threads the engine uses, so a shared server keeps room for its other work.
    OpenCV is capped here, at the start of a run. torch is capped in device(), when a stage first loads
    it, so that a process that never needs torch does not pay for importing it."""
    threads = int(setting("THREADS", "0"))
    if threads > 0:
        import cv2

        cv2.setNumThreads(threads)
