"""The Claude model switch: pricing, how requests are sent per model, and variant checks across models."""

import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from relentless import claude, config, multiply, serve


def usage(fresh, out, read=0, written=0):
    return SimpleNamespace(input_tokens=fresh, output_tokens=out, cache_read_input_tokens=read,
                           cache_creation_input_tokens=written)


def test_configured_model_is_priced_and_haiku_has_no_fallback():
    assert config.setting("NO_SUCH_SETTING", "x") == "x"
    assert "claude-haiku-5-5" in config.CLAUDE_PRICES
    assert config.CLAUDE_MODEL in config.CLAUDE_PRICES          # whatever is configured has a price
    assert "claude-haiku-5-5" not in config.CLAUDE_FALLBACK     # Haiku has no server-side fallback


def test_cost_per_model():
    label = usage(10_600, 2_000)
    assert claude.cost("claude-haiku-5-5", label) == pytest.approx(0.00206)
    assert claude.cost("claude-opus-5-5", label) == pytest.approx(0.0824)
    assert claude.cost("claude-opus-5-5", label) / claude.cost("claude-haiku-5-5", label) == pytest.approx(40)
    assert claude.cost("claude-haiku-5-5", usage(1_000, 100, read=10_000)) == pytest.approx(0.00025)


def test_haiku_long_prompts_cost_five_times_more():
    under, over = usage(100_000, 1_000), usage(100_001, 1_000)
    assert claude.cost("claude-haiku-5-5", under) == pytest.approx(0.0105)
    assert claude.cost("claude-haiku-5-5", over) == pytest.approx(5 * 0.0105, rel=1e-4)
    # cached parts of the prompt count towards the 100,000-token line too
    assert claude.cost("claude-haiku-5-5", usage(40_000, 0, read=70_000)) == pytest.approx(5 * (0.004 + 0.0007))
    assert claude.cost("claude-opus-5-5", over) == pytest.approx(0.420004)     # no such step on Opus


class Answer(BaseModel):
    ok: bool


class FakeClient:
    """Stands in for the Anthropic client and remembers how it was called."""

    def __init__(self, stop_reason="end_turn", parsed=None, category=None):
        self.calls = []
        self._resp = dict(stop_reason=stop_reason, parsed_output=parsed,
                          stop_details=SimpleNamespace(category=category) if category else None)
        self.messages = SimpleNamespace(parse=lambda **kw: self._reply("plain", kw))
        self.beta = SimpleNamespace(messages=SimpleNamespace(parse=lambda **kw: self._reply("beta", kw)))

    def _reply(self, route, kw):
        self.calls.append((route, kw))
        return SimpleNamespace(model=kw["model"], usage=usage(1_000, 100), **self._resp)


@pytest.fixture
def spend(monkeypatch):
    rows = []
    monkeypatch.setattr(claude.ledger, "record", lambda service, usd, **meta: rows.append((service, usd, meta)))
    return rows


def test_haiku_requests_carry_no_fallback(monkeypatch, spend):
    fake = FakeClient(parsed=Answer(ok=True))
    monkeypatch.setattr(claude, "client", lambda: fake)
    out, info = claude.ask_full([{"type": "text", "text": "hi"}], Answer, purpose="label:t", model="claude-haiku-5-5")
    route, kw = fake.calls[0]
    assert route == "plain" and "fallbacks" not in kw and "betas" not in kw
    assert kw["model"] == "claude-haiku-5-5" and kw["thinking"] == {"type": "adaptive"}
    assert kw["output_config"] == {"effort": "medium"} and kw["output_format"] is Answer
    assert out.ok and info["model"] == "claude-haiku-5-5" and info["usd"] == pytest.approx(0.00015)
    assert spend == [("anthropic", pytest.approx(0.00015),
                      {"purpose": "label:t", "model": "claude-haiku-5-5", "input_tokens": 1_000, "output_tokens": 100})]


def test_opus_requests_keep_the_fallback(monkeypatch, spend):
    fake = FakeClient(parsed=Answer(ok=True))
    monkeypatch.setattr(claude, "client", lambda: fake)
    claude.ask_full([{"type": "text", "text": "hi"}], Answer, purpose="x", model="claude-opus-5-5", system="be brief")
    route, kw = fake.calls[0]
    assert route == "beta" and kw["fallbacks"] == "default" and kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert kw["system"] == "be brief"


def test_a_declined_request_is_an_error_and_still_billed(monkeypatch, spend):
    fake = FakeClient(stop_reason="refusal", category="general_harms")
    monkeypatch.setattr(claude, "client", lambda: fake)
    with pytest.raises(RuntimeError, match="declined.*general_harms"):
        claude.ask_full([{"type": "text", "text": "hi"}], Answer, purpose="x", model="claude-haiku-5-5")
    assert len(spend) == 1


def test_unit_costs_come_from_the_ledger_for_the_configured_model(monkeypatch):
    rows = [{"service": "anthropic", "usd": 0.06, "purpose": "label:run:c000", "model": "claude-opus-5-5"},
            {"service": "anthropic", "usd": 0.002, "purpose": "label:run:c000", "model": "claude-haiku-5-5"},
            {"service": "anthropic", "usd": 0.004, "purpose": "label", "model": "claude-haiku-5-5"},
            {"service": "anthropic", "usd": 0.0004, "purpose": "screen:run", "model": "claude-haiku-5-5"},
            {"service": "fal", "usd": 0.11, "endpoint": "x"}]
    monkeypatch.setattr(serve.ledger, "rows", lambda: rows)
    monkeypatch.setattr(serve, "CLAUDE_MODEL", "claude-haiku-5-5")
    monkeypatch.setattr(serve.screen, "SCREEN_MODEL", None)
    costs = serve.claude_costs()
    assert costs["model"] == "claude-haiku-5-5"
    assert costs["label"] == pytest.approx(0.003) and costs["screen"] == pytest.approx(0.0004)
    assert costs["verify"] == pytest.approx((4_200 * 0.10 + 500 * 0.50) / 1e6)   # no history: estimated from tokens
    monkeypatch.setattr(serve, "CLAUDE_MODEL", "claude-opus-5-5")
    assert serve.claude_costs()["label"] == pytest.approx(0.06)


def verdict(score=7, **flags):
    base = dict(edit_applied=True, motion_preserved=True, hands_intact=True, faces_blurred=True,
                only_edit_changed=True, plausible=True, actions_seen=["picks up apple"], score=score, notes="ok")
    return multiply.Verdict(**{**base, **flags})


@pytest.fixture
def cached_variant(tmp_path, monkeypatch):
    """A variant generated and checked earlier by Opus, sitting in the cache folder."""
    out = tmp_path / "variant"
    out.mkdir()
    (out / "original.mp4").write_bytes(b"original")
    (out / "swap-apple-gemini.mp4").write_bytes(b"variant")
    (out / "swap-apple-gemini.json").write_text(json.dumps({
        "variant": "swap-apple", "prompt": "the prompt the video was made with", "file": "swap-apple-gemini.mp4",
        "cost_usd": 0.11, "labels": {}, "verify": verdict(score=8).model_dump(), "passed": True,
        "check_version": multiply.SWAP_VERSION}))
    checks = []

    def fake_verify(original, variant, edit, card, work):
        checks.append(edit.prompt)
        return verdict(score=5), multiply.CLAUDE_MODEL

    monkeypatch.setattr(multiply, "verify", fake_verify)
    monkeypatch.setattr(multiply, "run_edit", lambda *a, **k: pytest.fail("the cached video must not be regenerated"))
    monkeypatch.setattr(multiply.media, "probe", lambda p: {"seconds": 4.0, "width": 1280, "height": 720, "fps": 30.0})
    monkeypatch.setattr(multiply.media, "side_by_side", lambda paths, dst, **k: dst)
    monkeypatch.setattr(multiply.ledger, "spent", lambda service: 0.0)
    edit = multiply.presets(("cup", "apple"))["swap"]
    edit.name = "swap-apple"

    def run():
        return multiply.multiply(out / "upscaled.mp4", {"labels": []}, [edit], "gemini", start=0.0, seconds=4.0,
                                 crop=None, out_root=tmp_path, verify_variants=True, dry_run=False, force=False,
                                 out_dir=out)["variants"][0]
    return run, checks, out


def test_a_variant_checked_by_another_model_is_checked_again(cached_variant, monkeypatch):
    run, checks, out = cached_variant
    monkeypatch.setattr(multiply, "CLAUDE_MODEL", "claude-haiku-5-5")
    rec = run()
    assert checks == ["the prompt the video was made with"]      # judged against the instruction that made it
    assert rec["verify_model"] == "claude-haiku-5-5" and rec["passed"] is False and rec["cost_usd"] == 0
    assert rec["prompt"] == "the prompt the video was made with"
    assert json.loads((out / "swap-apple-gemini.json").read_text())["verify_model"] == "claude-haiku-5-5"
    again = run()
    assert len(checks) == 1 and again.get("cached") and again["passed"] is False   # now its own verdict is reused


def test_a_variant_checked_by_the_same_model_is_reused(cached_variant, monkeypatch):
    run, checks, _ = cached_variant
    monkeypatch.setattr(multiply, "CLAUDE_MODEL", "claude-opus-5-5")   # records without the field are Opus's
    rec = run()
    assert checks == [] and rec["cached"] and rec["passed"] is True


def test_a_variant_passed_by_an_older_check_is_checked_again(cached_variant, monkeypatch):
    """The check got stricter (it used to pass a gloved hand redrawn bare). A verdict from before that is
    not taken over: the video is kept, the verdict is not."""
    run, checks, out = cached_variant
    monkeypatch.setattr(multiply, "CLAUDE_MODEL", "claude-opus-5-5")
    record = json.loads((out / "swap-apple-gemini.json").read_text())
    del record["check_version"]
    (out / "swap-apple-gemini.json").write_text(json.dumps(record))
    rec = run()
    assert len(checks) == 1 and rec["passed"] is False and rec["cost_usd"] == 0      # checked again, not generated again
    assert json.loads((out / "swap-apple-gemini.json").read_text())["check_version"] == multiply.SWAP_VERSION
