"""Tests for the JSON-extraction + cache helpers in agents.llm.

These don't import openai/anthropic by default (both are optional extras) — they only
exercise the pure helpers, plus fully-mocked structured() calls per provider to verify
the cache + retry behavior is shared correctly.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from agents.llm import LLMConfig, _extract_json, _hash_inputs


class _Foo(BaseModel):
    name: str
    keep: bool


# ---- pure helpers --------------------------------------------------------

def test_extract_json_strips_markdown_fence():
    raw = "```json\n{\"name\": \"x\", \"keep\": true}\n```"
    parsed = json.loads(_extract_json(raw))
    assert parsed == {"name": "x", "keep": True}


def test_extract_json_finds_first_object_in_prose():
    raw = 'Here is the JSON: {"name": "x", "keep": false} and some trailing chatter.'
    assert json.loads(_extract_json(raw))["keep"] is False


def test_extract_json_handles_nested_braces():
    raw = '{"name": "x", "meta": {"y": 1, "z": {"a": 2}}, "keep": true}'
    obj = json.loads(_extract_json(raw))
    assert obj["meta"]["z"]["a"] == 2


def test_hash_inputs_is_stable_and_distinct():
    h1 = _hash_inputs("a", {"k": 1})
    h2 = _hash_inputs("a", {"k": 1})
    h3 = _hash_inputs("a", {"k": 2})
    assert h1 == h2
    assert h1 != h3


# ---- config defaults -----------------------------------------------------

def test_llmconfig_defaults_per_provider():
    aitta = LLMConfig(provider="aitta")
    assert aitta.api_key_env == "AITTA_API_TOKEN"
    assert aitta.base_url.startswith("https://aitta-api.csc.fi")
    assert aitta.model.startswith("LumiOpen/")

    anth = LLMConfig(provider="anthropic")
    assert anth.api_key_env == "ANTHROPIC_API_KEY"
    assert anth.model.startswith("claude-")


# ---- structured() cache hits across both providers ----------------------

def _seed_cache(tmp_path: Path, key: str) -> None:
    (tmp_path / f"{key}.json").write_text(_Foo(name="cached", keep=True).model_dump_json())


def _assert_cache_hit_skips_llm(client, tmp_path, key):
    """Helper: pre-populate cache, mock _chat() so it raises, expect structured() to return cached."""
    _seed_cache(tmp_path, key)

    def fail_if_called(*a, **kw):
        raise AssertionError("_chat() should not be called when cache hits")
    client._chat = fail_if_called  # type: ignore[method-assign]

    result = client.structured(system="sys", user="usr", schema=_Foo, cache_key=key)
    assert result.name == "cached"
    assert result.keep is True


def test_aitta_structured_cache_hit_skips_llm(tmp_path: Path, monkeypatch):
    pytest.importorskip("openai")
    from agents.llm import AittaClient
    monkeypatch.setenv("AITTA_API_TOKEN", "test-token")
    cfg = LLMConfig(provider="aitta", decision_cache_dir=str(tmp_path))
    _assert_cache_hit_skips_llm(AittaClient(cfg), tmp_path, key="aitta-key")


def test_anthropic_structured_cache_hit_skips_llm(tmp_path: Path, monkeypatch):
    pytest.importorskip("anthropic")
    from agents.llm import AnthropicClient
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    cfg = LLMConfig(provider="anthropic", decision_cache_dir=str(tmp_path))
    _assert_cache_hit_skips_llm(AnthropicClient(cfg), tmp_path, key="anth-key")


# ---- factory routing -----------------------------------------------------

def test_build_client_returns_correct_subclass(monkeypatch):
    pytest.importorskip("openai")
    pytest.importorskip("anthropic")
    from agents.llm import AittaClient, AnthropicClient, build_client
    monkeypatch.setenv("AITTA_API_TOKEN", "t")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "t")

    assert isinstance(build_client(LLMConfig(provider="aitta")), AittaClient)
    assert isinstance(build_client(LLMConfig(provider="anthropic")), AnthropicClient)


def test_aitta_client_rejects_anthropic_config(monkeypatch):
    pytest.importorskip("openai")
    from agents.llm import AittaClient
    monkeypatch.setenv("AITTA_API_TOKEN", "t")
    with pytest.raises(ValueError, match="provider='aitta'"):
        AittaClient(LLMConfig(provider="anthropic"))


# ---- token loading --------------------------------------------------------

def test_load_token_fails_clearly_when_missing(monkeypatch):
    pytest.importorskip("anthropic")
    from agents.llm import AnthropicClient
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match=r"ANTHROPIC_API_KEY"):
        AnthropicClient(LLMConfig(provider="anthropic"))
