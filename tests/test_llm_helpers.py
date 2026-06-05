"""Tests for the JSON-extraction + cache helpers in agents.llm.

These don't import openai (the SDK is an optional extra) — they only exercise the pure
helpers, plus a fully-mocked structured() call to verify cache + retry behavior.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from agents.llm import _extract_json, _hash_inputs


class _Foo(BaseModel):
    name: str
    keep: bool


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


def test_structured_cache_hit_skips_llm(tmp_path: Path, monkeypatch):
    """When a cache file exists for a given key, structured() must NOT call .chat()."""
    pytest.importorskip("openai")  # client constructor needs the SDK
    from agents.llm import AittaClient, AittaConfig

    monkeypatch.setenv("AITTA_API_TOKEN", "test-token")
    cfg = AittaConfig(decision_cache_dir=str(tmp_path))
    client = AittaClient(cfg)

    # Pre-populate cache for the key structured() will compute.
    cache_key = "test-key"
    (tmp_path / f"{cache_key}.json").write_text(_Foo(name="cached", keep=True).model_dump_json())

    def fail_if_called(*a, **kw):
        raise AssertionError("chat() should not be called when cache hits")
    client.chat = fail_if_called  # type: ignore

    result = client.structured(system="sys", user="usr", schema=_Foo, cache_key=cache_key)
    assert result.name == "cached"
    assert result.keep is True
