"""LLM client + structured-output helper.

Two providers, same `client.structured(...)` shape:

- **aitta** — CSC LUMI inference (OpenAI-compatible). Token from `$AITTA_API_TOKEN`.
- **anthropic** — Anthropic API. Key from `$ANTHROPIC_API_KEY`. Useful for local dev when
  you don't have an Aitta token, or for a higher-quality model on the planner narrative.

Both providers share:
- a disk cache keyed on `sha256(provider, model, system, user, schema)` so re-runs are free
- a JSON-schema-validated structured-output loop with parse-fail retries
- 429-aware retries (Aitta enforces a ~60s cooldown; Anthropic honors `retry-after` via the SDK)

The base class is `LLMClient`. Both subclasses implement `_chat(system, messages)`; the
shared `.structured(...)` does the cache + retry + parse work.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Type, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
Provider = Literal["aitta", "anthropic"]

_DEFAULT_KEY_ENV: dict[str, str] = {
    "aitta": "AITTA_API_TOKEN",
    "anthropic": "ANTHROPIC_API_KEY",
}
_DEFAULT_KEY_FILE_ENV: dict[str, str | None] = {
    "aitta": "AITTA_API_TOKEN_FILE",
    "anthropic": None,
}
_DEFAULT_MODEL: dict[str, str] = {
    "aitta": "LumiOpen/Llama-Poro-2-70B-Instruct",
    "anthropic": "claude-sonnet-4-5",
}
_AITTA_BASE_URL = "https://aitta-api.csc.fi/openai/v1"


# ---- Config ---------------------------------------------------------------

@dataclass
class LLMConfig:
    provider: Provider = "aitta"
    model: str | None = None              # default per-provider if None
    base_url: str | None = None           # aitta only; anthropic ignores
    api_key_env: str | None = None        # default per-provider if None
    temperature: float = 0.0
    max_tokens: int = 1024
    max_retries: int = 6
    backoff_initial_s: float = 2.0
    backoff_max_s: float = 90.0
    decision_cache_dir: str | None = "results/llm_cache"

    def __post_init__(self):
        if self.model is None:
            self.model = _DEFAULT_MODEL[self.provider]
        if self.api_key_env is None:
            self.api_key_env = _DEFAULT_KEY_ENV[self.provider]
        if self.provider == "aitta" and self.base_url is None:
            self.base_url = _AITTA_BASE_URL

    @classmethod
    def from_yaml(cls, path: Path) -> "LLMConfig":
        return cls(**yaml.safe_load(path.read_text()))


# Backward-compat alias — older code constructs `AittaConfig`.
AittaConfig = LLMConfig


def _load_token(env_var: str, file_env_var: str | None = None) -> str:
    tok = os.environ.get(env_var)
    if tok:
        return tok.strip()
    if file_env_var:
        path = os.environ.get(file_env_var)
        if path:
            return Path(path).read_text().strip()
    extra = f" (or ${file_env_var})" if file_env_var else ""
    raise RuntimeError(f"no LLM token — set ${env_var}{extra}.")


# ---- Base client (shared structured() + cache + retry) -------------------

class LLMClient:
    """Base class. Subclasses implement `_chat(system, messages) -> str`."""

    def __init__(self, cfg: LLMConfig, *, token: str | None = None):
        self.cfg = cfg
        if token is None:
            token = _load_token(cfg.api_key_env, _DEFAULT_KEY_FILE_ENV.get(cfg.provider))
        self._token = token
        self.cache_dir = Path(cfg.decision_cache_dir) if cfg.decision_cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # Subclass hook.
    def _chat(self, system: str, messages: list[dict]) -> str:
        raise NotImplementedError

    def chat(self, system: str, messages: list[dict]) -> str:
        """Provider-agnostic chat call with 429/transient retry."""
        attempt = 0
        last_err: Exception | None = None
        while attempt <= self.cfg.max_retries:
            try:
                return self._chat(system, messages)
            except Exception as e:
                last_err = e
                if not _is_retryable(e):
                    raise
                wait = _backoff_seconds(e, attempt, self.cfg.backoff_initial_s,
                                        self.cfg.backoff_max_s)
                log.warning("LLM call failed (attempt %d/%d, sleeping %.1fs): %s",
                            attempt + 1, self.cfg.max_retries + 1, wait, e)
                time.sleep(wait)
                attempt += 1
        raise RuntimeError(
            f"LLM call failed after {self.cfg.max_retries + 1} attempts"
        ) from last_err

    def structured(
        self,
        *,
        system: str,
        user: str,
        schema: Type[T],
        cache_key: str | None = None,
    ) -> T:
        """Ask the model for JSON conforming to `schema`. Caches on the inputs."""
        key = cache_key or _hash_inputs(
            self.cfg.provider, self.cfg.model, system, user, schema.model_json_schema()
        )
        cache_path = (self.cache_dir / f"{key}.json") if self.cache_dir else None
        if cache_path is not None and cache_path.exists():
            log.info("LLM cache hit %s", cache_path.name)
            return schema.model_validate_json(cache_path.read_text())

        schema_str = json.dumps(schema.model_json_schema(), indent=2)
        full_system = (
            f"{system}\n\nReply with a single JSON object that validates against this schema. "
            "Do not wrap it in markdown fences, do not add explanation outside the JSON.\n\n"
            f"```json-schema\n{schema_str}\n```"
        )
        messages: list[dict] = [{"role": "user", "content": user}]
        last_err: Exception | None = None
        for attempt in range(self.cfg.max_retries + 1):
            raw = self.chat(full_system, messages)
            try:
                obj = schema.model_validate_json(_extract_json(raw))
                if cache_path is not None:
                    cache_path.write_text(obj.model_dump_json(indent=2))
                return obj
            except (ValidationError, json.JSONDecodeError) as e:
                last_err = e
                log.warning("LLM JSON parse failed (attempt %d): %s", attempt + 1, e)
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": (f"The previous response failed validation: {e}. "
                                "Re-emit a single JSON object matching the schema, "
                                "with no other text."),
                })
        raise RuntimeError(
            "LLM structured call could not produce schema-valid JSON"
        ) from last_err


# ---- Aitta (OpenAI-compatible) -------------------------------------------

class AittaClient(LLMClient):
    def __init__(self, cfg: LLMConfig | None = None, *, token: str | None = None):
        cfg = cfg or LLMConfig(provider="aitta")
        if cfg.provider != "aitta":
            raise ValueError(f"AittaClient requires provider='aitta', got {cfg.provider!r}")
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai SDK not installed — add the `llm` extra: pip install -e '.[llm]'"
            ) from e
        super().__init__(cfg, token=token)
        self.client = OpenAI(base_url=cfg.base_url, api_key=self._token)

    def _chat(self, system: str, messages: list[dict]) -> str:
        full = [{"role": "system", "content": system}] + messages
        resp = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=full,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )
        return resp.choices[0].message.content or ""


# ---- Anthropic ------------------------------------------------------------

class AnthropicClient(LLMClient):
    def __init__(self, cfg: LLMConfig | None = None, *, token: str | None = None):
        cfg = cfg or LLMConfig(provider="anthropic")
        if cfg.provider != "anthropic":
            raise ValueError(
                f"AnthropicClient requires provider='anthropic', got {cfg.provider!r}"
            )
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic SDK not installed — add the `llm` extra: pip install -e '.[llm]'"
            ) from e
        super().__init__(cfg, token=token)
        self.client = Anthropic(api_key=self._token)

    def _chat(self, system: str, messages: list[dict]) -> str:
        resp = self.client.messages.create(
            model=self.cfg.model,
            system=system,
            messages=messages,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )
        # response.content is a list of content blocks; take all text blocks
        if not resp.content:
            return ""
        return "".join(getattr(b, "text", "") for b in resp.content)


# ---- Factory --------------------------------------------------------------

def build_client(cfg: LLMConfig, *, token: str | None = None) -> LLMClient:
    """Pick a client subclass from `cfg.provider`."""
    if cfg.provider == "aitta":
        return AittaClient(cfg, token=token)
    if cfg.provider == "anthropic":
        return AnthropicClient(cfg, token=token)
    raise ValueError(f"unknown LLM provider: {cfg.provider!r}")


# ---- helpers --------------------------------------------------------------

def _hash_inputs(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def _extract_json(text: str) -> str:
    """Pull the first top-level JSON object out of a model response, tolerating prose/fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
    text = text.strip()
    start = text.find("{")
    if start < 0:
        raise json.JSONDecodeError("no '{' in response", text, 0)
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    raise json.JSONDecodeError("unbalanced JSON", text, start)


def _is_retryable(e: Exception) -> bool:
    name = type(e).__name__
    if name in ("RateLimitError", "APIConnectionError", "APITimeoutError",
                "InternalServerError", "APIStatusError"):
        return True
    status = getattr(e, "status_code", None) or getattr(e, "http_status", None)
    return status in (408, 425, 429, 500, 502, 503, 504)


def _backoff_seconds(e: Exception, attempt: int, base: float, cap: float) -> float:
    """Honor `retry-after` if present; otherwise exponential backoff with jitter."""
    # Anthropic SDK sets `.response.headers["retry-after"]`; Aitta returns 429 with ~60s.
    retry_after = None
    resp = getattr(e, "response", None)
    if resp is not None:
        try:
            retry_after = float(resp.headers.get("retry-after", ""))
        except (AttributeError, TypeError, ValueError):
            retry_after = None
    if retry_after is not None:
        return min(cap, retry_after + random.uniform(0, 2))

    status = getattr(e, "status_code", None) or getattr(e, "http_status", None)
    if status == 429:
        return min(cap, 60.0 + random.uniform(0, 5))
    return min(cap, base * (2 ** attempt) + random.uniform(0, 1))
