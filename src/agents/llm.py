"""Aitta (CSC LUMI LLM inference) client + structured-output helper.

Aitta is OpenAI-API-compatible (https://aitta.csc.fi/page/docs), so we use the official
`openai` SDK with a custom `base_url`. Tokens come from `$AITTA_API_TOKEN` (or a path in
`$AITTA_API_TOKEN_FILE`).

The agents call `client.structured(...)` which:
- prompts the model to emit JSON conforming to a pydantic schema
- validates the parse and retries with the validation error message on failure
- backs off on 429 (Aitta enforces a ~1-minute cooldown)
- caches decisions on disk keyed by a hash of (model, system, user, schema) so re-running the
  planner is free — important because the SLURM dispatcher may be re-invoked many times while
  iterating on configs.

Why a cache instead of a per-call deterministic seed: Aitta tokens expire (24h default), and
LLM responses aren't bit-exact across model upgrades anyway. Caching keys on the inputs gives
us the only kind of reproducibility that actually matters: same inputs → same decision file.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Type, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# ---- Config ---------------------------------------------------------------

@dataclass
class AittaConfig:
    base_url: str = "https://aitta-api.csc.fi/openai/v1"
    model: str = "LumiOpen/Llama-Poro-2-70B-Instruct"
    temperature: float = 0.0
    max_tokens: int = 1024
    max_retries: int = 6
    backoff_initial_s: float = 2.0
    backoff_max_s: float = 90.0
    decision_cache_dir: str | None = "results/llm_cache"

    @classmethod
    def from_yaml(cls, path: Path) -> "AittaConfig":
        return cls(**yaml.safe_load(path.read_text()))


def _load_token() -> str:
    """Read the Aitta bearer token from env or file. Fail loudly if missing."""
    tok = os.environ.get("AITTA_API_TOKEN")
    if tok:
        return tok.strip()
    path = os.environ.get("AITTA_API_TOKEN_FILE")
    if path:
        return Path(path).read_text().strip()
    raise RuntimeError(
        "no Aitta token — set $AITTA_API_TOKEN (or $AITTA_API_TOKEN_FILE). "
        "Generate one at https://aitta-auth.csc.fi/myToken."
    )


# ---- Client ---------------------------------------------------------------

class AittaClient:
    """Wraps `openai.OpenAI` against the Aitta endpoint and adds structured calls + caching."""

    def __init__(self, cfg: AittaConfig | None = None, *, token: str | None = None):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai SDK not installed — add the `llm` extra: pip install -e '.[llm]'"
            ) from e
        self.cfg = cfg or AittaConfig()
        self.client = OpenAI(base_url=self.cfg.base_url, api_key=token or _load_token())
        self.cache_dir = Path(self.cfg.decision_cache_dir) if self.cfg.decision_cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ---- Raw chat with retries ----

    def chat(self, messages: list[dict], **kwargs) -> str:
        """Call chat/completions with 429/5xx retry. Returns the assistant message content."""
        attempt = 0
        last_err: Exception | None = None
        while attempt <= self.cfg.max_retries:
            try:
                resp = self.client.chat.completions.create(
                    model=kwargs.get("model", self.cfg.model),
                    messages=messages,
                    temperature=kwargs.get("temperature", self.cfg.temperature),
                    max_tokens=kwargs.get("max_tokens", self.cfg.max_tokens),
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                last_err = e
                if not _is_retryable(e):
                    raise
                wait = _backoff_seconds(e, attempt, self.cfg.backoff_initial_s,
                                        self.cfg.backoff_max_s)
                log.warning("Aitta call failed (attempt %d/%d, sleeping %.1fs): %s",
                            attempt + 1, self.cfg.max_retries + 1, wait, e)
                time.sleep(wait)
                attempt += 1
        raise RuntimeError(f"Aitta call failed after {self.cfg.max_retries + 1} attempts") \
            from last_err

    # ---- Structured ----

    def structured(
        self,
        *,
        system: str,
        user: str,
        schema: Type[T],
        cache_key: str | None = None,
    ) -> T:
        """Ask the model for JSON conforming to `schema`. Caches on (model, system, user, schema)."""
        key = cache_key or _hash_inputs(self.cfg.model, system, user, schema.model_json_schema())
        cache_path = (self.cache_dir / f"{key}.json") if self.cache_dir else None
        if cache_path is not None and cache_path.exists():
            log.info("LLM cache hit %s", cache_path.name)
            return schema.model_validate_json(cache_path.read_text())

        schema_str = json.dumps(schema.model_json_schema(), indent=2)
        sys_msg = (
            f"{system}\n\nReply with a single JSON object that validates against this schema. "
            "Do not wrap it in markdown fences, do not add explanation outside the JSON.\n\n"
            f"```json-schema\n{schema_str}\n```"
        )
        messages = [{"role": "system", "content": sys_msg},
                    {"role": "user", "content": user}]
        last_err: Exception | None = None
        for attempt in range(self.cfg.max_retries + 1):
            raw = self.chat(messages)
            try:
                obj = schema.model_validate_json(_extract_json(raw))
                if cache_path is not None:
                    cache_path.write_text(obj.model_dump_json(indent=2))
                return obj
            except (ValidationError, json.JSONDecodeError) as e:
                last_err = e
                log.warning("LLM JSON parse failed (attempt %d): %s", attempt + 1, e)
                # Re-prompt with the validation error and the previous response.
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": (f"The previous response failed validation: {e}. "
                                "Re-emit a single JSON object matching the schema, "
                                "with no other text."),
                })
        raise RuntimeError("LLM structured call could not produce schema-valid JSON") \
            from last_err


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
        # strip markdown fence; tolerate ```json
        text = text.strip("`")
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
    text = text.strip()
    # Find the first balanced { ... } block.
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
                "InternalServerError"):
        return True
    status = getattr(e, "status_code", None) or getattr(e, "http_status", None)
    return status in (408, 425, 429, 500, 502, 503, 504)


def _backoff_seconds(e: Exception, attempt: int, base: float, cap: float) -> float:
    """Aitta returns 429 with a ~60s cooldown — honor that when we see it."""
    status = getattr(e, "status_code", None) or getattr(e, "http_status", None)
    if status == 429:
        return min(cap, 60.0 + random.uniform(0, 5))
    return min(cap, base * (2 ** attempt) + random.uniform(0, 1))
