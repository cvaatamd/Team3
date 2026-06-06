#!/usr/bin/env python
"""End-to-end smoke test for Aitta (CSC LUMI LLM inference).

Unlike tests/test_llm_helpers.py (which is fully mocked), this actually hits the live
Aitta API to verify that the token, endpoint, and model are working end to end. It checks
three things against `openai/gpt-oss-120b`:

  1. the model is listed/reachable        -> GET /openai/v1/models via the SDK
  2. a raw chat completion round-trips     -> AittaClient.chat(...)
  3. structured (schema-validated) output  -> AittaClient.structured(...)

Token resolution (handled by agents.llm):
  - $AITTA_API_TOKEN            the bearer token directly, or
  - $AITTA_API_TOKEN_FILE       a path to a file containing the token, or
  - --api-key on the command line (sets $AITTA_API_TOKEN for this process)
Generate a token at https://aitta-auth.csc.fi/myToken.

Usage:
  export AITTA_API_TOKEN=...    # 24h token from aitta-auth.csc.fi/myToken
  python scripts/smoke_aitta.py
  python scripts/smoke_aitta.py --api-key "$AITTA_API_TOKEN" --model openai/gpt-oss-120b
  python scripts/smoke_aitta.py --no-cache      # bypass the on-disk decision cache

Exit code is 0 only if all checks pass, so this is safe to use in CI / job scripts.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from pydantic import BaseModel, Field


class Ping(BaseModel):
    """Minimal schema for the structured-output check."""
    answer: str = Field(description="A short greeting.")
    ok: bool = Field(description="Always true.")


def _ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def main() -> int:
    p = argparse.ArgumentParser(description="Aitta end-to-end smoke test.")
    p.add_argument("--model", default="openai/gpt-oss-120b",
                   help="Model id to test (default: openai/gpt-oss-120b).")
    p.add_argument("--api-key", default=None,
                   help="Aitta bearer token. Falls back to $AITTA_API_TOKEN / $AITTA_API_TOKEN_FILE.")
    p.add_argument("--no-cache", action="store_true",
                   help="Disable the on-disk decision cache for the structured check.")
    args = p.parse_args()

    if args.api_key:
        os.environ["AITTA_API_TOKEN"] = args.api_key

    # Imported here so --help works even without the `openai` extra installed.
    from agents.llm import AittaClient, AittaConfig

    cfg = AittaConfig(model=args.model,
                      decision_cache_dir=None if args.no_cache else "results/llm_cache")

    print(f"Aitta smoke test")
    print(f"  base_url : {cfg.base_url}")
    print(f"  model    : {cfg.model}")
    print("-" * 60)

    failures = 0

    # --- Build the client (validates token + SDK presence) ---
    try:
        client = AittaClient(cfg)
        _ok("client constructed (token + openai SDK present)")
    except Exception as e:
        _fail(f"could not construct client: {e}")
        return 2

    # --- Check 1: model is reachable in the catalog ---
    try:
        ids = [m.id for m in client.client.models.list().data]
        if cfg.model in ids:
            _ok(f"model '{cfg.model}' is listed by the API")
        else:
            _ok(f"models endpoint reachable ({len(ids)} models); "
                f"'{cfg.model}' not in catalog but will still be attempted")
    except Exception as e:
        _fail(f"GET /models failed: {e}")
        failures += 1

    # --- Check 2: raw chat completion ---
    try:
        t0 = time.time()
        reply = client.chat(
            [{"role": "user", "content": "Reply with exactly the word: pong"}],
            max_tokens=16,
        )
        dt = time.time() - t0
        if reply.strip():
            _ok(f"chat() returned in {dt:.1f}s: {reply.strip()!r}")
        else:
            _fail("chat() returned an empty response")
            failures += 1
    except Exception as e:
        _fail(f"chat() raised: {e}")
        failures += 1

    # --- Check 3: structured output ---
    try:
        t0 = time.time()
        out = client.structured(
            system="Reply tersely.",
            user="Say hello.",
            schema=Ping,
        )
        dt = time.time() - t0
        _ok(f"structured() returned valid {Ping.__name__} in {dt:.1f}s: {out.model_dump()}")
    except Exception as e:
        _fail(f"structured() raised: {e}")
        failures += 1

    print("-" * 60)
    if failures:
        print(f"SMOKE TEST FAILED: {failures} check(s) failed.")
        return 1
    print("SMOKE TEST PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
