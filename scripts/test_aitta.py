"""Quick connectivity + sanity test for the Aitta LLM endpoint.

Checks, in order:
  1. token presence
  2. /status service health
  3. online workers for the configured model
  4. a raw chat completion, printing finish_reason + usage to diagnose empty responses
  5. a structured() call against a tiny schema
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from pydantic import BaseModel

from agents.llm import AittaClient, AittaConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")


def main() -> None:
    cfg_path = Path("conf/llm.yaml")
    cfg = AittaConfig.from_yaml(cfg_path) if cfg_path.exists() else AittaConfig()
    print(f"[cfg] model={cfg.model} base_url={cfg.base_url} max_tokens={cfg.max_tokens}")

    tok = os.environ.get("AITTA_API_TOKEN", "")
    print(f"[token] present={bool(tok)} len={len(tok)}")

    client = AittaClient(cfg)

    print("\n[1] service status:")
    print("   ", client.service_status())

    print("\n[2] online workers (all):")
    print("   ", client.online_workers())
    print(f"[2b] is_model_online({cfg.model}):", client.is_model_online())

    print("\n[3] raw chat completion:")
    resp = client.client.chat.completions.create(
        model=cfg.model,
        messages=[{"role": "user", "content": "Reply with exactly: pong"}],
        temperature=0.0,
        max_tokens=cfg.max_tokens,
    )
    choice = resp.choices[0]
    content = choice.message.content
    print(f"    finish_reason = {choice.finish_reason}")
    print(f"    usage         = {resp.usage}")
    print(f"    content repr  = {content!r}")
    # gpt-oss reasoning channel sometimes lands here:
    reasoning = getattr(choice.message, "reasoning_content", None) or getattr(
        choice.message, "reasoning", None
    )
    if reasoning:
        print(f"    reasoning     = {reasoning[:300]!r}...")

    print("\n[4] structured() call:")

    class Ping(BaseModel):
        answer: str

    try:
        out = client.structured(
            system="You are a test.",
            user="Set answer to the string 'pong'.",
            schema=Ping,
            cache_key="aitta_selftest_ping",
        )
        print("    parsed:", out)
    except Exception as e:
        print(f"    structured() FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
