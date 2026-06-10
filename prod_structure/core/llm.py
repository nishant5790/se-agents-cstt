"""Shared LLM client (Azure OpenAI) with a safe offline fallback.

If Azure credentials are not configured, `chat_json` returns None so callers can
fall back to deterministic rules. This keeps the whole team runnable air-gapped.

This module also loads the package `.env` and appends any local TLS certs to the
certifi bundle at import time, so every other module just sees a configured
environment.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import certifi
from dotenv import load_dotenv

# llm.py now lives in agent_team/core/ — resolve the package + repo roots
# explicitly so the .env / certs paths are unaffected by the move.
_PKG = Path(__file__).resolve().parents[1]   # agent_team/
_ROOT = _PKG.parent                          # repo root (SE Agent/)

load_dotenv(_PKG / ".env")  # package-local .env, regardless of cwd
load_dotenv()                # also honour a cwd/parent .env if present

# Look for TLS proxy certs in both the package and the legacy tetris_mvp folder.
_CERT_DIRS = [_PKG / "certs", _ROOT / "tetris_mvp" / "certs"]


def _ensure_tls_trust() -> None:
    """Append any certs/*.crt to certifi so HTTPS works behind a TLS proxy."""
    bundle = Path(certifi.where())
    try:
        text = bundle.read_text(encoding="utf-8", errors="ignore")
        for cert_dir in _CERT_DIRS:
            if not cert_dir.exists():
                continue
            for crt in sorted(cert_dir.glob("*.crt")):
                if f"# >>> {crt.name}" in text:
                    continue
                with bundle.open("a", encoding="utf-8") as f:
                    f.write(f"\n# >>> {crt.name}\n{crt.read_text(errors='ignore')}")
                text += f"# >>> {crt.name}"
    except PermissionError:
        pass
    os.environ.setdefault("SSL_CERT_FILE", str(bundle))
    os.environ.setdefault("REQUESTS_CA_BUNDLE", str(bundle))


def available() -> bool:
    return bool(os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_KEY"))


def azure_openai_client():
    """Return ``(client, deployment)`` or ``None`` if Azure is not configured."""
    if not available():
        return None
    _ensure_tls_trust()
    from openai import AzureOpenAI

    client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    )
    model = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    return client, model


def chat_json(system: str, user: str, *, temperature: float = 0.2) -> dict[str, Any] | None:
    """Call the LLM and parse a JSON object response. Returns None if unavailable."""
    if not available():
        return None
    _ensure_tls_trust()
    from openai import AzureOpenAI

    client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    )
    resp = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    content = resp.choices[0].message.content or "{}"
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


def chat_text(system: str, user: str, *, temperature: float = 0.3) -> str | None:
    """Call the LLM for a plain-text answer. Returns None if unavailable."""
    if not available():
        return None
    _ensure_tls_trust()
    from openai import AzureOpenAI

    client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    )
    resp = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return (resp.choices[0].message.content or "").strip() or None
