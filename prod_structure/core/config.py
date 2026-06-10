"""Centralised configuration.

All environment-driven settings are read here so the rest of the codebase never
calls ``os.getenv`` directly for behaviour toggles. :func:`settings` returns a
fresh :class:`Settings` snapshot each call, so tests that mutate the environment
before invoking a tool see the updated values.

The ``.env`` file (and TLS certs) are loaded by :mod:`agent_team.core.llm` at
import time; importing this module after that picks up those values.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def env_str(name: str, default: str) -> str:
    val = os.getenv(name)
    return val if val is not None and val != "" else default


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def env_flag(name: str, default: bool) -> bool:
    return os.getenv(name, "1" if default else "0").strip().lower() in {
        "1", "true", "yes", "on",
    }


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of runtime configuration."""

    # --- Azure OpenAI ---
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_api_version: str
    azure_openai_deployment: str          # chat + vision (gpt-4o)
    azure_whisper_deployment: str

    # --- Azure Document Intelligence (optional OCR) ---
    docintel_endpoint: str
    docintel_key: str

    # --- transcription ---
    transcribe_backend: str               # "azure" | "local"
    local_whisper_model: str
    local_whisper_chunk_secs: int
    azure_media_chunk_secs: int

    # --- media / video tool ---
    media_max_frames: int
    caption_frames: bool
    ocr_frames: bool
    video_chunk_secs: float
    video_frame_interval: float
    video_similarity_threshold: float
    video_analyze_frames: bool

    # --- understanding agent ---
    understanding_batch: int
    understanding_max_llm_blocks: int

    @property
    def azure_openai_configured(self) -> bool:
        return bool(self.azure_openai_endpoint and self.azure_openai_api_key)

    @property
    def docintel_configured(self) -> bool:
        return bool(self.docintel_endpoint and self.docintel_key)

    @property
    def transcribe_azure(self) -> bool:
        return self.transcribe_backend == "azure" and self.azure_openai_configured


def settings() -> Settings:
    """Build a fresh :class:`Settings` from the current environment."""
    return Settings(
        azure_openai_endpoint=env_str("AZURE_OPENAI_ENDPOINT", ""),
        azure_openai_api_key=env_str("AZURE_OPENAI_API_KEY", ""),
        azure_openai_api_version=env_str("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        azure_openai_deployment=env_str("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
        azure_whisper_deployment=env_str("AZURE_OPENAI_WHISPER_DEPLOYMENT", "whisper"),
        docintel_endpoint=env_str("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", ""),
        docintel_key=env_str("AZURE_DOCUMENT_INTELLIGENCE_KEY", ""),
        transcribe_backend=env_str("MEDIA_TRANSCRIBE_BACKEND", "azure").strip().lower(),
        local_whisper_model=env_str("LOCAL_WHISPER_MODEL", "base"),
        local_whisper_chunk_secs=env_int("LOCAL_WHISPER_CHUNK_SECS", 30),
        azure_media_chunk_secs=env_int("AZURE_MEDIA_CHUNK_SECS", 600),
        media_max_frames=env_int("MEDIA_MAX_FRAMES", 60),
        caption_frames=env_flag("AZURE_MEDIA_CAPTION_FRAMES", True),
        ocr_frames=env_flag("AZURE_MEDIA_OCR_FRAMES", True),
        video_chunk_secs=env_float("VIDEO_CHUNK_SECS", 90.0),
        video_frame_interval=env_float("VIDEO_FRAME_INTERVAL", 5.0),
        video_similarity_threshold=env_float("VIDEO_SIMILARITY_THRESHOLD", 0.95),
        video_analyze_frames=env_flag("VIDEO_ANALYZE_FRAMES", True),
        understanding_batch=env_int("UNDERSTANDING_BATCH", 25),
        understanding_max_llm_blocks=env_int("UNDERSTANDING_MAX_LLM_BLOCKS", 300),
    )
