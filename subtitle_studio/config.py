from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict

from .constants import SETTINGS_FILE
from .models import (
    AppSettings,
    MistralProviderSettings,
    OutputSettings,
    TranscriptionSettings,
    TranslationSettings,
    VadSettings,
    WhisperProviderSettings,
)
from .resources import resource_path


def find_ffmpeg() -> str:
    bundled = resource_path("ffmpeg.exe")
    if bundled.exists():
        return str(bundled)
    custom = os.environ.get("FFMPEG_BINARY", "").strip()
    if custom and Path(custom).exists():
        return custom
    ffmpeg = shutil.which("ffmpeg")
    return ffmpeg or ""


def default_settings() -> AppSettings:
    return AppSettings(
        transcription=TranscriptionSettings(
            mistral=MistralProviderSettings(
                api_key=os.environ.get("MISTRAL_API_KEY", ""),
                model="voxtral-mini-latest",
            ),
            whisper=WhisperProviderSettings(
                base_url="https://api.openai.com/v1",
                api_key=os.environ.get("OPENAI_API_KEY", ""),
                model="whisper-1",
            ),
        ),
        translation=TranslationSettings(
            openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        ),
        output=OutputSettings(
            output_dir=Path.cwd() / "subtitles",
            ffmpeg_path=find_ffmpeg(),
        ),
        vad=VadSettings(),
    )


def settings_file_path() -> Path:
    return Path.cwd() / SETTINGS_FILE


def _safe_int(value: Any, fallback: int, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        result = int(value)
    except Exception:
        result = fallback
    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)
    return result


def _safe_bool(value: Any, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    return fallback


def _safe_str(value: Any, fallback: str) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return fallback
    return str(value)


def _safe_float(value: Any, fallback: float, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        result = float(value)
    except Exception:
        result = fallback
    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)
    return result


def resolve_ffmpeg_path(preferred: str) -> str:
    candidate = preferred.strip()
    if candidate:
        resolved = shutil.which(candidate)
        return resolved or candidate
    return find_ffmpeg()


def has_ffmpeg(path_or_command: str) -> bool:
    candidate = path_or_command.strip()
    if not candidate:
        return False
    return Path(candidate).exists() or shutil.which(candidate) is not None


def serialize_settings(settings: AppSettings) -> Dict[str, Any]:
    return {
        "transcription_provider": settings.transcription.provider,
        "api_key": settings.transcription.mistral.api_key,
        "model": settings.transcription.mistral.model,
        "whisper_base_url": settings.transcription.whisper.base_url,
        "whisper_api_key": settings.transcription.whisper.api_key,
        "whisper_model": settings.transcription.whisper.model,
        "language_mode": settings.transcription.language_mode,
        "language": settings.transcription.language,
        "timestamp": settings.transcription.timestamp_granularity,
        "diarize": settings.transcription.diarize,
        "context_bias": settings.transcription.context_bias,
        "thread_count": settings.transcription.thread_count,
        "translation_mode": settings.translation.mode,
        "translation_target": settings.translation.target_language,
        "translation_model": settings.translation.model,
        "translation_bilingual": settings.translation.bilingual_srt,
        "translation_keep_original_srt": settings.translation.keep_original_srt,
        "allow_subtitle_import": settings.translation.allow_subtitle_import,
        "subtitle_translation_thread_count": settings.translation.subtitle_translation_thread_count,
        "translation_openai_base": settings.translation.openai_base_url,
        "translation_openai_key": settings.translation.openai_api_key,
        "output_mode": settings.output.mode,
        "output_dir": str(settings.output.output_dir),
        "save_srt": settings.output.save_srt,
        "save_lrc": settings.output.save_lrc,
        "save_txt": settings.output.save_txt,
        "save_json": settings.output.save_json,
        "ffmpeg_path": settings.output.ffmpeg_path,
        "silero_vad_enabled": settings.vad.enabled,
        "vad_min_speech_ms": settings.vad.min_speech_ms,
        "vad_min_silence_ms": settings.vad.min_silence_ms,
        "vad_speech_pad_ms": settings.vad.speech_pad_ms,
        "vad_max_segment_seconds": settings.vad.max_segment_seconds,
        "vad_threshold": settings.vad.threshold,
    }


def deserialize_settings(data: Dict[str, Any]) -> AppSettings:
    settings = default_settings()

    settings.transcription.provider = _safe_str(
        data.get("transcription_provider", data.get("provider", settings.transcription.provider)),
        settings.transcription.provider,
    )
    settings.transcription.mistral.api_key = _safe_str(data.get("api_key"), settings.transcription.mistral.api_key)
    settings.transcription.mistral.model = _safe_str(data.get("model"), settings.transcription.mistral.model)
    settings.transcription.whisper.base_url = _safe_str(
        data.get("whisper_base_url"),
        settings.transcription.whisper.base_url,
    )
    settings.transcription.whisper.api_key = _safe_str(
        data.get("whisper_api_key"),
        settings.transcription.whisper.api_key,
    )
    settings.transcription.whisper.model = _safe_str(
        data.get("whisper_model"),
        settings.transcription.whisper.model,
    )
    settings.transcription.language_mode = _safe_str(
        data.get("language_mode", "manual" if data.get("language_mode_index") == 1 else settings.transcription.language_mode),
        settings.transcription.language_mode,
    )
    settings.transcription.language = _safe_str(data.get("language"), settings.transcription.language)
    settings.transcription.timestamp_granularity = _safe_str(
        data.get("timestamp"),
        settings.transcription.timestamp_granularity,
    )
    settings.transcription.diarize = _safe_bool(data.get("diarize"), settings.transcription.diarize)
    settings.transcription.context_bias = _safe_str(data.get("context_bias"), settings.transcription.context_bias)
    settings.transcription.thread_count = _safe_int(
        data.get("thread_count"),
        settings.transcription.thread_count,
        1,
        16,
    )

    translation_mode_index = _safe_int(data.get("translation_mode_index"), -1)
    settings.translation.mode = _safe_str(
        data.get(
            "translation_mode",
            {0: "none", 1: "mistral", 2: "openai"}.get(translation_mode_index, settings.translation.mode),
        ),
        settings.translation.mode,
    )
    settings.translation.target_language = _safe_str(
        data.get("translation_target"),
        settings.translation.target_language,
    )
    settings.translation.model = _safe_str(data.get("translation_model"), settings.translation.model)
    settings.translation.bilingual_srt = _safe_bool(
        data.get("translation_bilingual"),
        settings.translation.bilingual_srt,
    )
    settings.translation.keep_original_srt = _safe_bool(
        data.get("translation_keep_original_srt"),
        settings.translation.keep_original_srt,
    )
    settings.translation.allow_subtitle_import = _safe_bool(
        data.get("allow_subtitle_import"),
        settings.translation.allow_subtitle_import,
    )
    settings.translation.subtitle_translation_thread_count = _safe_int(
        data.get("subtitle_translation_thread_count"),
        settings.translation.subtitle_translation_thread_count,
        1,
        16,
    )
    settings.translation.openai_base_url = _safe_str(
        data.get("translation_openai_base"),
        settings.translation.openai_base_url,
    )
    settings.translation.openai_api_key = _safe_str(
        data.get("translation_openai_key"),
        settings.translation.openai_api_key,
    )

    settings.output.mode = _safe_str(
        data.get("output_mode", "custom" if data.get("output_mode_index") == 1 else settings.output.mode),
        settings.output.mode,
    )
    settings.output.output_dir = Path(_safe_str(data.get("output_dir"), str(settings.output.output_dir)))
    settings.output.save_srt = _safe_bool(data.get("save_srt"), settings.output.save_srt)
    settings.output.save_lrc = _safe_bool(data.get("save_lrc"), settings.output.save_lrc)
    settings.output.save_txt = _safe_bool(data.get("save_txt"), settings.output.save_txt)
    settings.output.save_json = _safe_bool(data.get("save_json"), settings.output.save_json)
    settings.output.ffmpeg_path = resolve_ffmpeg_path(_safe_str(data.get("ffmpeg_path"), settings.output.ffmpeg_path))

    settings.vad.enabled = _safe_bool(data.get("silero_vad_enabled"), settings.vad.enabled)
    settings.vad.min_speech_ms = _safe_int(
        data.get("vad_min_speech_ms"),
        settings.vad.min_speech_ms,
        1,
        60_000,
    )
    settings.vad.min_silence_ms = _safe_int(
        data.get("vad_min_silence_ms"),
        settings.vad.min_silence_ms,
        1,
        60_000,
    )
    settings.vad.speech_pad_ms = _safe_int(
        data.get("vad_speech_pad_ms"),
        settings.vad.speech_pad_ms,
        0,
        60_000,
    )
    settings.vad.max_segment_seconds = _safe_int(
        data.get("vad_max_segment_seconds"),
        settings.vad.max_segment_seconds,
        1,
        24 * 3600,
    )
    settings.vad.threshold = _safe_float(
        data.get("vad_threshold"),
        settings.vad.threshold,
        0.0,
        1.0,
    )
    return settings


def load_settings() -> AppSettings:
    path = settings_file_path()
    if not path.exists():
        return default_settings()
    try:
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except Exception:
        return default_settings()
    if not isinstance(payload, dict):
        return default_settings()
    return deserialize_settings(payload)


def save_settings(settings: AppSettings) -> Path:
    path = settings_file_path()
    path.write_text(json.dumps(serialize_settings(settings), ensure_ascii=False, indent=2), encoding="utf-8")
    return path
