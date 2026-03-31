from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Any, Callable, Dict, Optional, Protocol


Segment = Dict[str, Any]
ProgressCallback = Callable[[str, int, str], None]


class TaskCancelled(Exception):
    pass


@dataclass
class MistralProviderSettings:
    api_key: str = ""
    model: str = "voxtral-mini-latest"


@dataclass
class WhisperProviderSettings:
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "whisper-1"


@dataclass
class TranscriptionSettings:
    provider: str = "mistral"
    language_mode: str = "auto"
    language: str = ""
    timestamp_granularity: str = "segment"
    diarize: bool = False
    context_bias: str = ""
    thread_count: int = 3
    mistral: MistralProviderSettings = field(default_factory=MistralProviderSettings)
    whisper: WhisperProviderSettings = field(default_factory=WhisperProviderSettings)


@dataclass
class TranslationSettings:
    mode: str = "none"
    model: str = "mistral-small-latest"
    target_language: str = "zh"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    bilingual_srt: bool = True
    keep_original_srt: bool = False
    allow_subtitle_import: bool = True
    subtitle_translation_thread_count: int = 3


@dataclass
class OutputSettings:
    mode: str = "source"
    output_dir: Path = field(default_factory=lambda: Path.cwd() / "subtitles")
    save_srt: bool = True
    save_lrc: bool = True
    save_txt: bool = True
    save_json: bool = False


@dataclass
class VadSettings:
    enabled: bool = False
    min_speech_ms: int = 250
    min_silence_ms: int = 400
    speech_pad_ms: int = 200
    max_segment_seconds: int = 15 * 60
    threshold: float = 0.5


@dataclass
class AppSettings:
    transcription: TranscriptionSettings = field(default_factory=TranscriptionSettings)
    translation: TranslationSettings = field(default_factory=TranslationSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    vad: VadSettings = field(default_factory=VadSettings)


@dataclass
class TaskState:
    task_id: str
    source_path: Path
    row: int
    status: str = "Queued"
    progress: int = 0
    message: str = "就绪"
    outputs: Dict[str, str] = field(default_factory=dict)


@dataclass
class TranscriptionRequest:
    source_path: Path
    audio_path: Path
    language_mode: str
    language: str
    timestamp_granularity: str
    diarize: bool
    context_bias: str


@dataclass
class TranscriptionResult:
    text: str
    segments: list[Segment]
    language: str
    raw_payload: Dict[str, Any]


@dataclass
class TranslationRequest:
    model: str
    target_language: str


class TranscriptionProvider(Protocol):
    def transcribe(
        self,
        request: TranscriptionRequest,
        progress_cb: Optional[Callable[[str], None]],
        cancel_event: Event,
    ) -> TranscriptionResult:
        ...


class TranslationProvider(Protocol):
    def translate_lines(
        self,
        lines: list[str],
        request: TranslationRequest,
        cancel_event: Event,
        parallel_workers: int = 1,
    ) -> list[str]:
        ...

