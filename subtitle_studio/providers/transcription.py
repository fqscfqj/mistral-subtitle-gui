from __future__ import annotations

from pathlib import Path
from threading import Event
from typing import Any, Callable, Dict, Optional

from ..http_client import HttpClient
from ..models import (
    MistralProviderSettings,
    TranscriptionProvider,
    TranscriptionRequest,
    TranscriptionResult,
    WhisperProviderSettings,
)
from ..utils import detect_language_code, extract_segments, extract_text, normalize_response

try:
    from mistralai import Mistral
    _MISTRAL_IMPORT_ERROR: Exception | None = None
except Exception:
    try:
        # mistralai>=2 exposes the SDK entrypoint from mistralai.client.
        from mistralai.client import Mistral
        _MISTRAL_IMPORT_ERROR = None
    except Exception as exc:
        Mistral = None
        _MISTRAL_IMPORT_ERROR = exc


def normalize_audio_transcriptions_url(base_url: str) -> str:
    url = base_url.strip()
    if not url:
        return "https://api.openai.com/v1/audio/transcriptions"
    if url.endswith("/audio/transcriptions"):
        return url
    url = url.rstrip("/")
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return f"{url}/audio/transcriptions"


class MistralTranscriptionProvider(TranscriptionProvider):
    def __init__(self, settings: MistralProviderSettings) -> None:
        self.settings = settings

    def transcribe(
        self,
        request: TranscriptionRequest,
        progress_cb: Optional[Callable[[str], None]],
        cancel_event: Event,
    ) -> TranscriptionResult:
        if cancel_event.is_set():
            raise RuntimeError("转写前已取消")
        if Mistral is None:
            details = ""
            if _MISTRAL_IMPORT_ERROR is not None:
                details = f"（导入错误：{type(_MISTRAL_IMPORT_ERROR).__name__}: {_MISTRAL_IMPORT_ERROR}）"
            raise RuntimeError(f"缺少依赖：mistralai{details}")
        client = Mistral(api_key=self.settings.api_key)

        kwargs: Dict[str, Any] = {"model": self.settings.model}
        if request.timestamp_granularity != "none":
            kwargs["timestamp_granularities"] = [request.timestamp_granularity]
        elif request.language_mode == "manual" and request.language:
            kwargs["language"] = request.language
        if request.diarize:
            kwargs["diarize"] = True
        if request.context_bias:
            kwargs["context_bias"] = request.context_bias

        if progress_cb:
            progress_cb("正在调用 Mistral API")
        with request.audio_path.open("rb") as file_obj:
            response = client.audio.transcriptions.complete(
                file={"content": file_obj, "file_name": request.audio_path.name},
                **kwargs,
            )
        payload = normalize_response(response)
        return TranscriptionResult(
            text=extract_text(payload),
            segments=extract_segments(payload),
            language=detect_language_code(payload),
            raw_payload=payload,
        )


class WhisperOpenAICompatibleProvider(TranscriptionProvider):
    def __init__(self, settings: WhisperProviderSettings, http_client: Optional[HttpClient] = None) -> None:
        self.settings = settings
        self.http_client = http_client or HttpClient()

    def transcribe(
        self,
        request: TranscriptionRequest,
        progress_cb: Optional[Callable[[str], None]],
        cancel_event: Event,
    ) -> TranscriptionResult:
        if cancel_event.is_set():
            raise RuntimeError("转写前已取消")
        if progress_cb:
            progress_cb("正在调用 Whisper 接口")

        with request.audio_path.open("rb") as file_obj:
            audio_bytes = file_obj.read()

        endpoint = normalize_audio_transcriptions_url(self.settings.base_url)
        data = self._build_form_data(request, include_timestamps=request.timestamp_granularity != "none")
        headers = {"Authorization": f"Bearer {self.settings.api_key}"}
        files = {
            "file": (
                request.audio_path.name,
                audio_bytes,
                self._guess_mime_type(request.audio_path),
            )
        }

        response = self.http_client.post_multipart(endpoint, data=data, files=files, headers=headers)
        if response.status_code >= 400 and data.get("timestamp_granularities[]"):
            error_text = response.text.lower()
            if "timestamp" in error_text or "granularit" in error_text:
                fallback_data = self._build_form_data(request, include_timestamps=False)
                response = self.http_client.post_multipart(endpoint, data=fallback_data, files=files, headers=headers)

        if response.status_code >= 400:
            raise RuntimeError(
                f"Whisper 兼容接口返回错误: HTTP {response.status_code} {response.text[:240]}"
            )
        payload = response.payload if isinstance(response.payload, dict) else {"text": response.text}

        segments = extract_segments(payload)
        text = extract_text(payload)
        language = detect_language_code(payload)
        return TranscriptionResult(
            text=text,
            segments=segments,
            language=language,
            raw_payload=payload,
        )

    def _build_form_data(self, request: TranscriptionRequest, include_timestamps: bool) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "model": self.settings.model,
            "response_format": "verbose_json",
        }
        if request.language_mode == "manual" and request.language:
            data["language"] = request.language
        if request.context_bias:
            data["prompt"] = request.context_bias
        if include_timestamps:
            data["timestamp_granularities[]"] = request.timestamp_granularity
        return data

    def _guess_mime_type(self, path: Path) -> str:
        suffix = path.suffix.lower()
        return {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".m4a": "audio/mp4",
            ".aac": "audio/aac",
            ".flac": "audio/flac",
            ".ogg": "audio/ogg",
        }.get(suffix, "application/octet-stream")



def summarize_empty_transcription_response(payload: Dict[str, Any], raw_text: str) -> str:
    if payload:
        keys = ", ".join(sorted(str(key) for key in payload.keys())[:8])
        if keys:
            return f" 响应字段: {keys}。"
    snippet = raw_text.strip().replace("\r", " ").replace("\n", " ")
    if snippet:
        return f" 原始响应片段: {snippet[:240]}"
    return " 请检查服务端日志，以及所选模型是否真的支持 OpenAI 兼容的 audio/transcriptions 输出。"
