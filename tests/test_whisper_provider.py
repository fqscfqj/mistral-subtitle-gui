from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from subtitle_studio.http_client import HttpResponse
from subtitle_studio.models import TranscriptionRequest, WhisperProviderSettings
from subtitle_studio.providers.transcription import WhisperOpenAICompatibleProvider


class FakeHttpClient:
    def __init__(self) -> None:
        self.calls = []

    def post_multipart(self, url, data, files, headers):
        self.calls.append({"url": url, "data": dict(data), "headers": dict(headers)})
        if len(self.calls) == 1:
            return HttpResponse(status_code=400, payload=None, text="timestamp_granularities not supported")
        return HttpResponse(
            status_code=200,
            payload={
                "text": "hello world",
                "language": "en",
                "segments": [{"start": 0.0, "end": 1.0, "text": "hello world"}],
            },
            text="",
        )


class FakeTimestampHttpClient:
    def __init__(self, payload) -> None:
        self.payload = payload

    def post_multipart(self, url, data, files, headers):
        return HttpResponse(status_code=200, payload=self.payload, text="")


class FakeEmptyHttpClient:
    def post_multipart(self, url, data, files, headers):
        return HttpResponse(status_code=200, payload={"text": "", "segments": []}, text="")


class WhisperProviderTests(unittest.TestCase):
    def test_retry_without_timestamp_granularity(self) -> None:
        fake_client = FakeHttpClient()
        provider = WhisperOpenAICompatibleProvider(
            WhisperProviderSettings(base_url="https://example.com/v1", api_key="key", model="whisper-1"),
            http_client=fake_client,
        )
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(b"RIFFfake")
            audio_path = Path(tmp.name)
        try:
            result = provider.transcribe(
                TranscriptionRequest(
                    source_path=audio_path,
                    audio_path=audio_path,
                    language_mode="manual",
                    language="en",
                    timestamp_granularity="segment",
                    diarize=False,
                    context_bias="hello",
                ),
                progress_cb=None,
                cancel_event=type("Cancel", (), {"is_set": lambda self: False})(),
            )
        finally:
            audio_path.unlink(missing_ok=True)

        self.assertEqual(len(fake_client.calls), 2)
        self.assertIn("timestamp_granularities[]", fake_client.calls[0]["data"])
        self.assertNotIn("timestamp_granularities[]", fake_client.calls[1]["data"])
        self.assertEqual(result.text, "hello world")
        self.assertEqual(result.language, "en")

    def test_normalizes_nanosecond_like_timestamps(self) -> None:
        fake_client = FakeTimestampHttpClient(
            {
                "text": "hello world",
                "segments": [{"start": 152000000.0, "end": 2152000000.0, "text": "hello world"}],
            }
        )
        provider = WhisperOpenAICompatibleProvider(
            WhisperProviderSettings(base_url="https://example.com/v1", api_key="key", model="whisper-1"),
            http_client=fake_client,
        )
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(b"RIFFfake")
            audio_path = Path(tmp.name)
        try:
            result = provider.transcribe(
                TranscriptionRequest(
                    source_path=audio_path,
                    audio_path=audio_path,
                    language_mode="auto",
                    language="",
                    timestamp_granularity="segment",
                    diarize=False,
                    context_bias="",
                ),
                progress_cb=None,
                cancel_event=type("Cancel", (), {"is_set": lambda self: False})(),
            )
        finally:
            audio_path.unlink(missing_ok=True)

        self.assertAlmostEqual(result.segments[0]["start"], 0.152, places=3)
        self.assertAlmostEqual(result.segments[0]["end"], 2.152, places=3)

    def test_normalizes_split_billion_timestamps(self) -> None:
        fake_client = FakeTimestampHttpClient(
            {
                "text": "line1 line2",
                "segments": [
                    {"start": 10.232, "end": 1000000010.232, "text": "line1"},
                    {"start": 1000000010.232, "end": 2000000000.232, "text": "line2"},
                ],
            }
        )
        provider = WhisperOpenAICompatibleProvider(
            WhisperProviderSettings(base_url="https://example.com/v1", api_key="key", model="whisper-1"),
            http_client=fake_client,
        )
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(b"RIFFfake")
            audio_path = Path(tmp.name)
        try:
            result = provider.transcribe(
                TranscriptionRequest(
                    source_path=audio_path,
                    audio_path=audio_path,
                    language_mode="auto",
                    language="",
                    timestamp_granularity="segment",
                    diarize=False,
                    context_bias="",
                ),
                progress_cb=None,
                cancel_event=type("Cancel", (), {"is_set": lambda self: False})(),
            )
        finally:
            audio_path.unlink(missing_ok=True)

        self.assertAlmostEqual(result.segments[0]["start"], 10.232, places=3)
        self.assertAlmostEqual(result.segments[0]["end"], 11.232, places=3)
        self.assertAlmostEqual(result.segments[1]["start"], 11.232, places=3)
        self.assertAlmostEqual(result.segments[1]["end"], 13.232, places=3)

    def test_raises_on_empty_successful_response(self) -> None:
        provider = WhisperOpenAICompatibleProvider(
            WhisperProviderSettings(base_url="https://example.com/v1", api_key="key", model="nemo-parakeet-tdt-0.6b"),
            http_client=FakeEmptyHttpClient(),
        )
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(b"RIFFfake")
            audio_path = Path(tmp.name)
        try:
            with self.assertRaises(RuntimeError) as ctx:
                provider.transcribe(
                    TranscriptionRequest(
                        source_path=audio_path,
                        audio_path=audio_path,
                        language_mode="auto",
                        language="",
                        timestamp_granularity="segment",
                        diarize=False,
                        context_bias="",
                    ),
                    progress_cb=None,
                    cancel_event=type("Cancel", (), {"is_set": lambda self: False})(),
                )
        finally:
            audio_path.unlink(missing_ok=True)

        self.assertIn("空转写结果", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
