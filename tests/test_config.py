from __future__ import annotations

import unittest
from unittest.mock import patch

from subtitle_studio.config import deserialize_settings, has_ffmpeg, serialize_settings


class SettingsCompatibilityTests(unittest.TestCase):
    def test_deserialize_old_flat_settings(self) -> None:
        settings = deserialize_settings(
            {
                "api_key": "m-key",
                "model": "voxtral-small-latest",
                "language_mode_index": 1,
                "language": "en",
                "timestamp": "word",
                "translation_mode_index": 2,
                "translation_target": "zh",
                "translation_model": "gpt-4o-mini",
                "translation_openai_base": "https://example.com/v1",
                "translation_openai_key": "o-key",
                "output_mode_index": 1,
                "output_dir": "out",
                "silero_vad_enabled": True,
            }
        )
        self.assertEqual(settings.transcription.provider, "mistral")
        self.assertEqual(settings.transcription.mistral.api_key, "m-key")
        self.assertEqual(settings.transcription.language_mode, "manual")
        self.assertEqual(settings.translation.mode, "openai")
        self.assertTrue(settings.vad.enabled)

    def test_serialize_roundtrip_contains_new_fields(self) -> None:
        settings = deserialize_settings(
            {
                "transcription_provider": "whisper_openai_compatible",
                "ffmpeg_path": "C:/tools/ffmpeg.exe",
                "vad_min_speech_ms": 320,
                "vad_min_silence_ms": 520,
                "vad_speech_pad_ms": 180,
                "vad_max_segment_seconds": 120,
                "vad_threshold": 0.65,
            }
        )
        payload = serialize_settings(settings)
        self.assertIn("transcription_provider", payload)
        self.assertIn("whisper_base_url", payload)
        self.assertIn("whisper_api_key", payload)
        self.assertIn("whisper_model", payload)
        self.assertIn("silero_vad_enabled", payload)
        self.assertIn("vad_min_speech_ms", payload)
        self.assertIn("vad_min_silence_ms", payload)
        self.assertIn("vad_speech_pad_ms", payload)
        self.assertIn("vad_max_segment_seconds", payload)
        self.assertIn("vad_threshold", payload)
        self.assertNotIn("ffmpeg_path", payload)
        self.assertEqual(settings.vad.min_speech_ms, 320)
        self.assertEqual(settings.vad.min_silence_ms, 520)
        self.assertEqual(settings.vad.speech_pad_ms, 180)
        self.assertEqual(settings.vad.max_segment_seconds, 120)
        self.assertEqual(settings.vad.threshold, 0.65)

    @patch("subtitle_studio.config.find_ffmpeg", return_value="C:/bundle/ffmpeg.exe")
    def test_has_ffmpeg_uses_runtime_detection(self, mock_find_ffmpeg) -> None:
        self.assertTrue(has_ffmpeg())
        mock_find_ffmpeg.assert_called_once_with()

    @patch("subtitle_studio.config.find_ffmpeg", return_value="")
    def test_has_ffmpeg_false_when_runtime_binary_missing(self, mock_find_ffmpeg) -> None:
        self.assertFalse(has_ffmpeg())
        mock_find_ffmpeg.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
