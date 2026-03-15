from __future__ import annotations

import unittest

from subtitle_studio.config import deserialize_settings, serialize_settings


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
        self.assertIn("ffmpeg_path", payload)
        self.assertIn("silero_vad_enabled", payload)
        self.assertIn("vad_min_speech_ms", payload)
        self.assertIn("vad_min_silence_ms", payload)
        self.assertIn("vad_speech_pad_ms", payload)
        self.assertIn("vad_max_segment_seconds", payload)
        self.assertIn("vad_threshold", payload)
        self.assertEqual(settings.output.ffmpeg_path, "C:/tools/ffmpeg.exe")
        self.assertEqual(settings.vad.min_speech_ms, 320)
        self.assertEqual(settings.vad.min_silence_ms, 520)
        self.assertEqual(settings.vad.speech_pad_ms, 180)
        self.assertEqual(settings.vad.max_segment_seconds, 120)
        self.assertEqual(settings.vad.threshold, 0.65)


if __name__ == "__main__":
    unittest.main()
