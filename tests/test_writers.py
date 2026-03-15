from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from subtitle_studio.config import default_settings
from subtitle_studio.writers import build_bilingual_srt_text, write_translation_outputs


class WriterTests(unittest.TestCase):
    def test_build_bilingual_srt_text(self) -> None:
        text = build_bilingual_srt_text(
            [{"start": 0.0, "end": 1.0, "text": "hello"}],
            [{"start": 0.0, "end": 1.0, "text": "你好"}],
        )
        self.assertIn("hello", text)
        self.assertIn("你好", text)

    def test_write_translation_outputs_creates_srt(self) -> None:
        settings = default_settings()
        settings.translation.mode = "openai"
        settings.translation.target_language = "zh"
        settings.output.save_srt = True
        settings.output.save_txt = False
        settings.output.save_json = False
        with tempfile.TemporaryDirectory() as temp_dir:
            outputs = write_translation_outputs(
                target_dir=Path(temp_dir),
                source_stem="sample",
                source_lang_code="en",
                settings=settings,
                original_segments=[{"start": 0.0, "end": 1.0, "text": "hello"}],
                translated_segments=[{"start": 0.0, "end": 1.0, "text": "你好"}],
                original_text="hello",
                translated_text="你好",
            )
            self.assertIn("srt_翻译", outputs)
            self.assertTrue(Path(outputs["srt_翻译"]).exists())


if __name__ == "__main__":
    unittest.main()
