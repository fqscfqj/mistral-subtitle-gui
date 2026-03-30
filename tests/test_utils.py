from __future__ import annotations

import unittest

from subtitle_studio.utils import extract_segments, sanitize_transcribed_text


class SanitizeTranscribedTextTests(unittest.TestCase):
    def test_filters_stage_directions(self) -> None:
        for raw in ("*Sigh*", "[Music]", "(applause)", "（叹气）"):
            with self.subTest(raw=raw):
                self.assertEqual(sanitize_transcribed_text(raw), "")

    def test_filters_number_and_symbol_fragments(self) -> None:
        for raw in ("-", "- 10.", "...", "• 3)"):
            with self.subTest(raw=raw):
                self.assertEqual(sanitize_transcribed_text(raw), "")

    def test_keeps_pure_number_utterances(self) -> None:
        for raw in ("10", "10.", "2025"):
            with self.subTest(raw=raw):
                self.assertEqual(sanitize_transcribed_text(raw), raw)

    def test_filters_common_filler_utterances(self) -> None:
        for raw in ("Mm-hmm.", "Uh-huh.", "Hmm...", "Um..."):
            with self.subTest(raw=raw):
                self.assertEqual(sanitize_transcribed_text(raw), "")

    def test_normalizes_dialogue_dash(self) -> None:
        self.assertEqual(sanitize_transcribed_text("- Yeah, it's the first time."), "Yeah, it's the first time.")
        self.assertEqual(sanitize_transcribed_text("- Meili."), "Meili.")

    def test_keeps_real_short_utterances(self) -> None:
        cases = {
            "Yes.": "Yes.",
            "No.": "No.",
            "Okay.": "Okay.",
            "I'm here.": "I'm here.",
            "好的。": "好的。",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(sanitize_transcribed_text(raw), expected)

    def test_extract_segments_skips_filtered_noise(self) -> None:
        segments = extract_segments(
            {
                "segments": [
                    {"start": 0.0, "end": 0.5, "text": "*Sigh*"},
                    {"start": 0.5, "end": 1.0, "text": "Mm-hmm."},
                    {"start": 1.0, "end": 2.0, "text": "I’m here."},
                ]
            }
        )

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["text"], "I’m here.")
        self.assertAlmostEqual(segments[0]["start"], 1.0, places=3)
        self.assertAlmostEqual(segments[0]["end"], 2.0, places=3)


if __name__ == "__main__":
    unittest.main()
