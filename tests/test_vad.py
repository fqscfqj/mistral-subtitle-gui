from __future__ import annotations

import unittest

from subtitle_studio.vad import VadSegment, merge_speech_segments


class VadMergeTests(unittest.TestCase):
    def test_merge_segments_until_max_duration(self) -> None:
        segments = [
            VadSegment(0.0, 3.0),
            VadSegment(3.2, 5.0),
            VadSegment(12.0, 18.0),
        ]
        merged = merge_speech_segments(segments, max_segment_seconds=10)
        self.assertEqual(len(merged), 2)
        self.assertEqual((merged[0].start, merged[0].end), (0.0, 5.0))
        self.assertEqual((merged[1].start, merged[1].end), (12.0, 18.0))


if __name__ == "__main__":
    unittest.main()

