from __future__ import annotations

import unittest

from subtitle_studio.providers import transcription, translation


class MistralImportCompatibilityTests(unittest.TestCase):
    def test_transcription_provider_supports_installed_mistralai_sdk(self) -> None:
        self.assertIsNotNone(
            transcription.Mistral,
            f"transcription provider failed to import mistralai: {transcription._MISTRAL_IMPORT_ERROR}",
        )

    def test_translation_provider_supports_installed_mistralai_sdk(self) -> None:
        self.assertIsNotNone(
            translation.Mistral,
            f"translation provider failed to import mistralai: {translation._MISTRAL_IMPORT_ERROR}",
        )


if __name__ == "__main__":
    unittest.main()
