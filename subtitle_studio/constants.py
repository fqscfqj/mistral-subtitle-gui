from __future__ import annotations

SETTINGS_FILE = ".ai_subtitle_studio_settings.json"
MAX_MISTRAL_CHUNK_DURATION_SECONDS = 3 * 3600
DEFAULT_VAD_MAX_SEGMENT_SECONDS = 15 * 60
DEFAULT_VAD_THRESHOLD = 0.5
DEFAULT_VAD_MIN_SPEECH_MS = 250
DEFAULT_VAD_MIN_SILENCE_MS = 400
DEFAULT_VAD_SPEECH_PAD_MS = 200

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".wmv",
    ".webm",
    ".m4v",
    ".flv",
    ".ts",
}

AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".aac",
    ".flac",
    ".ogg",
    ".opus",
    ".wma",
}

SUBTITLE_EXTENSIONS = {
    ".srt",
    ".vtt",
    ".txt",
}

MEDIA_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS
SUPPORTED_EXTENSIONS = MEDIA_EXTENSIONS | SUBTITLE_EXTENSIONS

STATUS_LABELS = {
    "Queued": "排队中",
    "Preparing": "准备中",
    "Extracting": "提取音频",
    "Transcribing": "转写中",
    "Translating": "翻译中",
    "Writing": "写入文件",
    "Completed": "已完成",
    "Failed": "失败",
    "Cancelled": "已取消",
}

COMMON_LANGUAGE_CODES = {
    "zh",
    "en",
    "ja",
    "ko",
    "fr",
    "de",
    "es",
    "it",
    "pt",
    "ru",
    "ar",
    "hi",
    "tr",
    "th",
    "vi",
    "id",
    "ms",
    "pl",
    "nl",
    "sv",
    "da",
    "fi",
    "no",
    "cs",
    "ro",
    "hu",
    "uk",
}

