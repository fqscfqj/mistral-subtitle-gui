# Mistral Subtitle Studio (Desktop GUI)

A PySide6 desktop app for generating subtitles from video/audio files using Mistral Audio Transcription API.

## Features

- Drag & drop files/folders
- Add single files or recursively scan a folder
- Independent pages: `任务` and `设置`
- Multi-threaded task execution (configurable thread count)
- Per-task status and progress + global progress
- Configurable Mistral options:
  - `model`
  - language mode: auto-detect or manual language code
  - `timestamp_granularities` (`none`, `segment`, `word`)
  - `diarize`
  - `context_bias`
- Optional subtitle translation:
  - no translation / Mistral API / OpenAI-compatible API
  - OpenAI-compatible `base_url` + `api_key` (supports third-party providers)
  - target language code
  - bilingual SRT output (source + translated line)
- Output directory mode:
  - use source file directory
  - use custom directory
- Language-suffixed output naming (example: `abc.zh.srt`)
- Output formats:
  - `.srt`
  - `.txt`
  - `.json`
- Batch stop/cancel support for queued tasks

## Requirements

- Python 3.10+
- `ffmpeg` available in `PATH` (required for video files)
- Mistral API key (`MISTRAL_API_KEY`)

## Install

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## Run

```bash
python mistral_subtitle_gui.py
```

## Notes on API behavior

- Based on Mistral docs, `language` cannot be used together with `timestamp_granularities`.
- This app will ignore `language` when timestamp granularity is `segment` or `word`.
- `context_bias` is normalized and capped to 100 unique terms.

## Environment variables

- `MISTRAL_API_KEY`: default API key loaded at startup
- `OPENAI_API_KEY`: optional default key for OpenAI-compatible translation
- `FFMPEG_BINARY`: optional absolute path to ffmpeg executable

## Output

Outputs are named with source stem + language code suffix:

- `input_video.zh.srt`
- `input_video.zh.txt`
- `input_video.zh.json`

When translation is enabled:

- source transcript: `input_video.en.srt` (example)
- translated subtitle: `input_video.zh.srt`
- if source and target language are the same, translated files use `.translated` to avoid overwrite (e.g. `input_video.zh.translated.srt`)
