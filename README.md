# Mistral Subtitle Studio (Desktop GUI)

A PySide6 desktop app for generating subtitles from video/audio files using Mistral Audio Transcription API.

## Features

- Drag & drop files/folders
- Add single files or recursively scan a folder
- Multi-threaded task execution (configurable thread count)
- Per-task status and progress + global progress
- Configurable Mistral options:
  - `model`
  - `language` (only used when timestamp granularity is `none`)
  - `timestamp_granularities` (`none`, `segment`, `word`)
  - `diarize`
  - `context_bias`
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
- `FFMPEG_BINARY`: optional absolute path to ffmpeg executable

## Output

All outputs are written to the selected output folder using source filename stem:

- `input_video.srt`
- `input_video.txt`
- `input_video.json`
