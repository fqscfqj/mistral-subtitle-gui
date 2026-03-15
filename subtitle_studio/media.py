from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .constants import AUDIO_EXTENSIONS, MEDIA_EXTENSIONS, SUPPORTED_EXTENSIONS, VIDEO_EXTENSIONS
from .models import VadSettings
from .vad import detect_speech_segments, load_wave_16k_mono, merge_speech_segments


@dataclass
class AudioPreparation:
    audio_path: Path
    is_audio_input: bool
    cleanup_paths: list[Path]


@dataclass
class AudioChunk:
    path: Path
    start_offset: float
    end_offset: float


def discover_supported_files(folder: Path) -> list[Path]:
    found: list[Path] = []
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            found.append(path)
    return sorted(found)


def extract_audio_with_ffmpeg(ffmpeg_path: str, input_file: Path, output_file: Path) -> None:
    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        str(input_file),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_file),
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 执行失败: {result.stderr.strip()[-400:]}")


def ensure_wav_audio(ffmpeg_path: str, source_path: Path) -> AudioPreparation:
    with tempfile.NamedTemporaryFile(prefix="subtitle_audio_", suffix=".wav", delete=False) as tmp:
        output = Path(tmp.name)
    extract_audio_with_ffmpeg(ffmpeg_path, source_path, output)
    return AudioPreparation(
        audio_path=output,
        is_audio_input=source_path.suffix.lower() in AUDIO_EXTENSIONS,
        cleanup_paths=[output],
    )


def prepare_audio_source(source_path: Path, ffmpeg_path: str, use_vad: bool) -> AudioPreparation:
    suffix = source_path.suffix.lower()
    if suffix not in MEDIA_EXTENSIONS:
        raise RuntimeError("该文件类型不是音视频文件，无法执行转录")
    if suffix in VIDEO_EXTENSIONS:
        if not ffmpeg_path:
            raise RuntimeError("视频文件需要 ffmpeg")
        return ensure_wav_audio(ffmpeg_path, source_path)
    if use_vad:
        if not ffmpeg_path:
            raise RuntimeError("启用 Silero VAD 时需要 ffmpeg")
        return ensure_wav_audio(ffmpeg_path, source_path)
    return AudioPreparation(audio_path=source_path, is_audio_input=True, cleanup_paths=[])


def _derive_ffprobe_path(ffmpeg_path: str) -> str:
    path = Path(ffmpeg_path)
    candidate = path.parent / f"ffprobe{path.suffix}"
    if candidate.exists():
        return str(candidate)
    found = shutil.which("ffprobe")
    return found or ""


def get_audio_duration_seconds(ffmpeg_path: str, audio_path: Path) -> float:
    ffprobe_path = _derive_ffprobe_path(ffmpeg_path)
    if ffprobe_path:
        result = subprocess.run(
            [
                ffprobe_path,
                "-v",
                "quiet",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                str(audio_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        raw = result.stdout.strip()
        try:
            value = float(raw)
            if value > 0:
                return value
        except ValueError:
            pass

    result = subprocess.run(
        [ffmpeg_path, "-i", str(audio_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr)
    if not match:
        raise RuntimeError(f"无法获取音频时长，请确认文件格式受 ffmpeg 支持: {audio_path.name}")
    return int(match.group(1)) * 3600.0 + int(match.group(2)) * 60.0 + float(match.group(3))


def split_audio_into_chunks(
    ffmpeg_path: str,
    audio_path: Path,
    chunk_duration: float,
    temp_dir: Path,
    total_seconds: Optional[float] = None,
) -> list[AudioChunk]:
    total = total_seconds if total_seconds is not None else get_audio_duration_seconds(ffmpeg_path, audio_path)
    if total <= 0 or total <= chunk_duration:
        return [AudioChunk(path=audio_path, start_offset=0.0, end_offset=total)]

    suffix = audio_path.suffix or ".wav"
    chunks: list[AudioChunk] = []
    offset = 0.0
    idx = 0
    while offset < total:
        output_path = temp_dir / f"chunk_{idx:04d}{suffix}"
        duration = min(chunk_duration, total - offset)
        result = subprocess.run(
            [
                ffmpeg_path,
                "-y",
                "-ss",
                f"{offset:.3f}",
                "-t",
                f"{duration:.3f}",
                "-i",
                str(audio_path),
                "-c",
                "copy",
                str(output_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg 音频分割失败: {result.stderr.strip()[-400:]}")
        chunks.append(AudioChunk(path=output_path, start_offset=offset, end_offset=offset + duration))
        offset += duration
        idx += 1
    return chunks


def _write_wav_clip(output_path: Path, samples: np.ndarray) -> None:
    pcm = (samples.clip(-1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(output_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(pcm.tobytes())


def split_audio_with_vad(wav_path: Path, vad_settings: VadSettings, temp_dir: Path) -> list[AudioChunk]:
    audio = load_wave_16k_mono(wav_path)
    detected = detect_speech_segments(audio, vad_settings)
    merged = merge_speech_segments(detected, vad_settings.max_segment_seconds)
    if not merged:
        return [AudioChunk(path=wav_path, start_offset=0.0, end_offset=len(audio) / 16000.0)]

    chunks: list[AudioChunk] = []
    for idx, segment in enumerate(merged):
        start_idx = max(0, int(round(segment.start * 16000)))
        end_idx = min(len(audio), int(round(segment.end * 16000)))
        output_path = temp_dir / f"vad_{idx:04d}.wav"
        _write_wav_clip(output_path, audio[start_idx:end_idx])
        chunks.append(AudioChunk(path=output_path, start_offset=segment.start, end_offset=segment.end))
    return chunks


def cleanup_paths(paths: list[Path]) -> None:
    for path in paths:
        if not path.exists():
            continue
        try:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
        except Exception:
            continue
