from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import onnxruntime

from .constants import (
    DEFAULT_VAD_MAX_SEGMENT_SECONDS,
    DEFAULT_VAD_MIN_SILENCE_MS,
    DEFAULT_VAD_MIN_SPEECH_MS,
    DEFAULT_VAD_SPEECH_PAD_MS,
    DEFAULT_VAD_THRESHOLD,
)
from .models import VadSettings
from .resources import resource_path


@dataclass
class VadSegment:
    start: float
    end: float


class SileroVadSession:
    def __init__(self, model_path: Optional[Path] = None) -> None:
        path = model_path or resource_path("assets/silero_vad.onnx")
        if not path.exists():
            raise RuntimeError(f"未找到 Silero VAD 模型: {path}")
        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(
            str(path),
            providers=["CPUExecutionProvider"],
            sess_options=options,
        )
        self.reset_states()

    def reset_states(self, batch_size: int = 1) -> None:
        self.state = np.zeros((2, batch_size, 128), dtype=np.float32)
        self.context = np.zeros((batch_size, 64), dtype=np.float32)
        self.last_sample_rate = 16000
        self.last_batch_size = batch_size

    def __call__(self, chunk: np.ndarray, sample_rate: int) -> float:
        if chunk.ndim == 1:
            chunk = chunk[np.newaxis, :]
        if chunk.shape[1] != 512:
            raise ValueError(f"Silero VAD 需要 512 采样窗口，当前为 {chunk.shape[1]}")

        batch_size = chunk.shape[0]
        if self.last_batch_size != batch_size or self.last_sample_rate != sample_rate:
            self.reset_states(batch_size)

        full_chunk = np.asarray(np.concatenate([self.context, chunk], axis=1), dtype=np.float32)
        outputs = self.session.run(
            None,
            {"input": full_chunk, "state": self.state, "sr": np.array(sample_rate, dtype=np.int64)},
        )
        probabilities = np.asarray(outputs[0], dtype=np.float32)
        state = np.asarray(outputs[1], dtype=np.float32)
        self.state = state
        self.context = full_chunk[:, -64:]
        self.last_batch_size = batch_size
        self.last_sample_rate = sample_rate
        return float(probabilities[0][0])


def load_wave_16k_mono(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    if sample_rate != 16000:
        raise RuntimeError(f"VAD 输入音频采样率必须为 16000，当前为 {sample_rate}")
    if sample_width != 2:
        raise RuntimeError(f"VAD 输入音频采样宽度必须为 16-bit，当前为 {sample_width * 8}-bit")

    audio = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return (audio.astype(np.float32) / 32768.0).clip(-1.0, 1.0)


def _speech_probabilities(
    audio: np.ndarray,
    session: SileroVadSession,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> List[float]:
    window = 512
    session.reset_states()
    probabilities: List[float] = []
    total = max(len(audio), 1)
    for start in range(0, len(audio), window):
        chunk = audio[start : start + window]
        if len(chunk) < window:
            chunk = np.pad(chunk, (0, window - len(chunk)), mode="constant")
        probabilities.append(session(chunk.astype(np.float32), 16000))
        if progress_cb:
            progress_cb(min(100.0, 100.0 * min(start + window, len(audio)) / total))
    return probabilities


def detect_speech_segments(
    audio: np.ndarray,
    settings: VadSettings,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> List[VadSegment]:
    session = SileroVadSession()
    probabilities = _speech_probabilities(audio, session, progress_cb=progress_cb)
    threshold = settings.threshold or DEFAULT_VAD_THRESHOLD
    neg_threshold = max(threshold - 0.15, 0.01)
    min_speech_samples = int(16000 * (settings.min_speech_ms or DEFAULT_VAD_MIN_SPEECH_MS) / 1000)
    min_silence_samples = int(16000 * (settings.min_silence_ms or DEFAULT_VAD_MIN_SILENCE_MS) / 1000)
    speech_pad_samples = int(16000 * (settings.speech_pad_ms or DEFAULT_VAD_SPEECH_PAD_MS) / 1000)
    max_speech_samples = int(16000 * (settings.max_segment_seconds or DEFAULT_VAD_MAX_SEGMENT_SECONDS))
    min_silence_samples_at_max = int(16000 * 0.098)
    window = 512

    speeches: list[dict[str, int]] = []
    triggered = False
    current_speech: dict[str, int] = {}
    temp_end = 0
    prev_end = 0
    next_start = 0
    possible_ends: list[tuple[int, int]] = []

    for idx, prob in enumerate(probabilities):
        current_sample = idx * window

        if prob >= threshold and temp_end:
            silence_duration = current_sample - temp_end
            if silence_duration > min_silence_samples_at_max:
                possible_ends.append((temp_end, silence_duration))
            temp_end = 0
            if next_start < prev_end:
                next_start = current_sample

        if prob >= threshold and not triggered:
            triggered = True
            current_speech["start"] = current_sample
            continue

        if triggered and current_sample - current_speech["start"] > max_speech_samples:
            if possible_ends:
                prev_end, duration = max(possible_ends, key=lambda item: item[1])
                current_speech["end"] = prev_end
                speeches.append(current_speech)
                current_speech = {}
                next_start = prev_end + duration
                if next_start < prev_end + current_sample:
                    current_speech["start"] = next_start
                else:
                    triggered = False
                prev_end = 0
                next_start = 0
                temp_end = 0
                possible_ends = []
            elif prev_end:
                current_speech["end"] = prev_end
                speeches.append(current_speech)
                current_speech = {}
                if next_start >= prev_end:
                    current_speech["start"] = next_start
                else:
                    triggered = False
                prev_end = 0
                next_start = 0
                temp_end = 0
                possible_ends = []
            else:
                current_speech["end"] = current_sample
                speeches.append(current_speech)
                current_speech = {}
                triggered = False
                continue

        if prob < neg_threshold and triggered:
            if not temp_end:
                temp_end = current_sample
            silence_duration = current_sample - temp_end
            if silence_duration < min_silence_samples:
                continue
            current_speech["end"] = temp_end
            if current_speech["end"] - current_speech["start"] > min_speech_samples:
                speeches.append(current_speech)
            current_speech = {}
            triggered = False
            temp_end = 0
            prev_end = 0
            next_start = 0
            possible_ends = []

    if current_speech and len(audio) - current_speech["start"] > min_speech_samples:
        current_speech["end"] = len(audio)
        speeches.append(current_speech)

    for idx, speech in enumerate(speeches):
        if idx == 0:
            speech["start"] = max(0, speech["start"] - speech_pad_samples)
        if idx != len(speeches) - 1:
            silence_duration = speeches[idx + 1]["start"] - speech["end"]
            if silence_duration < 2 * speech_pad_samples:
                speech["end"] += silence_duration // 2
                speeches[idx + 1]["start"] = max(0, speeches[idx + 1]["start"] - silence_duration // 2)
            else:
                speech["end"] = min(len(audio), speech["end"] + speech_pad_samples)
                speeches[idx + 1]["start"] = max(0, speeches[idx + 1]["start"] - speech_pad_samples)
        else:
            speech["end"] = min(len(audio), speech["end"] + speech_pad_samples)

    return [VadSegment(start=speech["start"] / 16000.0, end=speech["end"] / 16000.0) for speech in speeches]


def merge_speech_segments(segments: List[VadSegment], max_segment_seconds: int) -> List[VadSegment]:
    if not segments:
        return []
    merged: List[VadSegment] = []
    current = VadSegment(start=segments[0].start, end=segments[0].end)
    for segment in segments[1:]:
        current_span = segment.end - current.start
        # Only merge overlapping/contiguous segments; keep silence boundaries intact.
        should_merge = segment.start <= current.end and current_span <= max_segment_seconds
        if should_merge:
            current.end = segment.end
            continue
        merged.append(current)
        current = VadSegment(start=segment.start, end=segment.end)
    merged.append(current)
    return merged
