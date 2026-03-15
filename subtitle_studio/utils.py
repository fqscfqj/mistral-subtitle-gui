from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List

from .constants import COMMON_LANGUAGE_CODES


def new_task_id() -> str:
    return uuid.uuid4().hex


def format_srt_timestamp(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000.0)))
    hours = ms // 3_600_000
    ms %= 3_600_000
    minutes = ms // 60_000
    ms %= 60_000
    secs = ms // 1000
    ms %= 1000
    return f"{hours:02}:{minutes:02}:{secs:02},{ms:03}"


def format_lrc_timestamp(seconds: float) -> str:
    centiseconds = max(0, int(round(seconds * 100.0)))
    minutes = centiseconds // 6000
    centiseconds %= 6000
    secs = centiseconds // 100
    centiseconds %= 100
    return f"{minutes:02}:{secs:02}.{centiseconds:02}"


def normalize_response(response: Any) -> Dict[str, Any]:
    if isinstance(response, dict):
        return response
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "dict"):
        return response.dict()
    if hasattr(response, "model_dump_json"):
        return json.loads(response.model_dump_json())
    return {"text": str(response)}


def normalize_language_code(value: str) -> str:
    cleaned = value.strip().lower().replace("_", "-")
    if not cleaned:
        return ""
    primary = cleaned.split("-", 1)[0].strip()
    return "".join(ch for ch in primary if ch.isalnum())


def parse_context_bias(raw_text: str) -> str:
    tokens: List[str] = []
    seen = set()
    normalized = raw_text.replace("\n", ",")
    for part in normalized.split(","):
        token = part.strip()
        if not token:
            continue
        key = token.casefold()
        if key in seen:
            continue
        seen.add(key)
        tokens.append(token)
        if len(tokens) >= 100:
            break
    return ",".join(tokens)


def detect_language_code(payload: Dict[str, Any]) -> str:
    candidates: List[str] = []
    for key in ("language", "detected_language", "lang"):
        value = payload.get(key)
        if isinstance(value, str):
            candidates.append(value)

    meta = payload.get("metadata")
    if isinstance(meta, dict):
        for key in ("language", "detected_language", "lang"):
            value = meta.get(key)
            if isinstance(value, str):
                candidates.append(value)

    for candidate in candidates:
        code = normalize_language_code(candidate)
        if code:
            return code
    return "und"


def extract_text(payload: Dict[str, Any]) -> str:
    for key in ("text", "transcript", "output_text"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def extract_segments(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = payload.get("segments")
    if not isinstance(raw, list):
        return []

    segments: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        start = item.get("start")
        end = item.get("end")
        text = item.get("text")
        if isinstance(start, (int, float)) and isinstance(end, (int, float)):
            segment: Dict[str, Any] = {
                "start": float(start),
                "end": float(end),
                "text": str(text or "").strip(),
            }
            if "speaker" in item:
                segment["speaker"] = item.get("speaker")
            segments.append(segment)
    return segments


def build_srt_text(segments: List[Dict[str, Any]], fallback_text: str) -> str:
    if not segments:
        text = fallback_text.strip() or "(无转写内容)"
        return "1\n00:00:00,000 --> 99:59:59,999\n" + f"{text}\n"

    lines: List[str] = []
    for idx, seg in enumerate(segments, start=1):
        start = format_srt_timestamp(float(seg["start"]))
        end = format_srt_timestamp(float(seg["end"]))
        speaker = seg.get("speaker")
        prefix = f"[{speaker}] " if speaker not in (None, "") else ""
        text = (prefix + str(seg.get("text", "")).strip()).strip() or "..."
        lines.append(f"{idx}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)


def build_lrc_text(segments: List[Dict[str, Any]], fallback_text: str) -> str:
    if not segments:
        text = re.sub(r"\s+", " ", fallback_text.strip() or "(无转写内容)")
        return f"[00:00.00]{text}\n"

    lines: List[str] = []
    for seg in segments:
        start = format_lrc_timestamp(float(seg["start"]))
        speaker = seg.get("speaker")
        prefix = f"[{speaker}] " if speaker not in (None, "") else ""
        text = (prefix + str(seg.get("text", "")).strip()).strip() or "..."
        text = re.sub(r"\s*\n\s*", " / ", text)
        lines.append(f"[{start}]{text}")
    return "\n".join(lines) + "\n"


def parse_subtitle_timestamp(raw: str) -> float:
    token = raw.strip().replace(",", ".")
    if not token:
        raise ValueError("空时间戳")
    token = token.split(" ", 1)[0].strip()
    parts = token.split(":")
    if len(parts) == 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
    elif len(parts) == 2:
        hours = 0
        minutes = int(parts[0])
        seconds = float(parts[1])
    else:
        raise ValueError(f"非法时间戳: {raw}")
    return hours * 3600.0 + minutes * 60.0 + seconds


def parse_timed_subtitle_segments(raw_text: str) -> List[Dict[str, Any]]:
    text = raw_text.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n", text)
    segments: List[Dict[str, Any]] = []

    for block in blocks:
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if not lines:
            continue
        if lines[0].upper() == "WEBVTT":
            continue
        if len(lines) >= 2 and lines[0].isdigit():
            lines = lines[1:]
        if not lines:
            continue

        timing_index = -1
        for idx, line in enumerate(lines):
            if "-->" in line:
                timing_index = idx
                break
        if timing_index < 0:
            continue

        parts = lines[timing_index].split("-->", 1)
        if len(parts) != 2:
            continue
        try:
            start_sec = parse_subtitle_timestamp(parts[0])
            end_sec = parse_subtitle_timestamp(parts[1])
        except Exception:
            continue

        cue_lines = [line for line in lines[timing_index + 1 :] if line]
        cue_text = "\n".join(cue_lines).strip()
        if not cue_text:
            continue
        segments.append({"start": start_sec, "end": max(start_sec, end_sec), "text": cue_text})

    return segments


def read_text_with_fallback(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "cp1252"):
        try:
            return path.read_text(encoding=encoding)
        except Exception:
            continue
    raise RuntimeError(f"无法读取文本文件编码: {path.name}")


def extract_subtitle_source(path: Path) -> tuple[List[Dict[str, Any]], str]:
    raw = read_text_with_fallback(path)
    ext = path.suffix.lower()

    if ext in {".srt", ".vtt"}:
        segments = parse_timed_subtitle_segments(raw)
        if segments:
            text = "\n".join(
                str(seg.get("text", "")).strip()
                for seg in segments
                if str(seg.get("text", "")).strip()
            )
            return segments, text.strip()

    if ext == ".txt":
        lines = [
            line.strip()
            for line in raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
            if line.strip()
        ]
        return [], "\n".join(lines).strip()

    return [], raw.strip()


def infer_language_code_from_filename(path: Path) -> str:
    stem_tokens = re.split(r"[.\-_ ]+", path.stem.strip())
    for token in reversed(stem_tokens):
        code = normalize_language_code(token)
        if code in COMMON_LANGUAGE_CODES:
            return code
    return "und"


def trim_language_suffix_from_stem(source_stem: str, language_code: str) -> str:
    code = normalize_language_code(language_code)
    if not code or code == "und":
        return source_stem

    stem_fold = source_stem.casefold()
    suffixes = [f".{code}", f"_{code}", f"-{code}", f" {code}"]
    for suffix in suffixes:
        if stem_fold.endswith(suffix.casefold()):
            trimmed = source_stem[: -len(suffix)].rstrip(" ._-")
            return trimmed or source_stem
    return source_stem


def is_chinese_language(lang_code: str) -> bool:
    return normalize_language_code(lang_code) in {"zh", "zho", "chi", "cn"}


def extract_chat_text(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content.strip()
                if isinstance(content, list):
                    parts: List[str] = []
                    for item in content:
                        if isinstance(item, dict):
                            text = item.get("text")
                            if isinstance(text, str):
                                parts.append(text)
                    if parts:
                        return "\n".join(parts).strip()
    return ""


def parse_json_array_output(raw_text: str) -> List[str]:
    def strip_code_fence(text: str) -> str:
        stripped = text.strip()
        if not stripped.startswith("```"):
            return stripped
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip().startswith("```"):
            return "\n".join(lines[1:-1]).strip()
        return stripped

    def find_balanced_json(text: str, open_char: str, close_char: str) -> str:
        in_string = False
        escaped = False
        depth = 0
        start = -1
        for idx, ch in enumerate(text):
            if escaped:
                escaped = False
                continue
            if ch == "\\" and in_string:
                escaped = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == open_char:
                if depth == 0:
                    start = idx
                depth += 1
                continue
            if ch == close_char and depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    return text[start : idx + 1]
        return ""

    def extract_text_from_item(item: Any) -> str:
        if item is None:
            return ""
        if isinstance(item, str):
            return item.strip()
        if isinstance(item, (int, float, bool)):
            return str(item).strip()
        if isinstance(item, dict):
            for key in ("translation", "translated", "text", "content", "output", "target"):
                value = item.get(key)
                if isinstance(value, str):
                    return value.strip()
            values = [v.strip() for v in item.values() if isinstance(v, str) and v.strip()]
            if len(values) == 1:
                return values[0]
        raise RuntimeError("翻译结果数组元素不是字符串")

    def coerce_to_list(parsed: Any) -> List[str] | None:
        if isinstance(parsed, str):
            nested = parsed.strip()
            if nested.startswith("[") or nested.startswith("{"):
                try:
                    reparsed = json.loads(nested)
                except Exception:
                    return None
                return coerce_to_list(reparsed)
            return None
        if isinstance(parsed, list):
            return [extract_text_from_item(item) for item in parsed]
        if isinstance(parsed, dict):
            for key in ("translations", "translation", "result", "results", "output", "outputs", "items", "lines", "data"):
                candidate = parsed.get(key)
                if isinstance(candidate, list):
                    return coerce_to_list(candidate)
            if len(parsed) == 1:
                only = next(iter(parsed.values()))
                if isinstance(only, list):
                    return coerce_to_list(only)
        return None

    text = raw_text.replace("\ufeff", "").strip()
    candidates: List[str] = []
    base = strip_code_fence(text)
    if base:
        candidates.append(base)
    for open_char, close_char in (("[", "]"), ("{", "}")):
        snippet = find_balanced_json(base, open_char, close_char)
        if snippet:
            candidates.append(snippet)

    seen = set()
    for candidate in candidates:
        normalized = candidate.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        try:
            parsed = json.loads(normalized)
        except Exception:
            continue
        coerced = coerce_to_list(parsed)
        if coerced is not None:
            return coerced

    raise RuntimeError("翻译结果不是有效 JSON 数组")
