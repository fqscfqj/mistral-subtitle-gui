
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import traceback
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

try:
    from mistralai import Mistral
except Exception:
    Mistral = None

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

SUPPORTED_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS
SETTINGS_FILE = ".mistral_subtitle_gui_settings.json"
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


class TaskCancelled(Exception):
    pass


@dataclass
class TranscriptionSettings:
    api_key: str
    model: str
    language_mode: str
    language: str
    timestamp_granularity: str
    diarize: bool
    context_bias: str
    output_mode: str
    output_dir: Path
    translation_mode: str
    translation_model: str
    translation_target_language: str
    translation_openai_api_key: str
    translation_openai_base_url: str
    translation_bilingual_srt: bool
    save_srt: bool
    save_txt: bool
    save_json: bool
    ffmpeg_path: str


@dataclass
class TaskState:
    task_id: str
    source_path: Path
    row: int
    status: str = "Queued"
    progress: int = 0
    message: str = "就绪"
    outputs: Dict[str, str] = field(default_factory=dict)


class WorkerSignals(QObject):
    progress = Signal(str, str, int, str)
    finished = Signal(str, bool, str, str, dict)


class DropFrame(QFrame):
    files_dropped = Signal(list)

    def __init__(self) -> None:
        super().__init__()
        self.setAcceptDrops(True)
        self.setObjectName("dropFrame")
        layout = QVBoxLayout(self)
        self.label = QLabel("将视频/音频文件或文件夹拖拽到这里")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:  # noqa: N802
        paths: List[str] = []
        for url in event.mimeData().urls():
            local = url.toLocalFile()
            if local:
                paths.append(local)
        if paths:
            self.files_dropped.emit(paths)
        event.acceptProposedAction()


def find_ffmpeg() -> str:
    custom = os.environ.get("FFMPEG_BINARY", "").strip()
    if custom and Path(custom).exists():
        return custom
    ffmpeg = shutil.which("ffmpeg")
    return ffmpeg or ""


def format_srt_timestamp(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000.0)))
    hours = ms // 3_600_000
    ms %= 3_600_000
    minutes = ms // 60_000
    ms %= 60_000
    secs = ms // 1000
    ms %= 1000
    return f"{hours:02}:{minutes:02}:{secs:02},{ms:03}"


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


def discover_media_files(folder: Path) -> List[Path]:
    found: List[Path] = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            found.append(p)
    return sorted(found)


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


def normalize_language_code(value: str) -> str:
    cleaned = value.strip().lower().replace("_", "-")
    if not cleaned:
        return ""
    primary = cleaned.split("-", 1)[0].strip()
    token = "".join(ch for ch in primary if ch.isalnum())
    return token


def detect_language_code(payload: Dict[str, Any]) -> str:
    candidates: List[str] = []
    for key in ("language", "detected_language", "lang"):
        val = payload.get(key)
        if isinstance(val, str):
            candidates.append(val)

    meta = payload.get("metadata")
    if isinstance(meta, dict):
        for key in ("language", "detected_language", "lang"):
            val = meta.get(key)
            if isinstance(val, str):
                candidates.append(val)

    for cand in candidates:
        code = normalize_language_code(cand)
        if code:
            return code
    return "und"


def normalize_openai_base_url(base_url: str) -> str:
    url = base_url.strip()
    if not url:
        return "https://api.openai.com/v1"
    if url.endswith("/chat/completions"):
        return url
    url = url.rstrip("/")
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return f"{url}/chat/completions"


def is_chinese_language(lang_code: str) -> bool:
    norm = normalize_language_code(lang_code)
    return norm in {"zh", "zho", "chi", "cn"}


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
            str_values = [v.strip() for v in item.values() if isinstance(v, str) and v.strip()]
            if len(str_values) == 1:
                return str_values[0]
        raise RuntimeError("翻译结果数组元素不是字符串")

    def coerce_to_list(parsed: Any) -> Optional[List[str]]:
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

    array_snippet = find_balanced_json(base, "[", "]")
    if array_snippet:
        candidates.append(array_snippet)

    object_snippet = find_balanced_json(base, "{", "}")
    if object_snippet:
        candidates.append(object_snippet)

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


def call_openai_compatible_chat(
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    endpoint = normalize_openai_base_url(base_url)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"OpenAI 兼容接口返回错误: HTTP {exc.code} {detail[:240]}")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAI 兼容接口连接失败: {exc.reason}")

    response_payload = json.loads(raw)
    content = extract_chat_text(response_payload)
    if not content:
        raise RuntimeError("OpenAI 兼容接口未返回可用文本")
    return content


def call_mistral_chat(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    if Mistral is None:
        raise RuntimeError("缺少依赖：mistralai")
    client = Mistral(api_key=api_key)
    response = client.chat.complete(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    payload = normalize_response(response)
    content = extract_chat_text(payload)
    if not content:
        raise RuntimeError("Mistral 翻译接口未返回可用文本")
    return content


def translate_lines(
    lines: List[str],
    settings: TranscriptionSettings,
    transcribe_api_key: str,
    cancel_event: threading.Event,
) -> List[str]:
    if not lines:
        return []

    target_lang = settings.translation_target_language
    zh_target = is_chinese_language(target_lang)
    if zh_target:
        style_instruction = (
            "目标语言为简体中文。请采用自然口语字幕风格，避免生硬直译；"
            "中文标点规范；保留专有名词/缩写/数字单位；语气符合中文语境。"
        )
    else:
        style_instruction = "请使用地道自然的目标语言表达，避免逐词直译。"

    system_prompt = (
        "你是专业字幕翻译。"
        "你将收到一个 JSON 字符串数组，必须逐条翻译并保持数组长度和顺序完全一致。"
        "不要添加解释、不要输出 Markdown、不要输出额外字段，只返回 JSON 数组。"
        f"{style_instruction}"
    )

    result: List[str] = []
    chunk_size = 40
    max_attempts = 3

    for i in range(0, len(lines), chunk_size):
        if cancel_event.is_set():
            raise TaskCancelled("翻译前已取消")

        chunk = lines[i : i + chunk_size]
        last_error = ""
        last_content = ""
        translated_chunk: Optional[List[str]] = None

        for attempt in range(1, max_attempts + 1):
            if attempt == 1:
                user_prompt = (
                    f"请把以下字幕翻译为 `{target_lang}`。"
                    "请直接返回 JSON 数组，每个元素对应一条翻译，不得缺失。"
                    f"\n输入：{json.dumps(chunk, ensure_ascii=False)}"
                )
            else:
                user_prompt = (
                    "你上一轮输出不符合格式要求。"
                    f"请重新翻译并只返回 JSON 字符串数组，数组长度必须是 {len(chunk)}。"
                    "禁止代码块、禁止解释、禁止额外字段。"
                    f"\n输入：{json.dumps(chunk, ensure_ascii=False)}"
                    f"\n上一次输出：{last_content[:800]}"
                )

            if settings.translation_mode == "mistral":
                content = call_mistral_chat(
                    api_key=transcribe_api_key,
                    model=settings.translation_model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
            elif settings.translation_mode == "openai":
                content = call_openai_compatible_chat(
                    base_url=settings.translation_openai_base_url,
                    api_key=settings.translation_openai_api_key,
                    model=settings.translation_model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
            else:
                raise RuntimeError("未知翻译模式")

            last_content = content
            try:
                translated = parse_json_array_output(content)
                if len(translated) != len(chunk):
                    raise RuntimeError("翻译结果条数与原字幕不一致")
                translated_chunk = translated
                break
            except Exception as exc:
                last_error = str(exc).strip() or "未知错误"
                if attempt >= max_attempts:
                    preview = content.strip().replace("\n", " ")
                    if len(preview) > 220:
                        preview = preview[:220] + "..."
                    raise RuntimeError(
                        f"翻译结果格式错误，已重试 {max_attempts} 次：{last_error} | 返回片段：{preview}"
                    )

        if translated_chunk is None:
            raise RuntimeError("翻译失败：未获得有效结果")
        result.extend(translated_chunk)

    return result


def build_bilingual_srt_text(
    original_segments: List[Dict[str, Any]],
    translated_segments: List[Dict[str, Any]],
) -> str:
    if not original_segments or not translated_segments:
        return ""
    if len(original_segments) != len(translated_segments):
        raise RuntimeError("双语字幕生成失败：原文和译文段落数量不一致")

    lines: List[str] = []
    for idx, (orig, trans) in enumerate(zip(original_segments, translated_segments), start=1):
        start = format_srt_timestamp(float(orig["start"]))
        end = format_srt_timestamp(float(orig["end"]))
        speaker = orig.get("speaker")
        prefix = f"[{speaker}] " if speaker not in (None, "") else ""
        orig_text = (prefix + str(orig.get("text", "")).strip()).strip() or "..."
        trans_text = (prefix + str(trans.get("text", "")).strip()).strip() or "..."
        lines.append(f"{idx}\n{start} --> {end}\n{orig_text}\n{trans_text}\n")
    return "\n".join(lines)


def resolve_translation_base(
    target_dir: Path,
    source_stem: str,
    source_lang_code: str,
    target_lang_code: str,
) -> Path:
    if source_lang_code == target_lang_code:
        return target_dir / f"{source_stem}.{target_lang_code}.translated"
    return target_dir / f"{source_stem}.{target_lang_code}"

def transcribe_task(
    task_id: str,
    source_path: Path,
    settings: TranscriptionSettings,
    signals: WorkerSignals,
    cancel_event: threading.Event,
) -> None:
    temp_audio: Optional[Path] = None

    def report(status: str, progress: int, message: str) -> None:
        signals.progress.emit(task_id, status, progress, message)

    try:
        if cancel_event.is_set():
            raise TaskCancelled("启动前已取消")

        report("Preparing", 5, "检查输入文件")
        source_ext = source_path.suffix.lower()
        audio_path = source_path

        if source_ext in VIDEO_EXTENSIONS:
            if not settings.ffmpeg_path:
                raise RuntimeError("视频文件需要 ffmpeg")
            report("Extracting", 20, "使用 ffmpeg 提取音频")
            with tempfile.NamedTemporaryFile(prefix="mistral_sub_", suffix=".wav", delete=False) as tmp:
                temp_audio = Path(tmp.name)
            extract_audio_with_ffmpeg(settings.ffmpeg_path, source_path, temp_audio)
            audio_path = temp_audio
            report("Extracting", 40, "音频提取完成")
        else:
            report("Preparing", 35, "输入文件为音频")

        if cancel_event.is_set():
            raise TaskCancelled("转写前已取消")

        if Mistral is None:
            raise RuntimeError("缺少依赖：mistralai")

        report("Transcribing", 60, "正在调用 Mistral API")
        client = Mistral(api_key=settings.api_key)

        kwargs: Dict[str, Any] = {"model": settings.model}
        if settings.timestamp_granularity != "none":
            kwargs["timestamp_granularities"] = [settings.timestamp_granularity]
        elif settings.language_mode == "manual" and settings.language:
            kwargs["language"] = settings.language
        if settings.diarize:
            kwargs["diarize"] = True
        if settings.context_bias:
            kwargs["context_bias"] = settings.context_bias

        with open(audio_path, "rb") as f:
            response = client.audio.transcriptions.complete(
                file={"content": f, "file_name": audio_path.name},
                **kwargs,
            )

        report("Writing", 85, "正在生成转写文件")
        payload = normalize_response(response)
        text = extract_text(payload)
        segments = extract_segments(payload)
        original_segments = [dict(seg) for seg in segments]

        if cancel_event.is_set():
            raise TaskCancelled("写入前已取消")

        if settings.output_mode == "source":
            target_dir = source_path.parent
        else:
            target_dir = settings.output_dir
            target_dir.mkdir(parents=True, exist_ok=True)

        lang_code = (
            normalize_language_code(settings.language)
            if settings.language_mode == "manual"
            else detect_language_code(payload)
        )
        if not lang_code:
            lang_code = "und"

        out_base = target_dir / f"{source_path.stem}.{lang_code}"
        outputs: Dict[str, str] = {}

        if settings.save_srt:
            srt_path = out_base.with_suffix(".srt")
            srt_text = build_srt_text(segments, text)
            srt_path.write_text(srt_text, encoding="utf-8")
            outputs["srt"] = str(srt_path)

        if settings.save_txt:
            txt_path = out_base.with_suffix(".txt")
            txt_path.write_text(text or "", encoding="utf-8")
            outputs["txt"] = str(txt_path)

        if settings.save_json:
            json_path = out_base.with_suffix(".json")
            json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            outputs["json"] = str(json_path)

        if settings.translation_mode != "none":
            if cancel_event.is_set():
                raise TaskCancelled("翻译前已取消")

            report("Translating", 90, "正在翻译字幕")
            lines_for_translation = [str(seg.get("text", "")).strip() for seg in original_segments]
            translated_segments: List[Dict[str, Any]] = []
            translated_text = ""

            if lines_for_translation:
                translated_lines = translate_lines(
                    lines=lines_for_translation,
                    settings=settings,
                    transcribe_api_key=settings.api_key,
                    cancel_event=cancel_event,
                )
                translated_segments = [dict(seg) for seg in original_segments]
                for seg, translated_line in zip(translated_segments, translated_lines):
                    seg["text"] = translated_line
                translated_text = "\n".join(translated_lines).strip()
            elif text.strip():
                translated_lines = translate_lines(
                    lines=[text],
                    settings=settings,
                    transcribe_api_key=settings.api_key,
                    cancel_event=cancel_event,
                )
                translated_text = translated_lines[0] if translated_lines else ""

            report("Writing", 96, "正在写入翻译文件")
            target_lang_code = normalize_language_code(settings.translation_target_language) or "tr"
            trans_base = resolve_translation_base(
                target_dir=target_dir,
                source_stem=source_path.stem,
                source_lang_code=lang_code,
                target_lang_code=target_lang_code,
            )

            if settings.save_srt:
                trans_srt_path = trans_base.with_suffix(".srt")
                if translated_segments:
                    if settings.translation_bilingual_srt:
                        trans_srt_text = build_bilingual_srt_text(original_segments, translated_segments)
                    else:
                        trans_srt_text = build_srt_text(translated_segments, translated_text)
                else:
                    trans_srt_text = build_srt_text([], translated_text)
                trans_srt_path.write_text(trans_srt_text, encoding="utf-8")
                outputs["srt_翻译"] = str(trans_srt_path)

            if settings.save_txt:
                trans_txt_path = trans_base.with_suffix(".txt")
                trans_txt_path.write_text(translated_text or "", encoding="utf-8")
                outputs["txt_翻译"] = str(trans_txt_path)

            if settings.save_json:
                trans_json_path = trans_base.with_suffix(".json")
                trans_payload = {
                    "type": "translation",
                    "mode": settings.translation_mode,
                    "model": settings.translation_model,
                    "source_language": lang_code,
                    "target_language": target_lang_code,
                    "text": translated_text,
                    "segments": translated_segments,
                }
                trans_json_path.write_text(
                    json.dumps(trans_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                outputs["json_翻译"] = str(trans_json_path)

        report("Completed", 100, "完成")
        signals.finished.emit(task_id, True, "Completed", "完成", outputs)
    except TaskCancelled as exc:
        signals.finished.emit(task_id, False, "Cancelled", str(exc), {})
    except Exception as exc:
        message = str(exc).strip() or traceback.format_exc(limit=1)
        signals.finished.emit(task_id, False, "Failed", message, {})
    finally:
        if temp_audio and temp_audio.exists():
            try:
                temp_audio.unlink(missing_ok=True)
            except Exception:
                pass


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Mistral 字幕工作台")
        self.resize(1220, 820)

        self.signals = WorkerSignals()
        self.signals.progress.connect(self.on_task_progress)
        self.signals.finished.connect(self.on_task_finished)

        self.executor: Optional[ThreadPoolExecutor] = None
        self.cancel_event = threading.Event()

        self.tasks: Dict[str, TaskState] = {}
        self.path_to_task: Dict[str, str] = {}

        self.active_run_ids: set[str] = set()
        self.completed_run_ids: set[str] = set()
        self.run_progress: Dict[str, int] = {}
        self.futures: Dict[str, Any] = {}
        self.is_running = False

        self.init_ui()

    def init_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(10)

        tabs = QTabWidget()

        task_page = QWidget()
        task_layout = QVBoxLayout(task_page)
        task_layout.setContentsMargins(0, 0, 0, 0)
        task_layout.setSpacing(10)

        self.drop_frame = DropFrame()
        self.drop_frame.files_dropped.connect(self.on_drop_paths)
        task_layout.addWidget(self.drop_frame)

        import_row = QWidget()
        import_layout = QHBoxLayout(import_row)
        import_layout.setContentsMargins(0, 0, 0, 0)

        self.add_file_btn = QPushButton("添加文件")
        self.add_file_btn.clicked.connect(self.on_add_file)

        self.add_folder_btn = QPushButton("添加文件夹")
        self.add_folder_btn.clicked.connect(self.on_add_folder)

        self.remove_btn = QPushButton("删除所选")
        self.remove_btn.clicked.connect(self.on_remove_selected)

        self.clear_btn = QPushButton("清空列表")
        self.clear_btn.clicked.connect(self.on_clear_all)

        self.start_btn = QPushButton("开始")
        self.start_btn.clicked.connect(self.on_start)

        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self.on_stop)
        self.stop_btn.setEnabled(False)

        self.open_output_btn = QPushButton("打开输出目录")
        self.open_output_btn.clicked.connect(self.on_open_output_dir)

        import_layout.addWidget(self.add_file_btn)
        import_layout.addWidget(self.add_folder_btn)
        import_layout.addWidget(self.remove_btn)
        import_layout.addWidget(self.clear_btn)
        import_layout.addStretch(1)
        import_layout.addWidget(self.start_btn)
        import_layout.addWidget(self.stop_btn)
        import_layout.addWidget(self.open_output_btn)
        task_layout.addWidget(import_row)

        self.task_table = QTableWidget(0, 5)
        self.task_table.setHorizontalHeaderLabels(["来源文件", "状态", "进度", "输出文件", "消息"])
        self.task_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.task_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.task_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.task_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.task_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.task_table.verticalHeader().setVisible(False)
        self.task_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.task_table.setAlternatingRowColors(True)
        self.task_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        task_layout.addWidget(self.task_table)

        self.total_progress = QProgressBar()
        self.total_progress.setRange(0, 100)
        self.total_progress.setValue(0)
        self.summary_label = QLabel("当前无运行任务")
        task_layout.addWidget(self.total_progress)
        task_layout.addWidget(self.summary_label)

        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFixedHeight(140)
        task_layout.addWidget(self.log_text)

        settings_page = QWidget()
        settings_page_layout = QVBoxLayout(settings_page)
        settings_page_layout.setContentsMargins(0, 0, 0, 0)
        settings_page_layout.setSpacing(10)

        settings_group = QGroupBox("Mistral API 与输出设置")
        settings_layout = QGridLayout(settings_group)

        self.api_key_input = QLineEdit(os.environ.get("MISTRAL_API_KEY", ""))
        self.api_key_input.setPlaceholderText("请输入 MISTRAL_API_KEY")
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.show_key_checkbox = QCheckBox("显示")
        self.show_key_checkbox.toggled.connect(self.on_toggle_show_key)

        api_key_row = QWidget()
        api_key_row_layout = QHBoxLayout(api_key_row)
        api_key_row_layout.setContentsMargins(0, 0, 0, 0)
        api_key_row_layout.addWidget(self.api_key_input)
        api_key_row_layout.addWidget(self.show_key_checkbox)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.addItems(["voxtral-mini-latest", "voxtral-small-latest"])

        self.language_mode_combo = QComboBox()
        self.language_mode_combo.addItems(["自动识别", "指定语言"])
        self.language_mode_combo.currentIndexChanged.connect(self.on_language_mode_changed)

        self.language_input = QLineEdit("zh")
        self.language_input.setPlaceholderText("语言代码，例如 zh / en")

        self.timestamp_combo = QComboBox()
        # timestamp granularity determines whether the API returns segments with start/end times.
        # default to "segment" so that users normally get timecodes without needing to change it.
        self.timestamp_combo.addItems(["none", "segment", "word"])
        self.timestamp_combo.setCurrentText("segment")

        self.diarize_checkbox = QCheckBox("启用说话人分离")

        self.thread_spin = QSpinBox()
        self.thread_spin.setRange(1, 16)
        self.thread_spin.setValue(3)

        self.output_mode_combo = QComboBox()
        self.output_mode_combo.addItems(["输出到原文件目录", "输出到指定目录"])
        self.output_mode_combo.currentIndexChanged.connect(self.on_output_mode_changed)

        self.output_dir_input = QLineEdit(str(Path.cwd() / "subtitles"))
        self.output_btn = QPushButton("浏览")
        self.output_btn.clicked.connect(self.on_choose_output_dir)

        output_row = QWidget()
        output_row_layout = QHBoxLayout(output_row)
        output_row_layout.setContentsMargins(0, 0, 0, 0)
        output_row_layout.addWidget(self.output_dir_input)
        output_row_layout.addWidget(self.output_btn)

        self.context_bias_input = QPlainTextEdit()
        self.context_bias_input.setPlaceholderText("可选词条，使用逗号或换行分隔")
        self.context_bias_input.setFixedHeight(64)

        self.save_srt_checkbox = QCheckBox("保存 .srt")
        self.save_srt_checkbox.setChecked(True)
        self.save_txt_checkbox = QCheckBox("保存 .txt")
        self.save_txt_checkbox.setChecked(True)
        self.save_json_checkbox = QCheckBox("保存 .json")

        format_row = QWidget()
        format_row_layout = QHBoxLayout(format_row)
        format_row_layout.setContentsMargins(0, 0, 0, 0)
        format_row_layout.addWidget(self.save_srt_checkbox)
        format_row_layout.addWidget(self.save_txt_checkbox)
        format_row_layout.addWidget(self.save_json_checkbox)
        format_row_layout.addStretch(1)

        translation_group = QGroupBox("字幕翻译设置")
        translation_layout = QGridLayout(translation_group)

        self.translation_mode_combo = QComboBox()
        self.translation_mode_combo.addItems(
            ["不翻译", "Mistral API 翻译", "OpenAI 兼容 API 翻译"]
        )
        self.translation_mode_combo.currentIndexChanged.connect(self.on_translation_mode_changed)

        self.translation_target_input = QLineEdit("zh")
        self.translation_target_input.setPlaceholderText("目标语言代码，例如 zh / en / ja")

        self.translation_model_input = QLineEdit("mistral-small-latest")
        self.translation_model_input.setPlaceholderText("翻译模型名称")

        self.translation_bilingual_checkbox = QCheckBox("SRT 输出双语（原文 + 译文）")
        self.translation_bilingual_checkbox.setChecked(True)

        self.translation_openai_base_input = QLineEdit("https://api.openai.com/v1")
        self.translation_openai_base_input.setPlaceholderText(
            "OpenAI 兼容地址，例如 https://api.openai.com/v1"
        )

        self.translation_openai_key_input = QLineEdit(os.environ.get("OPENAI_API_KEY", ""))
        self.translation_openai_key_input.setPlaceholderText("OpenAI/第三方兼容 API Key")
        self.translation_openai_key_input.setEchoMode(QLineEdit.EchoMode.Password)

        self.show_openai_key_checkbox = QCheckBox("显示")
        self.show_openai_key_checkbox.toggled.connect(self.on_toggle_show_openai_key)

        openai_key_row = QWidget()
        openai_key_row_layout = QHBoxLayout(openai_key_row)
        openai_key_row_layout.setContentsMargins(0, 0, 0, 0)
        openai_key_row_layout.addWidget(self.translation_openai_key_input)
        openai_key_row_layout.addWidget(self.show_openai_key_checkbox)

        translation_layout.addWidget(QLabel("翻译模式"), 0, 0)
        translation_layout.addWidget(self.translation_mode_combo, 0, 1)
        translation_layout.addWidget(QLabel("目标语言"), 1, 0)
        translation_layout.addWidget(self.translation_target_input, 1, 1)
        translation_layout.addWidget(QLabel("翻译模型"), 2, 0)
        translation_layout.addWidget(self.translation_model_input, 2, 1)
        translation_layout.addWidget(self.translation_bilingual_checkbox, 3, 0, 1, 2)
        translation_layout.addWidget(QLabel("OpenAI 兼容 Base URL"), 4, 0)
        translation_layout.addWidget(self.translation_openai_base_input, 4, 1)
        translation_layout.addWidget(QLabel("OpenAI 兼容 API Key"), 5, 0)
        translation_layout.addWidget(openai_key_row, 5, 1)

        settings_layout.addWidget(QLabel("API 密钥"), 0, 0)
        settings_layout.addWidget(api_key_row, 0, 1)
        settings_layout.addWidget(QLabel("模型"), 1, 0)
        settings_layout.addWidget(self.model_combo, 1, 1)
        settings_layout.addWidget(QLabel("语言模式"), 2, 0)
        settings_layout.addWidget(self.language_mode_combo, 2, 1)
        settings_layout.addWidget(QLabel("指定语言"), 3, 0)
        settings_layout.addWidget(self.language_input, 3, 1)
        settings_layout.addWidget(QLabel("时间戳粒度"), 4, 0)
        settings_layout.addWidget(self.timestamp_combo, 4, 1)
        settings_layout.addWidget(QLabel("最大线程数"), 5, 0)
        settings_layout.addWidget(self.thread_spin, 5, 1)
        settings_layout.addWidget(QLabel("输出目录模式"), 6, 0)
        settings_layout.addWidget(self.output_mode_combo, 6, 1)
        settings_layout.addWidget(QLabel("指定输出目录"), 7, 0)
        settings_layout.addWidget(output_row, 7, 1)
        settings_layout.addWidget(QLabel("上下文偏置"), 8, 0)
        settings_layout.addWidget(self.context_bias_input, 8, 1)
        settings_layout.addWidget(self.diarize_checkbox, 9, 0, 1, 2)
        settings_layout.addWidget(format_row, 10, 0, 1, 2)

        settings_page_layout.addWidget(settings_group)
        settings_page_layout.addWidget(translation_group)

        settings_action_row = QWidget()
        settings_action_layout = QHBoxLayout(settings_action_row)
        settings_action_layout.setContentsMargins(0, 0, 0, 0)
        settings_action_layout.addStretch(1)
        self.save_settings_btn = QPushButton("保存设置")
        self.save_settings_btn.clicked.connect(self.on_save_settings)
        settings_action_layout.addWidget(self.save_settings_btn)
        settings_page_layout.addWidget(settings_action_row)

        settings_page_layout.addStretch(1)

        tabs.addTab(task_page, "任务")
        tabs.addTab(settings_page, "设置")

        root_layout.addWidget(tabs)

        self.setCentralWidget(root)
        self.load_settings_from_disk()
        self.on_language_mode_changed()
        self.on_output_mode_changed()
        self.on_translation_mode_changed()
        self.apply_style()

    def apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow {
                background: #eef3f8;
            }
            QGroupBox {
                border: 1px solid #b8c7d9;
                border-radius: 10px;
                margin-top: 10px;
                font-weight: 600;
                background: #f9fbfd;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
                color: #1e2d3d;
            }
            #dropFrame {
                border: 2px dashed #4f83c2;
                border-radius: 10px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #f4f8fc, stop:1 #e1ecf8);
                min-height: 84px;
            }
            QLabel {
                color: #1c2b39;
            }
            QPushButton {
                background: #2e78c7;
                color: white;
                border: none;
                border-radius: 7px;
                padding: 7px 11px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #266ab1;
            }
            QPushButton:disabled {
                background: #9db5cc;
                color: #ebf2f9;
            }
            QLineEdit, QPlainTextEdit, QComboBox, QSpinBox {
                border: 1px solid #b4c4d4;
                border-radius: 6px;
                padding: 5px;
                background: white;
            }
            QTableWidget {
                border: 1px solid #b8c7d9;
                border-radius: 8px;
                background: white;
                alternate-background-color: #f4f7fb;
            }
            QHeaderView::section {
                background: #d9e6f3;
                padding: 6px;
                border: none;
                border-right: 1px solid #bfd0e1;
                color: #1a2a3a;
                font-weight: 600;
            }
            QProgressBar {
                border: 1px solid #9eb4c8;
                border-radius: 6px;
                text-align: center;
                background: #f4f8fc;
            }
            QProgressBar::chunk {
                background: #3f96dd;
                border-radius: 5px;
            }
            """
        )

        base_font = QFont("Segoe UI", 10)
        self.setFont(base_font)

    def log(self, message: str) -> None:
        self.log_text.appendPlainText(message)

    def settings_file_path(self) -> Path:
        return Path.cwd() / SETTINGS_FILE

    def collect_ui_settings(self) -> Dict[str, Any]:
        return {
            "api_key": self.api_key_input.text().strip(),
            "model": self.model_combo.currentText().strip(),
            "language_mode_index": self.language_mode_combo.currentIndex(),
            "language": self.language_input.text().strip(),
            "timestamp": self.timestamp_combo.currentText().strip(),
            "diarize": self.diarize_checkbox.isChecked(),
            "thread_count": self.thread_spin.value(),
            "output_mode_index": self.output_mode_combo.currentIndex(),
            "output_dir": self.output_dir_input.text().strip(),
            "context_bias": self.context_bias_input.toPlainText(),
            "save_srt": self.save_srt_checkbox.isChecked(),
            "save_txt": self.save_txt_checkbox.isChecked(),
            "save_json": self.save_json_checkbox.isChecked(),
            "translation_mode_index": self.translation_mode_combo.currentIndex(),
            "translation_target": self.translation_target_input.text().strip(),
            "translation_model": self.translation_model_input.text().strip(),
            "translation_bilingual": self.translation_bilingual_checkbox.isChecked(),
            "translation_openai_base": self.translation_openai_base_input.text().strip(),
            "translation_openai_key": self.translation_openai_key_input.text().strip(),
        }

    def apply_ui_settings(self, data: Dict[str, Any]) -> None:
        if not isinstance(data, dict):
            return

        self.api_key_input.setText(str(data.get("api_key", self.api_key_input.text())))

        model = str(data.get("model", self.model_combo.currentText())).strip()
        if model:
            self.model_combo.setCurrentText(model)

        language_mode_index = int(data.get("language_mode_index", self.language_mode_combo.currentIndex()))
        self.language_mode_combo.setCurrentIndex(0 if language_mode_index not in (0, 1) else language_mode_index)

        self.language_input.setText(str(data.get("language", self.language_input.text())))

        timestamp = str(data.get("timestamp", self.timestamp_combo.currentText())).strip()
        ts_index = self.timestamp_combo.findText(timestamp)
        self.timestamp_combo.setCurrentIndex(ts_index if ts_index >= 0 else 0)

        self.diarize_checkbox.setChecked(bool(data.get("diarize", self.diarize_checkbox.isChecked())))

        thread_count = int(data.get("thread_count", self.thread_spin.value()))
        thread_count = max(self.thread_spin.minimum(), min(self.thread_spin.maximum(), thread_count))
        self.thread_spin.setValue(thread_count)

        output_mode_index = int(data.get("output_mode_index", self.output_mode_combo.currentIndex()))
        self.output_mode_combo.setCurrentIndex(0 if output_mode_index not in (0, 1) else output_mode_index)

        self.output_dir_input.setText(str(data.get("output_dir", self.output_dir_input.text())))
        self.context_bias_input.setPlainText(str(data.get("context_bias", self.context_bias_input.toPlainText())))

        self.save_srt_checkbox.setChecked(bool(data.get("save_srt", self.save_srt_checkbox.isChecked())))
        self.save_txt_checkbox.setChecked(bool(data.get("save_txt", self.save_txt_checkbox.isChecked())))
        self.save_json_checkbox.setChecked(bool(data.get("save_json", self.save_json_checkbox.isChecked())))

        translation_mode_index = int(data.get("translation_mode_index", self.translation_mode_combo.currentIndex()))
        self.translation_mode_combo.setCurrentIndex(
            0 if translation_mode_index not in (0, 1, 2) else translation_mode_index
        )
        self.translation_target_input.setText(
            str(data.get("translation_target", self.translation_target_input.text()))
        )
        self.translation_model_input.setText(
            str(data.get("translation_model", self.translation_model_input.text()))
        )
        self.translation_bilingual_checkbox.setChecked(
            bool(data.get("translation_bilingual", self.translation_bilingual_checkbox.isChecked()))
        )
        self.translation_openai_base_input.setText(
            str(data.get("translation_openai_base", self.translation_openai_base_input.text()))
        )
        self.translation_openai_key_input.setText(
            str(data.get("translation_openai_key", self.translation_openai_key_input.text()))
        )

    def on_save_settings(self) -> None:
        path = self.settings_file_path()
        try:
            data = self.collect_ui_settings()
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            self.log(f"设置已保存到：{path}")
            QMessageBox.information(self, "保存成功", f"设置已保存到：\n{path}")
        except Exception as exc:
            QMessageBox.warning(self, "保存失败", f"无法保存设置：{exc}")

    def load_settings_from_disk(self) -> None:
        path = self.settings_file_path()
        if not path.exists():
            return
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
            self.apply_ui_settings(data)
            self.log(f"已加载本地设置：{path}")
        except Exception as exc:
            self.log(f"加载设置失败（已忽略）：{exc}")

    def on_toggle_show_key(self, checked: bool) -> None:
        mode = QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
        self.api_key_input.setEchoMode(mode)

    def on_toggle_show_openai_key(self, checked: bool) -> None:
        mode = QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
        self.translation_openai_key_input.setEchoMode(mode)

    def on_translation_mode_changed(self) -> None:
        mode = self.translation_mode_combo.currentIndex()
        enable_translation = mode != 0
        use_openai_compatible = mode == 2
        current_model = self.translation_model_input.text().strip()

        self.translation_target_input.setEnabled(enable_translation)
        self.translation_model_input.setEnabled(enable_translation)
        self.translation_bilingual_checkbox.setEnabled(enable_translation)
        self.translation_openai_base_input.setEnabled(use_openai_compatible)
        self.translation_openai_key_input.setEnabled(use_openai_compatible)
        self.show_openai_key_checkbox.setEnabled(use_openai_compatible)

        if mode == 1 and current_model in {"", "gpt-4o-mini"}:
            self.translation_model_input.setText("mistral-small-latest")
        if mode == 2 and current_model in {"", "mistral-small-latest"}:
            self.translation_model_input.setText("gpt-4o-mini")

    def on_language_mode_changed(self) -> None:
        manual = self.language_mode_combo.currentIndex() == 1
        self.language_input.setEnabled(manual)
        if manual:
            self.language_input.setPlaceholderText("语言代码，例如 zh / en")
        else:
            self.language_input.setPlaceholderText("自动识别时无需填写")

    def on_output_mode_changed(self) -> None:
        custom = self.output_mode_combo.currentIndex() == 1
        self.output_dir_input.setEnabled(custom)
        self.output_btn.setEnabled(custom)

    def on_choose_output_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "选择输出目录", self.output_dir_input.text())
        if directory:
            self.output_dir_input.setText(directory)

    def on_drop_paths(self, paths: List[str]) -> None:
        self.add_paths(paths)

    def on_add_file(self) -> None:
        filters = (
            "媒体文件 (*.mp4 *.mov *.mkv *.avi *.wmv *.webm *.m4v *.flv *.ts "
            "*.mp3 *.wav *.m4a *.aac *.flac *.ogg *.opus *.wma)"
        )
        files, _ = QFileDialog.getOpenFileNames(self, "选择媒体文件", str(Path.cwd()), filters)
        if files:
            self.add_paths(files)

    def on_add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择文件夹", str(Path.cwd()))
        if folder:
            self.add_paths([folder])

    def normalize_path_key(self, path: Path) -> str:
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path.absolute()
        return str(resolved).casefold()

    def add_paths(self, raw_paths: List[str]) -> None:
        discovered: List[Path] = []
        for raw in raw_paths:
            p = Path(raw)
            if p.is_dir():
                discovered.extend(discover_media_files(p))
            elif p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                discovered.append(p)

        if not discovered:
            self.log("未找到支持的媒体文件")
            return

        added = 0
        for path in discovered:
            key = self.normalize_path_key(path)
            if key in self.path_to_task:
                continue

            task_id = uuid.uuid4().hex
            row = self.task_table.rowCount()
            self.task_table.insertRow(row)

            source_item = QTableWidgetItem(str(path))
            status_item = QTableWidgetItem("排队中")
            output_item = QTableWidgetItem("-")
            message_item = QTableWidgetItem("就绪")

            progress_bar = QProgressBar()
            progress_bar.setRange(0, 100)
            progress_bar.setValue(0)

            self.task_table.setItem(row, 0, source_item)
            self.task_table.setItem(row, 1, status_item)
            self.task_table.setCellWidget(row, 2, progress_bar)
            self.task_table.setItem(row, 3, output_item)
            self.task_table.setItem(row, 4, message_item)

            self.tasks[task_id] = TaskState(task_id=task_id, source_path=path, row=row)
            self.path_to_task[key] = task_id
            added += 1

        self.log(f"已添加 {added} 个文件")
        self.update_summary_text()

    def on_remove_selected(self) -> None:
        if self.is_running:
            QMessageBox.information(self, "任务进行中", "任务运行时无法删除行")
            return

        rows = sorted({idx.row() for idx in self.task_table.selectionModel().selectedRows()}, reverse=True)
        if not rows:
            return

        remove_ids: List[str] = []
        for task_id, state in self.tasks.items():
            if state.row in rows:
                remove_ids.append(task_id)

        for row in rows:
            self.task_table.removeRow(row)

        for task_id in remove_ids:
            path_key = self.normalize_path_key(self.tasks[task_id].source_path)
            self.path_to_task.pop(path_key, None)
            self.tasks.pop(task_id, None)

        self.rebuild_row_mapping()
        self.log(f"已删除 {len(rows)} 行所选任务")
        self.update_summary_text()

    def on_clear_all(self) -> None:
        if self.is_running:
            QMessageBox.information(self, "任务进行中", "请先停止运行中的任务再清空")
            return
        self.task_table.setRowCount(0)
        self.tasks.clear()
        self.path_to_task.clear()
        self.active_run_ids.clear()
        self.completed_run_ids.clear()
        self.run_progress.clear()
        self.total_progress.setValue(0)
        self.update_summary_text()
        self.log("已清空所有任务")

    def rebuild_row_mapping(self) -> None:
        path_to_row: Dict[str, int] = {}
        for row in range(self.task_table.rowCount()):
            item = self.task_table.item(row, 0)
            if not item:
                continue
            key = self.normalize_path_key(Path(item.text()))
            path_to_row[key] = row

        for task in self.tasks.values():
            key = self.normalize_path_key(task.source_path)
            if key in path_to_row:
                task.row = path_to_row[key]

    def collect_settings(self) -> Optional[TranscriptionSettings]:
        api_key = self.api_key_input.text().strip()
        if not api_key:
            QMessageBox.warning(self, "缺少 API 密钥", "请输入 MISTRAL_API_KEY")
            return None

        if Mistral is None:
            QMessageBox.warning(
                self,
                "缺少依赖",
                "`mistralai` 未安装，请先安装 requirements.txt 中的依赖。",
            )
            return None

        model = self.model_combo.currentText().strip() or "voxtral-mini-latest"
        language_mode = "manual" if self.language_mode_combo.currentIndex() == 1 else "auto"
        language = normalize_language_code(self.language_input.text().strip())
        timestamp = self.timestamp_combo.currentText().strip() or "none"
        diarize = self.diarize_checkbox.isChecked()
        context_bias = parse_context_bias(self.context_bias_input.toPlainText())

        if language_mode == "manual" and not language:
            QMessageBox.warning(self, "语言设置无效", "已选择指定语言，请填写有效语言代码，例如 zh / en")
            return None

        if timestamp != "none" and language_mode == "manual":
            self.log("启用时间戳粒度后，language 参数将被忽略")

        output_mode = "custom" if self.output_mode_combo.currentIndex() == 1 else "source"
        output_dir = Path(self.output_dir_input.text().strip() or str(Path.cwd() / "subtitles"))
        if output_mode == "custom":
            output_dir.mkdir(parents=True, exist_ok=True)

        translation_mode_index = self.translation_mode_combo.currentIndex()
        translation_mode = {0: "none", 1: "mistral", 2: "openai"}.get(translation_mode_index, "none")
        translation_model = self.translation_model_input.text().strip()
        translation_target_language = normalize_language_code(self.translation_target_input.text().strip())
        translation_openai_base_url = self.translation_openai_base_input.text().strip() or "https://api.openai.com/v1"
        translation_openai_api_key = self.translation_openai_key_input.text().strip()
        translation_bilingual_srt = self.translation_bilingual_checkbox.isChecked()

        if translation_mode != "none":
            if not translation_target_language:
                QMessageBox.warning(self, "翻译设置无效", "请填写目标语言代码，例如 zh / en / ja")
                return None
            if not translation_model:
                QMessageBox.warning(self, "翻译设置无效", "请填写翻译模型名称")
                return None

        if translation_mode == "openai" and not translation_openai_api_key:
            QMessageBox.warning(self, "翻译设置无效", "OpenAI 兼容翻译模式需要填写 API Key")
            return None

        save_srt = self.save_srt_checkbox.isChecked()
        save_txt = self.save_txt_checkbox.isChecked()
        save_json = self.save_json_checkbox.isChecked()

        if not (save_srt or save_txt or save_json):
            QMessageBox.warning(self, "未选择输出格式", "请至少选择一种输出格式")
            return None

        ffmpeg_path = find_ffmpeg()
        return TranscriptionSettings(
            api_key=api_key,
            model=model,
            language_mode=language_mode,
            language=language,
            timestamp_granularity=timestamp,
            diarize=diarize,
            context_bias=context_bias,
            output_mode=output_mode,
            output_dir=output_dir,
            translation_mode=translation_mode,
            translation_model=translation_model,
            translation_target_language=translation_target_language,
            translation_openai_api_key=translation_openai_api_key,
            translation_openai_base_url=translation_openai_base_url,
            translation_bilingual_srt=translation_bilingual_srt,
            save_srt=save_srt,
            save_txt=save_txt,
            save_json=save_json,
            ffmpeg_path=ffmpeg_path,
        )

    def on_start(self) -> None:
        if self.is_running:
            return
        if not self.tasks:
            QMessageBox.information(self, "没有任务", "请先添加文件")
            return

        settings = self.collect_settings()
        if settings is None:
            return

        run_ids = [
            task_id
            for task_id, task in self.tasks.items()
            if task.status in {"Queued", "Failed", "Cancelled"}
        ]
        if not run_ids:
            QMessageBox.information(self, "没有可执行任务", "当前没有可运行的排队/失败/取消任务")
            return

        has_video = any(self.tasks[task_id].source_path.suffix.lower() in VIDEO_EXTENSIONS for task_id in run_ids)
        if has_video and not settings.ffmpeg_path:
            QMessageBox.warning(
                self,
                "缺少 ffmpeg",
                "视频任务需要在 PATH 中提供 ffmpeg（或设置 FFMPEG_BINARY）",
            )
            return

        if settings.output_mode == "custom":
            settings.output_dir.mkdir(parents=True, exist_ok=True)

        max_workers = self.thread_spin.value()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.cancel_event.clear()

        self.active_run_ids = set(run_ids)
        self.completed_run_ids.clear()
        self.run_progress = {task_id: 0 for task_id in run_ids}
        self.futures.clear()

        self.is_running = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.remove_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)

        for task_id in run_ids:
            task = self.tasks[task_id]
            self.update_task_row(task_id, "Queued", 0, "等待执行")
            future = self.executor.submit(
                transcribe_task,
                task_id,
                task.source_path,
                settings,
                self.signals,
                self.cancel_event,
            )
            self.futures[task_id] = future

        self.log(f"已启动 {len(run_ids)} 个任务，线程数：{max_workers}")
        self.update_total_progress()
        self.update_summary_text()

    def on_stop(self) -> None:
        if not self.is_running:
            return

        self.cancel_event.set()
        canceled_count = 0
        for task_id, future in self.futures.items():
            if task_id in self.completed_run_ids:
                continue
            if future.cancel():
                canceled_count += 1
                self.mark_task_done(task_id, False, "Cancelled", "启动前已取消", {})

        self.log(f"已请求停止，取消了 {canceled_count} 个排队任务")

    def on_open_output_dir(self) -> None:
        if self.output_mode_combo.currentIndex() == 0:
            selected_rows = self.task_table.selectionModel().selectedRows()
            if selected_rows:
                row = selected_rows[0].row()
                item = self.task_table.item(row, 0)
                folder = Path(item.text()).parent if item else Path.cwd()
            elif self.tasks:
                first_task = next(iter(self.tasks.values()))
                folder = first_task.source_path.parent
            else:
                folder = Path.cwd()
        else:
            folder = Path(self.output_dir_input.text().strip() or str(Path.cwd() / "subtitles"))
            folder.mkdir(parents=True, exist_ok=True)

        if sys.platform.startswith("win"):
            os.startfile(str(folder))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(folder)])
        else:
            subprocess.Popen(["xdg-open", str(folder)])

    def update_task_row(self, task_id: str, status: str, progress: int, message: str, outputs: str = "-") -> None:
        task = self.tasks.get(task_id)
        if not task:
            return

        task.status = status
        task.progress = progress
        task.message = message
        row = task.row

        status_item = self.task_table.item(row, 1)
        if status_item:
            status_item.setText(STATUS_LABELS.get(status, status))

        progress_bar = self.task_table.cellWidget(row, 2)
        if isinstance(progress_bar, QProgressBar):
            progress_bar.setValue(progress)

        output_item = self.task_table.item(row, 3)
        if output_item:
            output_item.setText(outputs)

        msg_item = self.task_table.item(row, 4)
        if msg_item:
            msg_item.setText(message)

    def on_task_progress(self, task_id: str, status: str, progress: int, message: str) -> None:
        if task_id not in self.active_run_ids:
            return
        self.run_progress[task_id] = max(0, min(100, progress))
        self.update_task_row(task_id, status, progress, message)
        self.update_total_progress()

    def on_task_finished(self, task_id: str, success: bool, status: str, message: str, outputs: dict) -> None:
        self.mark_task_done(task_id, success, status, message, outputs)

    def mark_task_done(self, task_id: str, success: bool, status: str, message: str, outputs: dict) -> None:
        if task_id not in self.active_run_ids:
            return
        if task_id in self.completed_run_ids:
            return

        self.completed_run_ids.add(task_id)
        self.run_progress[task_id] = 100 if success else self.run_progress.get(task_id, 0)

        output_text = " | ".join(outputs.values()) if outputs else "-"
        display_message = message
        if status == "Failed" and len(display_message) > 180:
            display_message = display_message[:180] + "..."

        final_progress = 100 if success else max(0, self.run_progress.get(task_id, 0))
        self.update_task_row(task_id, status, final_progress, display_message, output_text)

        if success:
            self.log(f"[{task_id[:8]}] 已完成: {self.tasks[task_id].source_path.name}")
        else:
            self.log(
                f"[{task_id[:8]}] {STATUS_LABELS.get(status, status)}: "
                f"{self.tasks[task_id].source_path.name} | {message}"
            )

        self.update_total_progress()
        self.update_summary_text()

        if len(self.completed_run_ids) >= len(self.active_run_ids):
            self.finish_run()

    def finish_run(self) -> None:
        finished_ids = list(self.active_run_ids)
        if self.executor:
            self.executor.shutdown(wait=False, cancel_futures=False)
            self.executor = None

        self.is_running = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.remove_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)

        success_count = 0
        failed_count = 0
        canceled_count = 0
        for task_id in finished_ids:
            status = self.tasks[task_id].status
            if status == "Completed":
                success_count += 1
            elif status == "Cancelled":
                canceled_count += 1
            else:
                failed_count += 1

        self.log(f"任务结束：成功={success_count}，失败={failed_count}，取消={canceled_count}")
        self.active_run_ids.clear()
        self.completed_run_ids.clear()
        self.run_progress.clear()
        self.futures.clear()
        self.update_total_progress()
        self.update_summary_text()

    def update_total_progress(self) -> None:
        if not self.active_run_ids:
            self.total_progress.setValue(0)
            return

        total = 0
        for task_id in self.active_run_ids:
            total += self.run_progress.get(task_id, 0)
        value = int(round(total / len(self.active_run_ids)))
        self.total_progress.setValue(max(0, min(100, value)))

    def update_summary_text(self) -> None:
        total = len(self.tasks)
        if total == 0:
            self.summary_label.setText("暂无任务")
            return

        if not self.active_run_ids:
            queued = sum(1 for t in self.tasks.values() if t.status == "Queued")
            done = sum(1 for t in self.tasks.values() if t.status == "Completed")
            failed = sum(1 for t in self.tasks.values() if t.status == "Failed")
            canceled = sum(1 for t in self.tasks.values() if t.status == "Cancelled")
            self.summary_label.setText(
                f"总数={total} | 排队={queued} | 完成={done} | 失败={failed} | 取消={canceled}"
            )
            return

        running = len(self.active_run_ids) - len(self.completed_run_ids)
        self.summary_label.setText(
            f"当前批次：已完成 {len(self.completed_run_ids)}/{len(self.active_run_ids)} | 运行中={running}"
        )


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
