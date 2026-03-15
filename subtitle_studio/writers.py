from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import AppSettings
from .utils import build_lrc_text, build_srt_text, format_srt_timestamp, normalize_language_code


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


def write_transcription_outputs(
    target_dir: Path,
    source_stem: str,
    lang_code: str,
    save_srt: bool,
    save_lrc: bool,
    save_txt: bool,
    save_json: bool,
    is_audio_input: bool,
    payload: Dict[str, Any],
    segments: List[Dict[str, Any]],
    text: str,
) -> Dict[str, str]:
    out_base = target_dir / f"{source_stem}.{lang_code}"
    outputs: Dict[str, str] = {}
    if save_srt:
        path = out_base.with_suffix(".srt")
        path.write_text(build_srt_text(segments, text), encoding="utf-8")
        outputs["srt"] = str(path)
    if save_lrc and is_audio_input:
        path = out_base.with_suffix(".lrc")
        path.write_text(build_lrc_text(segments, text), encoding="utf-8")
        outputs["lrc"] = str(path)
    if save_txt:
        path = out_base.with_suffix(".txt")
        path.write_text(text or "", encoding="utf-8")
        outputs["txt"] = str(path)
    if save_json:
        path = out_base.with_suffix(".json")
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        outputs["json"] = str(path)
    return outputs


def write_translation_outputs(
    target_dir: Path,
    source_stem: str,
    source_lang_code: str,
    settings: AppSettings,
    original_segments: List[Dict[str, Any]],
    translated_segments: List[Dict[str, Any]],
    original_text: str,
    translated_text: str,
    source_path: Optional[Path] = None,
    write_lrc: bool = False,
) -> Dict[str, str]:
    outputs: Dict[str, str] = {}
    target_lang_code = normalize_language_code(settings.translation.target_language) or "tr"
    trans_base = target_dir / f"{source_stem}.{target_lang_code}"
    orig_base = target_dir / f"{source_stem}.orig"

    def with_output_ext(base: Path, extension: str) -> Path:
        return base.with_name(f"{base.name}{extension}")

    def same_path(left: Path, right: Path) -> bool:
        try:
            left_resolved = left.resolve()
            right_resolved = right.resolve()
        except Exception:
            left_resolved = left.absolute()
            right_resolved = right.absolute()
        return str(left_resolved).casefold() == str(right_resolved).casefold()

    def avoid_overwrite(path: Path) -> Path:
        base_path = path
        attempt = 0
        while True:
            conflict_with_source = source_path is not None and same_path(path, source_path)
            if not conflict_with_source and not path.exists():
                return path
            attempt += 1
            suffix = ".translated" if attempt == 1 else f".translated{attempt}"
            path = base_path.with_name(f"{base_path.stem}{suffix}{base_path.suffix}")

    if settings.output.save_srt:
        trans_srt_path = avoid_overwrite(with_output_ext(trans_base, ".srt"))
        if translated_segments:
            if settings.translation.bilingual_srt:
                trans_srt_text = build_bilingual_srt_text(original_segments, translated_segments)
            else:
                trans_srt_text = build_srt_text(translated_segments, translated_text)
        else:
            trans_srt_text = build_srt_text([], translated_text)
        trans_srt_path.write_text(trans_srt_text, encoding="utf-8")
        outputs["srt_翻译"] = str(trans_srt_path)
        if settings.translation.keep_original_srt:
            orig_srt_path = avoid_overwrite(with_output_ext(orig_base, ".srt"))
            orig_srt_path.write_text(build_srt_text(original_segments, original_text), encoding="utf-8")
            outputs["srt_原文"] = str(orig_srt_path)

    if write_lrc:
        trans_lrc_path = avoid_overwrite(with_output_ext(trans_base, ".lrc"))
        trans_lrc_path.write_text(build_lrc_text(translated_segments, translated_text), encoding="utf-8")
        outputs["lrc_翻译"] = str(trans_lrc_path)

    if settings.output.save_txt:
        trans_txt_path = avoid_overwrite(with_output_ext(trans_base, ".txt"))
        trans_txt_path.write_text(translated_text or "", encoding="utf-8")
        outputs["txt_翻译"] = str(trans_txt_path)

    if settings.output.save_json:
        trans_json_path = avoid_overwrite(with_output_ext(trans_base, ".json"))
        trans_payload = {
            "type": "translation",
            "mode": settings.translation.mode,
            "model": settings.translation.model,
            "source_language": source_lang_code,
            "target_language": target_lang_code,
            "text": translated_text,
            "segments": translated_segments,
        }
        trans_json_path.write_text(json.dumps(trans_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        outputs["json_翻译"] = str(trans_json_path)

    return outputs
