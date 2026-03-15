from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Event
from typing import Any, Callable, Dict, List, Optional, Protocol

from ..http_client import HttpClient
from ..models import TranslationProvider, TranslationRequest
from ..utils import extract_chat_text, is_chinese_language, normalize_response, parse_json_array_output

try:
    from mistralai import Mistral
except Exception:
    Mistral = None


def normalize_chat_completions_url(base_url: str) -> str:
    url = base_url.strip()
    if not url:
        return "https://api.openai.com/v1/chat/completions"
    if url.endswith("/chat/completions"):
        return url
    url = url.rstrip("/")
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return f"{url}/chat/completions"


class ChatCompletionBackend(Protocol):
    def complete(self, model: str, system_prompt: str, user_prompt: str) -> str:
        ...


class OpenAICompatibleChatBackend(ChatCompletionBackend):
    def __init__(self, base_url: str, api_key: str, http_client: Optional[HttpClient] = None) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.http_client = http_client or HttpClient()

    def complete(self, model: str, system_prompt: str, user_prompt: str) -> str:
        response = self.http_client.post_json(
            normalize_chat_completions_url(self.base_url),
            payload={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
            },
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"OpenAI 兼容接口返回错误: HTTP {response.status_code} {response.text[:240]}"
            )
        payload = response.payload if isinstance(response.payload, dict) else {}
        content = extract_chat_text(payload)
        if not content:
            raise RuntimeError("OpenAI 兼容接口未返回可用文本")
        return content


class MistralChatBackend(ChatCompletionBackend):
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def complete(self, model: str, system_prompt: str, user_prompt: str) -> str:
        if Mistral is None:
            raise RuntimeError("缺少依赖：mistralai")
        client = Mistral(api_key=self.api_key)
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


class StructuredSubtitleTranslationProvider(TranslationProvider):
    def __init__(self, backend: ChatCompletionBackend) -> None:
        self.backend = backend

    def translate_lines(
        self,
        lines: list[str],
        request: TranslationRequest,
        cancel_event: Event,
        parallel_workers: int = 1,
    ) -> list[str]:
        if not lines:
            return []

        target_lang = request.target_language
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

        chunk_size = 40
        max_attempts = 3
        chunks = [lines[i : i + chunk_size] for i in range(0, len(lines), chunk_size)]
        results: list[str] = []

        def translate_single_line(line: str) -> list[str]:
            last_content = ""
            last_error = ""
            for attempt in range(1, max_attempts + 2):
                user_prompt = (
                    f"请把以下单条字幕翻译为 `{target_lang}`。"
                    "只返回 JSON 字符串数组，且长度必须为 1。"
                    "禁止代码块、禁止解释、禁止额外字段。"
                    f"\n输入：{json.dumps([line], ensure_ascii=False)}"
                )
                content = self.backend.complete(request.model, system_prompt, user_prompt)
                last_content = content
                try:
                    parsed = parse_json_array_output(content)
                    if not parsed:
                        raise RuntimeError("翻译结果为空")
                    return [str(parsed[0]).strip()]
                except Exception as exc:
                    last_error = str(exc).strip() or "未知错误"
                    if attempt >= max_attempts + 1:
                        preview = last_content.strip().replace("\n", " ")
                        if len(preview) > 220:
                            preview = preview[:220] + "..."
                        raise RuntimeError(
                            f"单条字幕翻译失败：{last_error} | 返回片段：{preview}"
                        )
            raise RuntimeError("单条字幕翻译失败：未获得有效结果")

        def translate_chunk(chunk: list[str]) -> list[str]:
            if cancel_event.is_set():
                raise RuntimeError("翻译前已取消")

            translated_chunk: list[str] | None = None
            last_content = ""
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
                content = self.backend.complete(request.model, system_prompt, user_prompt)
                last_content = content
                try:
                    translated = parse_json_array_output(content)
                    if len(translated) != len(chunk):
                        raise RuntimeError(
                            f"翻译结果条数与原字幕不一致（期望 {len(chunk)}，实际 {len(translated)}）"
                        )
                    translated_chunk = translated
                    break
                except Exception:
                    if attempt < max_attempts:
                        continue

            if translated_chunk is None:
                if len(chunk) <= 1:
                    return translate_single_line(chunk[0] if chunk else "")
                mid = max(1, len(chunk) // 2)
                return translate_chunk(chunk[:mid]) + translate_chunk(chunk[mid:])
            return translated_chunk

        if parallel_workers <= 1 or len(chunks) <= 1:
            for chunk in chunks:
                results.extend(translate_chunk(chunk))
            return results

        ordered: list[list[str] | None] = [None] * len(chunks)
        workers = min(max(1, parallel_workers), len(chunks))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_idx = {pool.submit(translate_chunk, chunk): idx for idx, chunk in enumerate(chunks)}
            for future in as_completed(future_to_idx):
                ordered[future_to_idx[future]] = future.result()

        for item in ordered:
            if item is None:
                raise RuntimeError("翻译失败：并行结果缺失")
            results.extend(item)
        return results


def build_translation_provider(
    mode: str,
    mistral_api_key: str,
    openai_base_url: str,
    openai_api_key: str,
) -> TranslationProvider | None:
    if mode == "none":
        return None
    if mode == "mistral":
        return StructuredSubtitleTranslationProvider(MistralChatBackend(mistral_api_key))
    if mode == "openai":
        return StructuredSubtitleTranslationProvider(
            OpenAICompatibleChatBackend(openai_base_url, openai_api_key)
        )
    raise RuntimeError("未知翻译模式")

