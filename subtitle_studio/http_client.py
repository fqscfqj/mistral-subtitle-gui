from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import requests


@dataclass
class HttpResponse:
    status_code: int
    payload: Any
    text: str


class HttpClient:
    def __init__(self, timeout_seconds: int = 180) -> None:
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()

    def post_json(
        self,
        url: str,
        payload: Mapping[str, Any],
        headers: Optional[Mapping[str, str]] = None,
    ) -> HttpResponse:
        response = self.session.post(
            url,
            json=dict(payload),
            headers=dict(headers or {}),
            timeout=self.timeout_seconds,
        )
        return self._build_response(response)

    def post_multipart(
        self,
        url: str,
        data: Mapping[str, Any],
        files: Mapping[str, tuple[str, bytes, str]],
        headers: Optional[Mapping[str, str]] = None,
    ) -> HttpResponse:
        response = self.session.post(
            url,
            data=dict(data),
            files=dict(files),
            headers=dict(headers or {}),
            timeout=self.timeout_seconds,
        )
        return self._build_response(response)

    def _build_response(self, response: requests.Response) -> HttpResponse:
        raw_text = response.text
        try:
            payload = response.json()
        except Exception:
            payload = None
        return HttpResponse(status_code=response.status_code, payload=payload, text=raw_text)

