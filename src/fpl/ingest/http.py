from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class DownloadError(RuntimeError):
    """Raised when a remote file or API cannot be read."""


def fetch_bytes(url: str, *, timeout: int, user_agent: str) -> bytes:
    request = Request(
        url,
        headers={"User-Agent": user_agent, "Accept": "*/*"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read()
    except HTTPError as exc:
        raise DownloadError(f"HTTP {exc.code} for {url}") from exc
    except URLError as exc:
        raise DownloadError(f"Request failed for {url}: {exc.reason}") from exc


def fetch_text(url: str, *, timeout: int, user_agent: str) -> str:
    return fetch_bytes(url, timeout=timeout, user_agent=user_agent).decode("utf-8")


def fetch_json(url: str, *, timeout: int, user_agent: str) -> Any:
    return json.loads(fetch_text(url, timeout=timeout, user_agent=user_agent))
