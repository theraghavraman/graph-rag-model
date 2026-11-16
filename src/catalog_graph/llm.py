"""A minimal client for any OpenAI-compatible endpoint (Ollama, LM Studio, vLLM, hosted APIs).

Standard library only. Configure with environment variables:

    CATALOG_LLM_BASE_URL     default http://localhost:11434/v1  (Ollama)
    CATALOG_LLM_API_KEY      default "not-needed" (local servers ignore it)
    CATALOG_LLM_MODEL        chat model name, required for chat
    CATALOG_EMBED_MODEL      embedding model name, required for embeddings
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field


class LLMError(RuntimeError):
    pass


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0

    def snapshot(self) -> tuple[int, int, int, float]:
        return (self.calls, self.prompt_tokens, self.completion_tokens, self.seconds)


@dataclass
class LLMClient:
    base_url: str = field(default_factory=lambda: os.environ.get("CATALOG_LLM_BASE_URL", "http://localhost:11434/v1"))
    api_key: str = field(default_factory=lambda: os.environ.get("CATALOG_LLM_API_KEY", "not-needed"))
    model: str = field(default_factory=lambda: os.environ.get("CATALOG_LLM_MODEL", ""))
    embed_model: str = field(default_factory=lambda: os.environ.get("CATALOG_EMBED_MODEL", ""))
    timeout: float = 120.0
    retries: int = 2
    usage: Usage = field(default_factory=Usage)

    def _post(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            self.base_url.rstrip("/") + path, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}, method="POST")
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            started = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode())
                self.usage.seconds += time.perf_counter() - started
                return body
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:300]
                last = LLMError(f"{path} returned HTTP {e.code}: {detail}")
                if e.code not in (408, 429, 500, 502, 503, 504):      # not worth retrying
                    break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                last = LLMError(f"{path} failed: {e}")
            if attempt < self.retries:
                time.sleep(0.5 * (attempt + 1))
        raise last or LLMError(f"{path} failed")

    def chat(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 600) -> str:
        if not self.model:
            raise LLMError("no chat model configured: set CATALOG_LLM_MODEL")
        body = self._post("/chat/completions", {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens})
        try:
            text = body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"unexpected chat response: {str(body)[:200]}")
        used = body.get("usage") or {}
        self.usage.calls += 1
        self.usage.prompt_tokens += int(used.get("prompt_tokens") or 0)
        self.usage.completion_tokens += int(used.get("completion_tokens") or 0)
        return text

    def embed(self, texts: list[str], batch: int = 64) -> list[list[float]]:
        if not self.embed_model:
            raise LLMError("no embedding model configured: set CATALOG_EMBED_MODEL")
        out: list[list[float]] = []
        for i in range(0, len(texts), batch):
            body = self._post("/embeddings", {"model": self.embed_model, "input": texts[i:i + batch]})
            try:
                rows = sorted(body["data"], key=lambda r: r.get("index", 0))
                out.extend(r["embedding"] for r in rows)
            except (KeyError, TypeError):
                raise LLMError(f"unexpected embeddings response: {str(body)[:200]}")
        if len(out) != len(texts):
            raise LLMError(f"asked for {len(texts)} embeddings, got {len(out)}")
        return out


def extract_json(text: str) -> dict:
    """The first JSON object in a model reply, tolerating code fences and surrounding prose."""
    start = text.find("{")
    while start != -1:
        depth, in_string, escaped = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(text[start:i + 1])
                        if isinstance(value, dict):
                            return value
                    except json.JSONDecodeError:
                        pass
                    break
        start = text.find("{", start + 1)
    raise ValueError("no JSON object in the reply")
