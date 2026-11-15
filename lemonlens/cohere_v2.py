"""Thin client for Cohere's v2 REST API: /v2/chat, /v2/embed, /v2/rerank.

Calling the v2 endpoints directly (rather than through the SDK) keeps the request
shapes explicit and lets new fields such as `thinking` and Rerank's
`relevant_spans` work regardless of the installed SDK version.
"""
from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
import requests

EMBED_BATCH = 96  # max texts per /v2/embed call
RETRY_STATUS = {429, 500, 502, 503, 504}


class CohereError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(f"Cohere API error {status}: {message}")
        self.status = status


class CohereV2:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.cohere.com",
        timeout: float = 180.0,
        max_retries: int = 4,
        session: Any = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = session or requests.Session()
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Client-Name": "lemonlens",
        }
        self.calls: list[dict] = []  # telemetry: one entry per API call

    # ------------------------------------------------------------------ transport
    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        started = time.time()
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.post(url, headers=self.headers, json=payload, timeout=self.timeout)
            except requests.RequestException as exc:
                if attempt == self.max_retries:
                    raise CohereError(0, f"network error: {exc}") from exc
                time.sleep(min(2**attempt, 20))
                continue
            if resp.status_code == 200:
                data = resp.json()
                self._record(path, payload, data, time.time() - started)
                return data
            if resp.status_code in RETRY_STATUS and attempt < self.max_retries:
                time.sleep(min(2**attempt, 20))
                continue
            try:
                message = resp.json().get("message", resp.text)
            except ValueError:
                message = resp.text
            raise CohereError(resp.status_code, str(message)[:500])
        raise CohereError(0, "retries exhausted")

    def _record(self, path: str, payload: dict, data: dict, seconds: float) -> None:
        usage = data.get("usage") or data.get("meta") or {}
        tokens = usage.get("tokens") or usage.get("billed_units") or {}
        self.calls.append(
            {
                "endpoint": path,
                "model": payload.get("model"),
                "seconds": round(seconds, 2),
                "input_tokens": int(tokens.get("input_tokens") or 0),
                "output_tokens": int(tokens.get("output_tokens") or 0),
            }
        )

    # ------------------------------------------------------------------ endpoints
    def chat(
        self,
        messages: list[dict],
        model: str,
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        documents: list | None = None,
        response_format: dict | None = None,
        thinking: dict | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        """POST /v2/chat. Returns the raw response dict (see `text_of`, `thinking_of`)."""
        payload: dict[str, Any] = {"model": model, "messages": messages}
        for key, value in (
            ("tools", tools),
            ("tool_choice", tool_choice),
            ("documents", documents),
            ("response_format", response_format),
            ("thinking", thinking),
            ("temperature", temperature),
            ("max_tokens", max_tokens),
        ):
            if value is not None:
                payload[key] = value
        return self._post("/v2/chat", payload)

    def embed(
        self,
        texts: list[str],
        model: str,
        input_type: str,
        output_dimension: int | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        """POST /v2/embed in batches of 96. Returns a float32 array (n, dim)."""
        chunks = []
        for start in range(0, len(texts), EMBED_BATCH):
            payload: dict[str, Any] = {
                "model": model,
                "input_type": input_type,  # search_document | search_query | ...
                "texts": texts[start : start + EMBED_BATCH],
                "embedding_types": ["float"],  # required in v2
                "truncate": "END",
            }
            if output_dimension:
                payload["output_dimension"] = output_dimension
            data = self._post("/v2/embed", payload)
            chunks.append(np.asarray(data["embeddings"]["float"], dtype=np.float32))
            if progress:
                progress(min(start + EMBED_BATCH, len(texts)), len(texts))
        if not chunks:
            return np.zeros((0, output_dimension or 0), dtype=np.float32)
        return np.vstack(chunks)

    def rerank(
        self,
        query: str,
        documents: list[str],
        model: str,
        top_n: int | None = None,
        relevant_spans: dict | None = None,
        max_tokens_per_doc: int | None = None,
    ) -> dict:
        """POST /v2/rerank. `relevant_spans` (Rerank v5+) returns the character
        ranges of each document that drove its score."""
        payload: dict[str, Any] = {"model": model, "query": query, "documents": documents}
        if top_n is not None:
            payload["top_n"] = top_n
        if relevant_spans is not None:
            payload["relevant_spans"] = relevant_spans
        if max_tokens_per_doc is not None:
            payload["max_tokens_per_doc"] = max_tokens_per_doc
        return self._post("/v2/rerank", payload)


# ---------------------------------------------------------------------- helpers
def text_of(response: dict) -> str:
    """Final answer text. v2 returns a list of content blocks; with reasoning
    models a `thinking` block can precede the `text` block."""
    blocks = (response.get("message") or {}).get("content") or []
    if isinstance(blocks, str):
        return blocks
    return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")


def thinking_of(response: dict) -> str:
    blocks = (response.get("message") or {}).get("content") or []
    if isinstance(blocks, str):
        return ""
    return "\n".join(b.get("thinking", "") for b in blocks if b.get("type") == "thinking")


def assistant_turn(message: dict) -> dict:
    """The assistant message to append back onto `messages` during a tool loop."""
    turn = {"role": "assistant"}
    for key in ("content", "tool_plan", "tool_calls"):
        if message.get(key):
            turn[key] = message[key]
    return turn
