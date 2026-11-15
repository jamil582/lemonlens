"""Offline stand-in for the Cohere client, returning v2-shaped responses.

Lets the dashboard and tests run without an API key or network. Embeddings are
hashed bags of words, rerank is word overlap, and chat follows a fixed script,
so results are illustrative only; the UI labels this mode clearly.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Callable

import numpy as np

STOP = set(
    "the a an and or of to in on at is was it my i with for while when that this has have had be been "
    "are were not no from by as vehicle car what which how do does about most common problems".split()
)


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP and len(t) > 2]


class MockCohere:
    cache_tag = "mock"

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._n = 0

    def _record(self, endpoint: str, model: str) -> None:
        self.calls.append(
            {"endpoint": endpoint, "model": f"{model} (mock)", "seconds": 0.0, "input_tokens": 0, "output_tokens": 0}
        )

    # ---------------------------------------------------------------- embed
    def embed(
        self,
        texts: list[str],
        model: str,
        input_type: str,
        output_dimension: int | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        dim = output_dimension or 256
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in _tokens(text):
                h = int(hashlib.md5(token.encode()).hexdigest(), 16)
                out[i, h % dim] += 1.0 if (h >> 20) % 2 else -1.0
        self._record("/v2/embed", model)
        if progress:
            progress(len(texts), len(texts))
        return out

    # --------------------------------------------------------------- rerank
    def rerank(
        self,
        query: str,
        documents: list[str],
        model: str,
        top_n: int | None = None,
        relevant_spans: dict | None = None,
        max_tokens_per_doc: int | None = None,
    ) -> dict:
        q = set(_tokens(query))
        results = []
        for i, doc in enumerate(documents):
            words = set(_tokens(doc))
            score = len(q & words) / max(len(q), 1)
            item: dict[str, Any] = {"index": i, "relevance_score": round(min(score, 0.999), 4)}
            if relevant_spans and relevant_spans.get("mode") == "enabled":
                spans = []
                for match in re.finditer(r"[^.!?]+[.!?]?", doc):
                    hits = len(q & set(_tokens(match.group())))
                    if hits:
                        spans.append(
                            {"start": match.start(), "end": match.end(), "score": round(min(hits / max(len(q), 1), 0.999), 4)}
                        )
                spans.sort(key=lambda s: -s["score"])
                if spans:
                    item["relevant_spans"] = spans[: relevant_spans.get("max_spans") or 3]
            results.append(item)
        results.sort(key=lambda r: -r["relevance_score"])
        self._record("/v2/rerank", model)
        return {"id": "mock", "results": results[: top_n or len(results)], "meta": {"api_version": {"version": "2"}}}

    # ----------------------------------------------------------------- chat
    def chat(self, messages: list[dict], model: str, tools: list[dict] | None = None, **kwargs: Any) -> dict:
        self._record("/v2/chat", model)
        last = messages[-1]
        if kwargs.get("response_format"):
            return self._message(text=self._theme_json(last.get("content", "")))
        if tools and last["role"] == "user" and kwargs.get("tool_choice") != "NONE":
            question = last["content"]
            self._n += 1
            calls = [
                self._call("search_complaints", {"query": question, "top_k": 5}),
                self._call(
                    "plot_chart",
                    {"chart_type": "bar", "group_by": "component", "top_n": 8, "title": "Complaints by component"},
                ),
            ]
            return self._message(tool_calls=calls, tool_plan="(demo) Search the narratives, then chart the components.")
        if last["role"] == "tool":
            return self._summarise(messages)
        return self._message(text="Demo mode: add a Cohere API key to get real answers from Command A+.")

    def _call(self, name: str, args: dict) -> dict:
        return {
            "id": f"{name}_{self._n}_{len(args)}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }

    def _message(self, text: str = "", tool_calls: list | None = None, tool_plan: str = "", citations: list | None = None) -> dict:
        message: dict[str, Any] = {"role": "assistant"}
        if text:
            message["content"] = [{"type": "text", "text": text}]
        if tool_calls:
            message["tool_calls"], message["tool_plan"] = tool_calls, tool_plan
        if citations:
            message["citations"] = citations
        return {
            "id": "mock",
            "finish_reason": "TOOL_CALL" if tool_calls else "COMPLETE",
            "message": message,
            "usage": {"tokens": {"input_tokens": 0, "output_tokens": 0}},
        }

    def _summarise(self, messages: list[dict]) -> dict:
        docs = []
        for m in messages:
            if m["role"] == "tool":
                for block in m["content"]:
                    docs.append((m["tool_call_id"], json.loads(block["document"]["data"])))
        complaints = [(cid, d) for cid, d in docs if str(d.get("ref", "")).startswith("complaint:")]
        text = "Demo mode (no Cohere key), so this is a scripted summary rather than model analysis. "
        citations = []
        if complaints:
            text += f"The closest {len(complaints)} complaints by word overlap: "
            for i, (cid, d) in enumerate(complaints[:3]):
                start = len(text)
                text += f"{d['vehicle']} ({d['component'].lower()})"
                citations.append(
                    {
                        "start": start,
                        "end": len(text),
                        "text": text[start:],
                        "sources": [{"type": "tool", "id": f"{cid}:{i}", "tool_output": d}],
                    }
                )
                text += "; " if i < min(len(complaints), 3) - 1 else ". "
        else:
            text += "No complaints matched that wording. "
        text += "The chart shows complaint counts by component for the loaded data."
        return self._message(text=text, citations=citations)

    @staticmethod
    def _theme_json(prompt: str) -> str:
        themes = []
        try:
            clusters = json.loads(prompt[prompt.index("Clusters:") + len("Clusters:") :])
        except ValueError:
            clusters = []
        for c in clusters:
            counts: dict[str, int] = {}
            for sample in c.get("samples", []):
                component = sample[1 : sample.index("]")] if sample.startswith("[") and "]" in sample else "MIXED"
                counts[component] = counts.get(component, 0) + 1
            top = max(counts, key=counts.get) if counts else "MIXED"
            themes.append(
                {
                    "cluster_id": c["cluster_id"],
                    "label": f"{top.title()} issues",
                    "description": "Demo label taken from the cluster's most common component.",
                }
            )
        return json.dumps({"themes": themes})
