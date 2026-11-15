"""Semantic index over complaint narratives.

Retrieval is two-stage:
  1. Embed 5: documents are embedded once with `embed-v5.0-pro`; queries use
     `embed-v5.0-fast`, which shares the same embedding space and is quicker.
  2. Rerank 5: the vector shortlist is re-scored by `rerank-v5.0-pro`, with
     `relevant_spans` returning the passages that drove each score.
"""
from __future__ import annotations

import hashlib
import html
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from . import config


def doc_text(row: Any) -> str:
    """Text embedded for a complaint: vehicle and component context plus narrative."""
    return (
        f"{row['model_year']} {row['vehicle']} | Component: {', '.join(row['components'])}\n"
        f"{row['summary']}"
    )


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


class ComplaintIndex:
    def __init__(self, co: Any, df: pd.DataFrame, cache_dir: Path | None = None, dim: int | None = None):
        self.co = co
        self.df = df.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.dim = dim or config.EMBED_DIM
        self.vectors: np.ndarray | None = None

    def _cache_file(self) -> Path | None:
        if not self.cache_dir:
            return None
        ids = ",".join(self.df["odi_number"].astype(str))
        tag = f"{getattr(self.co, 'cache_tag', 'live')}|{config.EMBED_DOC_MODEL}|{self.dim}|{ids}"
        return self.cache_dir / f"emb_{hashlib.sha1(tag.encode()).hexdigest()[:16]}.npy"

    def build(self, progress: Callable[[int, int], None] | None = None) -> "ComplaintIndex":
        cache = self._cache_file()
        if cache and cache.exists():
            vectors = np.load(cache)
            if vectors.shape[0] == len(self.df):
                self.vectors = vectors
                return self
        texts = [doc_text(row) for _, row in self.df.iterrows()]
        vectors = self.co.embed(
            texts,
            model=config.EMBED_DOC_MODEL,
            input_type="search_document",
            output_dimension=self.dim,
            progress=progress,
        )
        self.vectors = _normalize(vectors)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, self.vectors)
        return self

    def embed_query(self, query: str) -> np.ndarray:
        vec = self.co.embed(
            [query], model=config.EMBED_QUERY_MODEL, input_type="search_query", output_dimension=self.dim
        )
        return _normalize(vec)[0]

    def search(
        self,
        query: str,
        rows: pd.Index | None = None,
        top_k: int | None = None,
        candidates: int | None = None,
        spans: bool = True,
    ) -> list[dict]:
        """Vector shortlist, then rerank. `rows` restricts the search to those
        positions of the index's dataframe (e.g. the current dashboard filter)."""
        if self.vectors is None:
            raise RuntimeError("index not built")
        top_k = top_k or config.TOP_K
        candidates = candidates or config.CANDIDATES
        pool = np.arange(len(self.df)) if rows is None else np.asarray(list(rows), dtype=int)
        if len(pool) == 0:
            return []
        sims = self.vectors[pool] @ self.embed_query(query)
        order = np.argsort(-sims)[:candidates]
        shortlist = pool[order]
        summaries = [self.df.at[int(i), "summary"] for i in shortlist]

        reranked = self.co.rerank(
            query=query,
            documents=summaries,
            model=config.RERANK_MODEL,
            top_n=min(top_k, len(summaries)),
            relevant_spans={"mode": "enabled", "threshold": 0.1, "max_spans": 3} if spans else None,
        )
        hits = []
        for result in reranked.get("results", []):
            position = int(shortlist[result["index"]])
            hits.append(
                {
                    "position": position,
                    "row": self.df.iloc[position].to_dict(),
                    "relevance": float(result["relevance_score"]),
                    "vector_score": float(sims[order[result["index"]]]),
                    "spans": result.get("relevant_spans") or [],
                }
            )
        return hits


def span_text(summary: str, spans: list[dict], fallback_chars: int = 900) -> str:
    """Context compression: keep only the spans Rerank marked relevant."""
    if not spans:
        return summary[:fallback_chars]
    ordered = sorted(spans, key=lambda s: s["start"])
    return " … ".join(summary[s["start"] : s["end"]].strip() for s in ordered)


def highlight_html(summary: str, spans: list[dict]) -> str:
    """Escape `summary` and wrap the relevant spans in <mark>."""
    merged: list[list[int]] = []
    for s in sorted(spans, key=lambda s: s["start"]):
        start, end = max(0, int(s["start"])), min(len(summary), int(s["end"]))
        if start >= end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    out, cursor = [], 0
    for start, end in merged:
        out.append(html.escape(summary[cursor:start]))
        out.append(f"<mark>{html.escape(summary[start:end])}</mark>")
        cursor = end
    out.append(html.escape(summary[cursor:]))
    return "".join(out)
