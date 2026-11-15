"""Discover failure themes: cluster the Embed vectors, then have Command A+ name
each cluster using structured outputs (a JSON schema the reply must satisfy)."""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from . import config
from .cohere_v2 import text_of

MAX_THEMES = 8  # one per categorical colour slot; more would need an "Other" fold

SCHEMA = {
    "type": "object",
    "properties": {
        "themes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "cluster_id": {"type": "integer"},
                    "label": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["cluster_id", "label", "description"],
            },
        }
    },
    "required": ["themes"],
}

PROMPT = """You are labelling clusters of vehicle-safety complaints filed with NHTSA.
For each cluster below, read the sample complaints and return:
- label: a specific 2-5 word name for the failure pattern (e.g. "Oil dilution in engine"), unique across clusters
- description: one sentence on what owners report, in plain language
Use only what the samples say. Return JSON with a `themes` array, one entry per cluster_id.

Clusters:
{clusters}"""


def cluster(vectors: np.ndarray, k: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    k = max(2, min(k, MAX_THEMES, len(vectors)))
    model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(vectors)
    return model.labels_, model.cluster_centers_


def representatives(vectors: np.ndarray, labels: np.ndarray, centers: np.ndarray, per: int = 8) -> dict[int, list[int]]:
    """Positions of the complaints closest to each cluster centre."""
    out = {}
    for cid in range(len(centers)):
        members = np.where(labels == cid)[0]
        dist = np.linalg.norm(vectors[members] - centers[cid], axis=1)
        out[cid] = members[np.argsort(dist)[:per]].tolist()
    return out


def discover(co: Any, df: pd.DataFrame, vectors: np.ndarray, k: int = 6) -> tuple[pd.Series, pd.DataFrame]:
    """Return (theme label per complaint, table of themes)."""
    labels, centers = cluster(vectors, k)
    reps = representatives(vectors, labels, centers)
    payload = [
        {
            "cluster_id": cid,
            "size": int((labels == cid).sum()),
            "samples": [
                f"[{df.at[i, 'component']}] {df.at[i, 'summary'][:400]}" for i in positions
            ],
        }
        for cid, positions in reps.items()
    ]
    names: dict[int, dict] = {}
    try:
        response = co.chat(
            model=config.CHAT_MODEL,
            messages=[{"role": "user", "content": PROMPT.format(clusters=json.dumps(payload, indent=1))}],
            response_format={"type": "json_object", "json_schema": SCHEMA},
            thinking={"type": "disabled"},  # a labelling task: no reasoning budget needed
            temperature=0.2,
        )
        for item in json.loads(text_of(response)).get("themes", []):
            names[int(item["cluster_id"])] = item
    except Exception as exc:  # fall back to component-based names rather than fail the page
        names = {"_error": {"label": str(exc)}}  # type: ignore[dict-item]

    rows, used = [], set()
    for cid, positions in reps.items():
        item = names.get(cid) or {}
        label = str(item.get("label") or "").strip()
        if not label:
            top = df.loc[labels == cid, "component"].mode()
            label = f"{top.iloc[0].title() if len(top) else 'Mixed'} issues"
        base, n = label, 2
        while label in used:  # labels must be unique: they key the charts
            label, n = f"{base} ({n})", n + 1
        used.add(label)
        rows.append(
            {
                "cluster_id": cid,
                "theme": label,
                "description": str(item.get("description") or ""),
                "complaints": int((labels == cid).sum()),
                "examples": positions[:3],
            }
        )
    table = pd.DataFrame(rows).sort_values("complaints", ascending=False).reset_index(drop=True)
    table.attrs["error"] = names.get("_error", {}).get("label") if "_error" in names else None
    mapping = {r["cluster_id"]: r["theme"] for r in rows}
    return pd.Series([mapping[int(c)] for c in labels], index=df.index, name="theme"), table
