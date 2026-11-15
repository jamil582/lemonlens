"""Fetch and normalise complaints and recalls from NHTSA's public API (no key needed).

Endpoints (https://api.nhtsa.gov):
  /complaints/complaintsByVehicle?make=&model=&modelYear=
  /recalls/recallsByVehicle?make=&model=&modelYear=
  /products/vehicle/models?modelYear=&make=&issueType=c
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import requests

BASE = "https://api.nhtsa.gov"
CACHE_TTL = 24 * 3600

COMPLAINT_COLUMNS = [
    "odi_number", "make", "model", "model_year", "vehicle", "components", "component",
    "date_filed", "date_incident", "crash", "fire", "injuries", "deaths", "summary",
]
RECALL_COLUMNS = [
    "campaign", "make", "model", "model_year", "component", "date_received",
    "summary", "consequence", "remedy",
]


def _get(path: str, params: dict, cache_dir: Path | None, session: Any = None) -> dict:
    key = hashlib.sha1(f"{path}?{json.dumps(params, sort_keys=True)}".encode()).hexdigest()[:16]
    cache_file = cache_dir / f"nhtsa_{key}.json" if cache_dir else None
    if cache_file and cache_file.exists() and time.time() - cache_file.stat().st_mtime < CACHE_TTL:
        return json.loads(cache_file.read_text())
    resp = (session or requests).get(f"{BASE}{path}", params=params, timeout=60)
    if resp.status_code == 400:  # NHTSA answers 400 when a vehicle has no records
        return {"results": []}
    resp.raise_for_status()
    data = resp.json()
    if cache_file:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(data))
    return data


def _results(data: dict) -> list[dict]:
    return data.get("results") or data.get("Results") or []


def _split_components(value: Any) -> list[str]:
    if isinstance(value, list):
        parts = [str(v) for v in value]
    else:
        parts = str(value or "").split(",")
    parts = [p.strip().upper() for p in parts if p and p.strip()]
    return parts or ["UNKNOWN OR OTHER"]


def _date(value: Any, formats: tuple[str, ...]) -> pd.Timestamp:
    for fmt in formats:
        ts = pd.to_datetime(value, format=fmt, errors="coerce")
        if not pd.isna(ts):
            return ts
    return pd.to_datetime(value, errors="coerce")


def normalize_complaints(results: list[dict], make: str, model: str, year: int) -> list[dict]:
    rows = []
    for r in results:
        product = next(
            (p for p in r.get("products") or [] if str(p.get("type", "")).lower() == "vehicle"), {}
        )
        mk = str(product.get("productMake") or make).upper()
        md = str(product.get("productModel") or model).upper()
        try:
            yr = int(product.get("productYear") or year)
        except (TypeError, ValueError):
            yr = int(year)
        comps = _split_components(r.get("components"))
        rows.append(
            {
                "odi_number": str(r.get("odiNumber")),
                "make": mk,
                "model": md,
                "model_year": yr,
                "vehicle": f"{mk} {md}",
                "components": comps,
                "component": comps[0],
                "date_filed": _date(r.get("dateComplaintFiled"), ("%m/%d/%Y",)),
                "date_incident": _date(r.get("dateOfIncident"), ("%m/%d/%Y",)),
                "crash": bool(r.get("crash")),
                "fire": bool(r.get("fire")),
                "injuries": int(r.get("numberOfInjuries") or 0),
                "deaths": int(r.get("numberOfDeaths") or 0),
                "summary": " ".join(str(r.get("summary") or "").split()),
            }
        )
    return rows


def normalize_recalls(results: list[dict], make: str, model: str, year: int) -> list[dict]:
    rows = []
    for r in results:
        rows.append(
            {
                "campaign": str(r.get("NHTSACampaignNumber") or ""),
                "make": str(r.get("Make") or make).upper(),
                "model": str(r.get("Model") or model).upper(),
                "model_year": int(r.get("ModelYear") or year),
                "component": str(r.get("Component") or "").upper(),
                "date_received": _date(r.get("ReportReceivedDate"), ("%d/%m/%Y", "%m/%d/%Y")),
                "summary": " ".join(str(r.get("Summary") or "").split()),
                "consequence": " ".join(str(r.get("Consequence") or r.get("Conequence") or "").split()),
                "remedy": " ".join(str(r.get("Remedy") or "").split()),
            }
        )
    return rows


def list_models(make: str, year: int, cache_dir: Path | None = None) -> list[str]:
    data = _get("/products/vehicle/models", {"modelYear": year, "make": make, "issueType": "c"}, cache_dir)
    return sorted({str(r.get("model")) for r in _results(data) if r.get("model")})


def parse_vehicle_lines(text: str) -> list[tuple[str, str, int]]:
    """Parse lines like 'HONDA, CR-V, 2017-2019' into (make, model, year) triples."""
    triples: list[tuple[str, str, int]] = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3 or not all(parts):
            continue
        make, model, years = parts
        try:
            if "-" in years:
                lo, hi = (int(y) for y in years.split("-", 1))
            else:
                lo = hi = int(years)
        except ValueError:
            continue
        for year in range(min(lo, hi), max(lo, hi) + 1):
            triples.append((make.upper(), model.upper(), year))
    return list(dict.fromkeys(triples))


def load(
    vehicles: list[tuple[str, str, int]],
    cache_dir: Path | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch complaints and recalls for each (make, model, year)."""
    complaints: list[dict] = []
    recalls: list[dict] = []
    for i, (make, model, year) in enumerate(vehicles):
        params = {"make": make, "model": model, "modelYear": year}
        complaints += normalize_complaints(
            _results(_get("/complaints/complaintsByVehicle", params, cache_dir)), make, model, year
        )
        recalls += normalize_recalls(
            _results(_get("/recalls/recallsByVehicle", params, cache_dir)), make, model, year
        )
        if progress:
            progress(i + 1, len(vehicles))
    return frames(complaints, recalls)


def frames(complaints: list[dict], recalls: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cdf = pd.DataFrame(complaints, columns=COMPLAINT_COLUMNS)
    if len(cdf):
        cdf = cdf[cdf["summary"].str.len() > 0].drop_duplicates("odi_number")
        cdf = cdf.sort_values("date_filed", na_position="first").reset_index(drop=True)
    rdf = pd.DataFrame(recalls, columns=RECALL_COLUMNS)
    if len(rdf):
        rdf = rdf.drop_duplicates(["campaign", "model", "model_year"]).reset_index(drop=True)
    return cdf, rdf
