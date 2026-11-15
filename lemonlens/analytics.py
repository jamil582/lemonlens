"""Filtering and aggregation over the complaints table. Pure pandas; no API calls."""
from __future__ import annotations

from typing import Any

import pandas as pd

GROUPS = ["component", "vehicle", "make", "model", "model_year", "year_filed", "month_filed", "theme"]
METRICS = ["complaints", "crashes", "fires", "injuries", "deaths", "crash_rate", "fire_rate"]
ORDERED_GROUPS = {"model_year", "year_filed", "month_filed"}  # sorted by key, not by value

FILTER_KEYS = [
    "make", "model", "model_year_min", "model_year_max", "component",
    "crash_only", "fire_only", "date_from", "date_to", "theme",
]


def apply_filters(df: pd.DataFrame, filters: dict[str, Any] | None) -> pd.DataFrame:
    """Return the rows matching `filters`. Unknown or empty keys are ignored."""
    f = {k: v for k, v in (filters or {}).items() if v not in (None, "", [], False)}
    mask = pd.Series(True, index=df.index)
    if "make" in f:
        mask &= df["make"].str.upper() == str(f["make"]).upper()
    if "model" in f:
        mask &= df["model"].str.upper() == str(f["model"]).upper()
    if "model_year_min" in f:
        mask &= df["model_year"] >= int(f["model_year_min"])
    if "model_year_max" in f:
        mask &= df["model_year"] <= int(f["model_year_max"])
    if "component" in f:
        needle = str(f["component"]).upper()
        mask &= df["components"].map(lambda comps: any(needle in c for c in comps))
    if f.get("crash_only"):
        mask &= df["crash"]
    if f.get("fire_only"):
        mask &= df["fire"]
    if "date_from" in f:
        mask &= df["date_filed"] >= pd.Timestamp(f["date_from"])
    if "date_to" in f:
        mask &= df["date_filed"] <= pd.Timestamp(f["date_to"])
    if "theme" in f and "theme" in df.columns:
        mask &= df["theme"].astype(str).str.upper() == str(f["theme"]).upper()
    return df[mask]


def _keyed(df: pd.DataFrame, key: str, out: str) -> pd.DataFrame:
    if key not in GROUPS:
        raise ValueError(f"unknown group '{key}'; choose one of {GROUPS}")
    if key == "component":
        # A complaint can list several components; it counts once under each.
        d = df.explode("components")
        d[out] = d["components"]
        return d
    d = df.copy()
    if key == "year_filed":
        d = d[d["date_filed"].notna()]
        d[out] = d["date_filed"].dt.year.astype(int)
    elif key == "month_filed":
        d = d[d["date_filed"].notna()]
        d[out] = d["date_filed"].dt.strftime("%Y-%m")
    elif key == "theme":
        if "theme" not in d.columns:
            raise ValueError("themes have not been discovered yet (run 'Discover themes' first)")
        d[out] = d["theme"]
    else:
        d[out] = d[key]
    return d


def aggregate(
    df: pd.DataFrame,
    group_by: str,
    metric: str = "complaints",
    split_by: str | None = None,
    filters: dict | None = None,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Group complaints and compute one metric. Columns: group_by, [split_by], metric."""
    if metric not in METRICS:
        raise ValueError(f"unknown metric '{metric}'; choose one of {METRICS}")
    if split_by and split_by == group_by:
        split_by = None
    d = _keyed(apply_filters(df, filters), group_by, group_by + "__g")
    keys = [group_by + "__g"]
    if split_by:
        d = _keyed(d, split_by, split_by + "__s")
        keys.append(split_by + "__s")
    if d.empty:
        return pd.DataFrame(columns=[group_by] + ([split_by] if split_by else []) + [metric])
    out = (
        d.groupby(keys)
        .agg(
            complaints=("odi_number", "nunique"),
            crashes=("crash", "sum"),
            fires=("fire", "sum"),
            injuries=("injuries", "sum"),
            deaths=("deaths", "sum"),
        )
        .reset_index()
    )
    out["crash_rate"] = (100 * out["crashes"] / out["complaints"]).round(1)
    out["fire_rate"] = (100 * out["fires"] / out["complaints"]).round(1)
    out = out.rename(columns={keys[0]: group_by, **({keys[1]: split_by} if split_by else {})})
    out = out[[group_by] + ([split_by] if split_by else []) + [metric]]

    if group_by in ORDERED_GROUPS:
        out = out.sort_values([group_by] + ([split_by] if split_by else []))
    else:
        totals = out.groupby(group_by)[metric].sum().sort_values(ascending=False)
        if top_n:
            totals = totals.head(int(top_n))
        out = out[out[group_by].isin(totals.index)]
        order = {k: i for i, k in enumerate(totals.index)}
        out = out.assign(_o=out[group_by].map(order)).sort_values(["_o"]).drop(columns="_o")
    return out.reset_index(drop=True)


def fold_top(data: pd.DataFrame, col: str, value: str, keep: int = 7) -> pd.DataFrame:
    """Keep the `keep` largest categories of `col`; fold the rest into 'Other'."""
    totals = data.groupby(col)[value].sum().sort_values(ascending=False)
    if len(totals) <= keep + 1:
        return data
    top = set(totals.head(keep).index)
    folded = data.assign(**{col: data[col].where(data[col].isin(top), "Other")})
    others = [c for c in data.columns if c not in (col, value)]
    return folded.groupby(others + [col], as_index=False)[value].sum()


def kpis(df: pd.DataFrame) -> dict[str, Any]:
    dates = df["date_filed"].dropna()
    return {
        "complaints": int(len(df)),
        "crashes": int(df["crash"].sum()),
        "fires": int(df["fire"].sum()),
        "injuries": int(df["injuries"].sum()),
        "deaths": int(df["deaths"].sum()),
        "first_filed": dates.min().date().isoformat() if len(dates) else None,
        "last_filed": dates.max().date().isoformat() if len(dates) else None,
    }


def describe(df: pd.DataFrame) -> str:
    """One-paragraph description of the loaded data, used in the agent's system prompt."""
    k = kpis(df)
    vehicles = ", ".join(
        f"{v} ({g['model_year'].min()}-{g['model_year'].max()}, {len(g)} complaints)"
        for v, g in df.groupby("vehicle")
    )
    comps = aggregate(df, "component", top_n=15)["component"].tolist()
    text = (
        f"{k['complaints']} complaints filed {k['first_filed']} to {k['last_filed']}. "
        f"Vehicles: {vehicles}. Most common components: {', '.join(comps)}."
    )
    if "theme" in df.columns:
        text += f" Discovered themes: {', '.join(sorted(df['theme'].dropna().unique()))}."
    return text
