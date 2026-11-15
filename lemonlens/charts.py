"""Plotly figures with one consistent visual system.

Colour rules: categorical hues are assigned to entities in a fixed order and
keep their colour when filters change; magnitude uses a single blue ramp; more
than eight series fold into "Other".
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from . import analytics

# Categorical slots (fixed order), light and dark steps of the same hues.
CATEGORICAL = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}
OTHER = {"light": "#9a9992", "dark": "#7c7b75"}
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK = {
    "light": {"primary": "#0b0b0b", "secondary": "#52514e", "grid": "#e6e5e0", "surface": "#fcfcfb"},
    "dark": {"primary": "#f0efec", "secondary": "#c3c2b7", "grid": "#33332f", "surface": "#1a1a19"},
}
LABELS = {
    "complaints": "Complaints", "crashes": "Crashes reported", "fires": "Fires reported",
    "injuries": "Injuries reported", "deaths": "Deaths reported",
    "crash_rate": "Complaints reporting a crash (%)", "fire_rate": "Complaints reporting a fire (%)",
    "component": "Component", "vehicle": "Vehicle", "make": "Make", "model": "Model",
    "model_year": "Model year", "year_filed": "Year filed", "month_filed": "Month filed", "theme": "Theme",
}


def label(key: str) -> str:
    return LABELS.get(key, key.replace("_", " ").title())


def color_map(entities: list, mode: str = "light") -> dict:
    """Stable entity -> colour assignment. Pass the full entity list (not the
    filtered one) so a filter never repaints the survivors."""
    palette = CATEGORICAL[mode]
    mapping = {e: palette[i] for i, e in enumerate(entities[: len(palette)])}
    mapping["Other"] = OTHER[mode]
    return mapping


def _layout(fig: go.Figure, title: str, mode: str, height: int, legend: bool) -> go.Figure:
    ink = INK[mode]
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=ink["primary"]), x=0, xanchor="left"),
        height=height,
        margin=dict(l=10, r=10, t=70 if legend else 46, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=ink["secondary"], size=12),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title_text=""),
        hoverlabel=dict(bgcolor=ink["surface"], font_color=ink["primary"], bordercolor=ink["grid"]),
        bargap=0.3,
    )
    fig.update_xaxes(gridcolor=ink["grid"], zeroline=False, linecolor=ink["grid"], automargin=True)
    fig.update_yaxes(gridcolor=ink["grid"], zeroline=False, linecolor=ink["grid"], automargin=True)
    return fig


def bar(data: pd.DataFrame, cat: str, value: str, title: str, mode: str = "light",
        series: str | None = None, colors: dict | None = None) -> go.Figure:
    """Horizontal bars, largest on top. With `series`, bars are grouped by it."""
    fig = go.Figure()
    surface = INK[mode]["surface"]
    order = list(dict.fromkeys(data[cat]))[::-1]
    if series:
        data = analytics.fold_top(data, series, value)
        names = [n for n in dict.fromkeys(data[series]) if n != "Other"]
        names += ["Other"] if "Other" in set(data[series]) else []
        colors = colors or color_map(names, mode)
        fallback = color_map([n for n in names if n not in colors], mode)
        for name in names:
            part = data[data[series] == name]
            fig.add_bar(
                y=part[cat].astype(str), x=part[value], name=str(name), orientation="h",
                marker=dict(color=colors.get(name) or fallback[name], line=dict(color=surface, width=2)),
                hovertemplate=f"%{{y}}<br>{name}: %{{x:,}}<extra></extra>",
            )
        fig.update_layout(barmode="group")
    else:
        fig.add_bar(
            y=data[cat].astype(str), x=data[value], orientation="h",
            marker=dict(color=CATEGORICAL[mode][0], line=dict(color=surface, width=2)),
            hovertemplate=f"%{{y}}<br>{label(value)}: %{{x:,}}<extra></extra>",
        )
    height = max(260, 34 * len(order) * (1 + 0.35 * bool(series)) + 90)
    _layout(fig, title, mode, int(height), legend=bool(series))
    fig.update_yaxes(categoryorder="array", categoryarray=[str(o) for o in order], showgrid=False, type="category")
    fig.update_xaxes(title_text=label(value), rangemode="tozero")
    return fig


def line(data: pd.DataFrame, x: str, value: str, title: str, mode: str = "light",
         series: str | None = None, colors: dict | None = None) -> go.Figure:
    fig = go.Figure()
    markers = data[x].nunique() <= 24
    if series:
        data = analytics.fold_top(data, series, value)
        names = [n for n in dict.fromkeys(data[series]) if n != "Other"]
        names += ["Other"] if "Other" in set(data[series]) else []
        colors = colors or color_map(names, mode)
        fallback = color_map([n for n in names if n not in colors], mode)
        for name in names:
            part = data[data[series] == name].sort_values(x)
            fig.add_scatter(
                x=part[x], y=part[value], name=str(name), mode="lines+markers" if markers else "lines",
                line=dict(color=colors.get(name) or fallback[name], width=2), marker=dict(size=8),
                hovertemplate=f"{name}: %{{y:,}}<extra></extra>",
            )
    else:
        part = data.sort_values(x)
        fig.add_scatter(
            x=part[x], y=part[value], mode="lines+markers" if markers else "lines",
            line=dict(color=CATEGORICAL[mode][0], width=2), marker=dict(size=8),
            hovertemplate=f"{label(value)}: %{{y:,}}<extra></extra>",
        )
    _layout(fig, title, mode, 360, legend=bool(series))
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(showgrid=False, title_text=label(x), type="category" if x in ("model_year", "year_filed") else None)
    fig.update_yaxes(title_text=label(value), rangemode="tozero")
    return fig


def heatmap(data: pd.DataFrame, row: str, col: str, value: str, title: str, mode: str = "light") -> go.Figure:
    pivot = data.pivot_table(index=row, columns=col, values=value, aggfunc="sum", fill_value=0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values().index]  # largest row on top
    scale = SEQUENTIAL if mode == "light" else SEQUENTIAL[::-1]
    fig = go.Figure(
        go.Heatmap(
            z=pivot.values, x=[str(c) for c in pivot.columns], y=[str(r) for r in pivot.index],
            colorscale=[[i / (len(scale) - 1), c] for i, c in enumerate(scale)],
            xgap=2, ygap=2, zmin=0, colorbar=dict(title=label(value), thickness=12, outlinewidth=0),
            hovertemplate=f"%{{y}} · %{{x}}<br>{label(value)}: %{{z:,}}<extra></extra>",
        )
    )
    _layout(fig, title, mode, max(280, 30 * len(pivot) + 110), legend=False)
    fig.update_xaxes(type="category", showgrid=False, title_text=label(col))
    fig.update_yaxes(type="category", showgrid=False)
    return fig


def from_spec(spec: dict, df: pd.DataFrame, mode: str = "light", colors: dict | None = None) -> go.Figure:
    """Render a chart requested by the agent's plot_chart tool."""
    kind, group, metric, split = spec["chart_type"], spec["group_by"], spec["metric"], spec.get("split_by")
    top_n = spec.get("top_n") or (12 if group not in analytics.ORDERED_GROUPS else None)
    data = analytics.aggregate(df, group, metric, split_by=split, filters=spec.get("filters"), top_n=top_n)
    colors = colors if split == "vehicle" else None
    if kind == "heatmap" and split:
        return heatmap(data, group, split, metric, spec["title"], mode)
    if kind == "line":
        return line(data, group, metric, spec["title"], mode, series=split, colors=colors)
    return bar(data, group, metric, spec["title"], mode, series=split, colors=colors)
