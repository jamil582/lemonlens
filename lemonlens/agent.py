"""Tool-using analyst agent on Command A+ (v2 chat).

The model decides which tools to call; tool results go back as `document`
content blocks, so the final answer carries native citations that point at the
exact complaint, recall or aggregate row each claim came from.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable

import pandas as pd

from . import analytics, config
from .cohere_v2 import assistant_turn, text_of, thinking_of
from .index import ComplaintIndex, span_text

FILTER_PROPS = {
    "make": {"type": "string", "description": "Vehicle make, e.g. HONDA."},
    "model": {"type": "string", "description": "Vehicle model, e.g. CR-V."},
    "model_year_min": {"type": "integer", "description": "Earliest model year to include."},
    "model_year_max": {"type": "integer", "description": "Latest model year to include."},
    "component": {"type": "string", "description": "Component name or fragment, e.g. BRAKES."},
    "crash_only": {"type": "boolean", "description": "Only complaints that report a crash."},
    "fire_only": {"type": "boolean", "description": "Only complaints that report a fire."},
    "date_from": {"type": "string", "description": "Earliest filing date, YYYY-MM-DD."},
    "date_to": {"type": "string", "description": "Latest filing date, YYYY-MM-DD."},
    "theme": {"type": "string", "description": "A discovered theme label, if themes exist."},
}
GROUP = {"type": "string", "enum": analytics.GROUPS}
METRIC = {"type": "string", "enum": analytics.METRICS}


def _tool(name: str, description: str, props: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {**props, **FILTER_PROPS}, "required": required},
        },
    }


TOOLS = [
    _tool(
        "search_complaints",
        "Semantic search over complaint narratives. Use for 'what do owners say about…' questions and "
        "to find evidence. Returns the most relevant complaints with the passages that matched.",
        {
            "query": {"type": "string", "description": "What to look for, in natural language."},
            "top_k": {"type": "integer", "description": "How many complaints to return (default 8, max 20)."},
        },
        ["query"],
    ),
    _tool(
        "aggregate_complaints",
        "Count or rate complaints grouped by a field. Use for any 'how many', 'which is most', "
        "trend or comparison question. Never estimate numbers; call this.",
        {
            "group_by": GROUP,
            "metric": METRIC,
            "split_by": {**GROUP, "description": "Optional second grouping."},
            "top_n": {"type": "integer", "description": "Keep only the largest N groups."},
        },
        ["group_by"],
    ),
    _tool(
        "plot_chart",
        "Draw a chart in the dashboard for the user and return its data. Use bar for comparing "
        "categories, line for trends over year_filed/month_filed/model_year, heatmap for two fields.",
        {
            "chart_type": {"type": "string", "enum": ["bar", "line", "heatmap"]},
            "group_by": GROUP,
            "metric": METRIC,
            "split_by": {**GROUP, "description": "Series (line/bar) or second axis (heatmap, required)."},
            "top_n": {"type": "integer", "description": "Keep only the largest N groups."},
            "title": {"type": "string", "description": "Short chart title stating what it shows."},
        },
        ["chart_type", "group_by", "title"],
    ),
    _tool(
        "get_recalls",
        "List official NHTSA recalls for the loaded vehicles (summary, consequence, remedy).",
        {},
        [],
    ),
]

SYSTEM = """You are LemonLens, an analyst for vehicle-safety complaints filed by owners with the US \
National Highway Traffic Safety Administration (NHTSA). Today is {today}.

Loaded data: {data}
{scope}
How to work:
- Answer from the tools only. Use aggregate_complaints for every number, search_complaints for what \
owners describe, get_recalls for official recalls. Combine them when a question needs both.
- When a comparison or trend would be clearer as a picture, call plot_chart (at most two charts per answer).
- Complaints are unverified owner reports, and counts are not adjusted for how many vehicles were sold. \
Say so when it affects the conclusion; do not call a vehicle defective or unsafe on complaint counts alone.
- Lead with the answer, then the supporting evidence. Be concise and specific; quote owners sparingly.
- If the data cannot answer the question, say what is missing."""


@dataclass
class AgentResult:
    text: str = ""
    citations: list[dict] = field(default_factory=list)
    charts: list[dict] = field(default_factory=list)
    sources: dict[str, dict] = field(default_factory=dict)
    trace: list[dict] = field(default_factory=list)
    thinking: str = ""


def _filters(args: dict) -> dict:
    return {k: args[k] for k in analytics.FILTER_KEYS if args.get(k) not in (None, "", False)}


class Analyst:
    def __init__(self, co: Any, df: pd.DataFrame, index: ComplaintIndex, recalls: pd.DataFrame | None = None):
        self.co = co
        self.df = df  # must share row positions with index.df
        self.index = index
        self.recalls = recalls if recalls is not None else pd.DataFrame()

    # ------------------------------------------------------------------ tools
    def _search(self, args: dict, result: AgentResult) -> list[dict]:
        rows = analytics.apply_filters(self.df, _filters(args)).index
        hits = self.index.search(args["query"], rows=rows, top_k=min(int(args.get("top_k") or config.TOP_K), 20))
        docs = []
        for hit in hits:
            row = hit["row"]
            ref = f"complaint:{row['odi_number']}"
            filed = row["date_filed"]
            data = {
                "ref": ref,
                "odi_number": row["odi_number"],
                "vehicle": f"{row['model_year']} {row['vehicle']}",
                "component": ", ".join(row["components"]),
                "date_filed": "" if pd.isna(filed) else str(pd.Timestamp(filed).date()),
                "crash": bool(row["crash"]),
                "fire": bool(row["fire"]),
                "injuries": int(row["injuries"]),
                # Rerank's relevant spans double as context compression for the model.
                "relevant_text": span_text(row["summary"], hit["spans"]),
            }
            result.sources[ref] = {
                "kind": "complaint",
                "title": f"ODI {row['odi_number']} · {data['vehicle']} · {data['component']}",
                "snippet": data["relevant_text"][:400],
            }
            docs.append({"id": ref, "data": data})
        return docs or [{"id": "search:none", "data": {"ref": "search:none", "note": "No complaints matched."}}]

    def _aggregate(self, args: dict, result: AgentResult, label: str = "agg") -> tuple[pd.DataFrame, list[dict]]:
        table = analytics.aggregate(
            self.df,
            group_by=args["group_by"],
            metric=args.get("metric") or "complaints",
            split_by=args.get("split_by"),
            filters=_filters(args),
            top_n=args.get("top_n"),
        )
        ref = f"{label}:{len(result.sources) + 1}"
        records = json.loads(table.head(200).to_json(orient="records"))
        title = f"{args.get('metric') or 'complaints'} by {args['group_by']}" + (
            f" and {args['split_by']}" if args.get("split_by") else ""
        )
        if _filters(args):
            title += f" ({', '.join(f'{k}={v}' for k, v in _filters(args).items())})"
        result.sources[ref] = {"kind": "table", "title": title, "snippet": f"{len(table)} rows"}
        return table, [{"id": ref, "data": {"ref": ref, "table": title, "rows": records}}]

    def _plot(self, args: dict, result: AgentResult) -> list[dict]:
        if args["chart_type"] == "heatmap" and not args.get("split_by"):
            raise ValueError("heatmap needs split_by as its second axis")
        _, docs = self._aggregate(args, result, label="chart")
        result.charts.append(
            {
                "chart_type": args["chart_type"],
                "group_by": args["group_by"],
                "metric": args.get("metric") or "complaints",
                "split_by": args.get("split_by"),
                "top_n": args.get("top_n"),
                "filters": _filters(args),
                "title": args.get("title") or docs[0]["data"]["table"],
            }
        )
        docs[0]["data"]["note"] = "Chart displayed to the user."
        return docs

    def _recalls(self, args: dict, result: AgentResult) -> list[dict]:
        r = self.recalls
        if len(r) and args.get("model"):
            r = r[r["model"].str.upper() == str(args["model"]).upper()]
        if len(r) and args.get("component"):
            r = r[r["component"].str.contains(str(args["component"]).upper(), regex=False)]
        docs = []
        for campaign, group in r.groupby("campaign"):
            first = group.iloc[0]
            ref = f"recall:{campaign}"
            years = sorted(group["model_year"].unique())
            data = {
                "ref": ref,
                "campaign": campaign,
                "vehicles": f"{first['make']} {', '.join(sorted(group['model'].unique()))} {years[0]}-{years[-1]}",
                "component": first["component"],
                "date_received": "" if pd.isna(first["date_received"]) else str(first["date_received"].date()),
                "summary": first["summary"][:700],
                "consequence": first["consequence"][:300],
                "remedy": first["remedy"][:300],
            }
            result.sources[ref] = {
                "kind": "recall",
                "title": f"Recall {campaign} · {data['vehicles']} · {data['component']}",
                "snippet": data["summary"][:400],
            }
            docs.append({"id": ref, "data": data})
        return docs[:25] or [{"id": "recall:none", "data": {"ref": "recall:none", "note": "No recalls found."}}]

    def _dispatch(self, name: str, args: dict, result: AgentResult) -> list[dict]:
        try:
            if name == "search_complaints":
                return self._search(args, result)
            if name == "aggregate_complaints":
                return self._aggregate(args, result)[1]
            if name == "plot_chart":
                return self._plot(args, result)
            if name == "get_recalls":
                return self._recalls(args, result)
            raise ValueError(f"unknown tool {name}")
        except Exception as exc:  # report the error to the model so it can correct the call
            return [{"id": "error", "data": {"error": f"{type(exc).__name__}: {exc}"}}]

    # ------------------------------------------------------------------- loop
    def ask(
        self,
        question: str,
        history: list[dict] | None = None,
        scope: dict | None = None,
        on_step: Callable[[dict], None] | None = None,
    ) -> AgentResult:
        """Run the tool loop. `history` is prior {role, content} text turns;
        `scope` is the dashboard's current filter, given to the model as context."""
        result = AgentResult()
        scope_line = (
            f"The user's dashboard is currently filtered to: {json.dumps(scope)}. "
            "Apply the same filters in tool calls unless they ask otherwise.\n"
            if scope
            else ""
        )
        messages: list[dict] = [
            {
                "role": "system",
                "content": SYSTEM.format(
                    today=date.today().isoformat(), data=analytics.describe(self.df), scope=scope_line
                ),
            },
            *(history or []),
            {"role": "user", "content": question},
        ]
        thinking = (
            {"type": "enabled", "token_budget": config.THINKING_BUDGET}
            if config.THINKING_BUDGET > 0
            else {"type": "disabled"}
        )
        response: dict = {}
        for step in range(config.MAX_AGENT_STEPS + 1):
            last = step == config.MAX_AGENT_STEPS
            response = self.co.chat(
                model=config.CHAT_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="NONE" if last else None,  # out of steps: force a written answer
                thinking=thinking,
            )
            message = response.get("message") or {}
            calls = message.get("tool_calls") or []
            result.thinking = "\n".join(t for t in (result.thinking, thinking_of(response)) if t)
            if not calls or last:
                break
            messages.append(assistant_turn(message))
            entry = {"plan": message.get("tool_plan") or "", "calls": []}
            for call in calls:
                name = call["function"]["name"]
                try:
                    args = json.loads(call["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                docs = self._dispatch(name, args, result)
                entry["calls"].append({"tool": name, "args": args, "results": len(docs)})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": [
                            {"type": "document", "document": {"id": d["id"], "data": json.dumps(d["data"])}}
                            for d in docs
                        ],
                    }
                )
            result.trace.append(entry)
            if on_step:
                on_step(entry)
        result.text = text_of(response)
        result.citations = (response.get("message") or {}).get("citations") or []
        return result


# ---------------------------------------------------------------- citations
def _source_ref(source: dict, known: dict[str, dict]) -> str | None:
    """Map a citation source back to one of our document ids."""
    sid = str(source.get("id") or "")
    if sid in known:
        return sid
    payload = source.get("tool_output") or source.get("document") or {}
    if isinstance(payload, dict):
        ref = payload.get("ref")
        if ref is None and isinstance(payload.get("data"), str):
            try:
                ref = json.loads(payload["data"]).get("ref")
            except (ValueError, AttributeError):
                ref = None
        if ref in known:
            return str(ref)
    return next((k for k in known if k and k in sid), None)


def annotate(text: str, citations: list[dict], sources: dict[str, dict]) -> tuple[str, list[dict]]:
    """Insert [n] markers after each cited span; return the text and numbered sources."""
    numbers: dict[str, int] = {}
    inserts: list[tuple[int, str]] = []
    for c in sorted(citations, key=lambda c: (c.get("end", 0), c.get("start", 0))):
        refs = []
        for source in c.get("sources") or []:
            ref = _source_ref(source, sources)
            if ref and ref not in refs:
                refs.append(ref)
        if not refs:
            continue
        for ref in refs:
            numbers.setdefault(ref, len(numbers) + 1)
        marker = "".join(f"[{numbers[r]}]" for r in refs)
        inserts.append((min(int(c.get("end", len(text))), len(text)), " " + marker))
    for position, marker in sorted(inserts, reverse=True):
        text = text[:position] + marker + text[position:]
    ordered = [{"n": n, "ref": ref, **sources[ref]} for ref, n in sorted(numbers.items(), key=lambda kv: kv[1])]
    return text, ordered
