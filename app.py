"""LemonLens dashboard. Run with:  streamlit run app.py"""
from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from lemonlens import analytics, charts, config, demo_data, nhtsa, themes
from lemonlens.agent import Analyst, annotate
from lemonlens.cohere_v2 import CohereError, CohereV2
from lemonlens.index import ComplaintIndex, highlight_html
from lemonlens.mock import MockCohere

st.set_page_config(page_title="LemonLens", page_icon="🍋", layout="wide")
S = st.session_state

DEFAULT_VEHICLES = "HONDA, CR-V, 2017-2019\nTOYOTA, RAV4, 2019-2020"
SUGGESTIONS = [
    "What are owners saying about engine problems, and is it getting worse?",
    "Which components have the highest share of crash reports? Chart it.",
    "Compare the vehicles: where does each one stand out?",
    "Do any recalls line up with the most common complaints?",
]


def theme_mode() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


def stretch(render, *args, **kwargs):
    """Full-width element across Streamlit versions (width= replaced use_container_width=)."""
    try:
        return render(*args, width="stretch", **kwargs)
    except TypeError:
        return render(*args, use_container_width=True, **kwargs)


def plot(fig, key: str) -> None:
    stretch(st.plotly_chart, fig, key=key, theme=None, config={"displayModeBar": False})


def get_secret_key() -> str | None:
    try:
        return st.secrets.get("COHERE_API_KEY")
    except Exception:
        return None


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🍋 LemonLens")
    st.caption("An agentic dashboard over NHTSA vehicle-safety complaints, on Cohere's v2 API.")

    key = config.api_key() or get_secret_key()
    if not key:
        key = st.text_input(
            "Cohere API key", type="password", help="Kept in this browser session only. Or set COHERE_API_KEY in .env."
        ) or None
    live_models = bool(key)
    if "co" not in S or S.get("client_key") != key:
        S.client_key = key
        S.co = CohereV2(key) if key else MockCohere()
        for stale in ("index", "df", "recalls", "themes", "chat"):
            S.pop(stale, None)
    if not live_models:
        st.warning("No API key: models are mocked, so search, themes and answers are illustrative only.", icon="⚠️")

    st.subheader("Data")
    source = st.radio("Source", ["NHTSA (live)", "Synthetic demo"], index=0 if live_models else 1, horizontal=True)
    vehicle_text = DEFAULT_VEHICLES
    if source == "NHTSA (live)":
        vehicle_text = st.text_area(
            "Vehicles (MAKE, MODEL, YEARS per line)", value=DEFAULT_VEHICLES, height=110,
            help="Example: FORD, ESCAPE, 2020-2022. Each model year is one NHTSA request.",
        )
    if st.button("Load and index", type="primary") or "df" not in S and source == "Synthetic demo":
        try:
            if source == "NHTSA (live)":
                triples = nhtsa.parse_vehicle_lines(vehicle_text)
                if not triples:
                    st.error("Enter at least one line like: HONDA, CR-V, 2017-2019")
                    st.stop()
                bar_ = st.progress(0.0, "Fetching from NHTSA…")
                df, recalls = nhtsa.load(
                    triples, config.CACHE_DIR, lambda i, n: bar_.progress(i / n, f"Fetching from NHTSA… {i}/{n}")
                )
                bar_.empty()
            else:
                df, recalls = demo_data.complaints()
            if df.empty:
                st.error("NHTSA returned no complaints for those vehicles. Check the make and model spelling.")
                st.stop()
            if len(df) > config.MAX_COMPLAINTS:
                st.info(f"Keeping the {config.MAX_COMPLAINTS:,} most recent of {len(df):,} complaints.")
                df = df.tail(config.MAX_COMPLAINTS).reset_index(drop=True)
            bar_ = st.progress(0.0, "Embedding complaints…")
            index = ComplaintIndex(S.co, df, config.CACHE_DIR).build(
                lambda i, n: bar_.progress(i / n, f"Embedding complaints… {i:,}/{n:,}")
            )
            bar_.empty()
            S.df, S.recalls, S.index, S.synthetic = index.df, recalls, index, source != "NHTSA (live)"
            S.pop("themes", None)
            S.chat = []
        except CohereError as exc:
            st.error(f"{exc}")
            st.stop()
        except Exception as exc:
            st.error(f"Could not load data: {type(exc).__name__}: {exc}")
            st.stop()

if "df" not in S:
    st.header("LemonLens")
    st.write(
        "Pick vehicles in the sidebar and press **Load and index**. LemonLens pulls their complaints and recalls "
        "from NHTSA, embeds every narrative, and gives you charts, semantic search and an analyst agent to question."
    )
    st.stop()

df: pd.DataFrame = S.df
mode = theme_mode()
vehicle_colors = charts.color_map(sorted(df["vehicle"].unique()), mode)

# ------------------------------------------------------------------ filters
with st.sidebar:
    st.subheader("Filters")
    f_vehicles = st.multiselect("Vehicle", sorted(df["vehicle"].unique()))
    years = sorted(df["model_year"].unique())
    f_years = st.select_slider("Model years", options=years, value=(years[0], years[-1])) if len(years) > 1 else (years[0], years[0])
    all_components = analytics.aggregate(df, "component")["component"].tolist()
    f_component = st.selectbox("Component", ["All"] + all_components)
    f_crash = st.checkbox("Crash reported")
    f_fire = st.checkbox("Fire reported")

scope = {
    "model_year_min": int(f_years[0]) if f_years[0] != years[0] else None,
    "model_year_max": int(f_years[1]) if f_years[1] != years[-1] else None,
    "component": None if f_component == "All" else f_component,
    "crash_only": f_crash,
    "fire_only": f_fire,
}
view = analytics.apply_filters(df, scope)
if f_vehicles:
    view = view[view["vehicle"].isin(f_vehicles)]
scope = {k: v for k, v in scope.items() if v} | ({"vehicles": f_vehicles} if f_vehicles else {})

# ------------------------------------------------------------------- header
st.header("Vehicle complaint analysis")
if S.get("synthetic"):
    st.info("Synthetic demo data: invented vehicles and narratives, not real NHTSA records.", icon="🧪")
k = analytics.kpis(view)
st.caption(
    f"{k['complaints']:,} of {len(df):,} complaints · filed {k['first_filed']} to {k['last_filed']} · "
    "owner-reported and unverified; counts are not adjusted for vehicles sold"
)
cols = st.columns(5)
for col, (name, field) in zip(cols, [("Complaints", "complaints"), ("Crashes reported", "crashes"),
                                     ("Fires reported", "fires"), ("Injuries reported", "injuries"),
                                     ("Deaths reported", "deaths")]):
    col.metric(name, f"{k[field]:,}")

if view.empty:
    st.warning("No complaints match the current filters.")
    st.stop()

tab_overview, tab_themes, tab_search, tab_agent, tab_recalls, tab_data = st.tabs(
    ["Overview", "Themes", "Search", "Ask the analyst", "Recalls", "Data"]
)

# ----------------------------------------------------------------- overview
with tab_overview:
    left, right = st.columns(2)
    with left:
        monthly = analytics.aggregate(view, "month_filed", split_by="vehicle")
        plot(charts.line(monthly, "month_filed", "complaints", "Complaints filed per month", mode,
                         series="vehicle", colors=vehicle_colors), "ov_month")
    with right:
        comps = analytics.aggregate(view, "component", top_n=10)
        plot(charts.bar(comps, "component", "complaints", "Most reported components", mode), "ov_comp")
    left, right = st.columns(2)
    with left:
        grid = analytics.aggregate(view, "component", split_by="model_year", top_n=10)
        plot(charts.heatmap(grid, "component", "model_year", "complaints",
                            "Where complaints concentrate: component by model year", mode), "ov_heat")
    with right:
        rate = analytics.aggregate(view, "component", "crash_rate", top_n=None)
        volume = analytics.aggregate(view, "component").set_index("component")["complaints"]
        rate = rate[rate["component"].map(volume) >= 10].sort_values("crash_rate", ascending=False).head(10)
        if len(rate):
            plot(charts.bar(rate, "component", "crash_rate",
                            "Share of complaints reporting a crash (components with 10+ complaints)", mode), "ov_rate")
        else:
            st.caption("Not enough complaints per component to compare crash shares.")

# ------------------------------------------------------------------- themes
with tab_themes:
    st.write(
        "Complaints are clustered on their Embed 5 vectors, then Command A+ names each cluster "
        "with a structured (JSON-schema) output. Themes cover all loaded complaints."
    )
    c1, c2 = st.columns([1, 3])
    n_themes = c1.slider("Number of themes", 3, themes.MAX_THEMES, 6)
    if c1.button("Discover themes"):
        with st.spinner("Clustering and naming themes…"):
            series, table = themes.discover(S.co, S.index.df, S.index.vectors, n_themes)
            S.df = S.index.df = df.assign(theme=series.values)
            S.themes = table
        st.rerun()
    if "themes" in S and "theme" in df.columns:
        table = S.themes
        if table.attrs.get("error"):
            st.warning(f"Command A+ naming failed, so clusters are named by component. ({table.attrs['error']})")
        theme_colors = charts.color_map(table["theme"].tolist(), mode)
        left, right = st.columns(2)
        with left:
            sizes = analytics.aggregate(view, "theme")
            plot(charts.bar(sizes, "theme", "complaints", "Complaints per theme", mode), "th_size")
        with right:
            trend = analytics.aggregate(view, "year_filed", split_by="theme")
            plot(charts.line(trend, "year_filed", "complaints", "Themes over time", mode,
                             series="theme", colors=theme_colors), "th_trend")
        for _, row in table.iterrows():
            with st.expander(f"{row['theme']} · {row['complaints']:,} complaints"):
                st.write(row["description"])
                for position in row["examples"]:
                    r = df.iloc[position]
                    st.caption(f"ODI {r['odi_number']} · {r['model_year']} {r['vehicle']} · {r['component']}")
                    st.write(r["summary"][:600])
    else:
        st.caption("No themes yet. Press Discover themes.")

# ------------------------------------------------------------------- search
with tab_search:
    st.write(
        "Embed 5 finds candidates by meaning, Rerank 5 re-scores them and highlights the passages "
        "that matched. Search respects the sidebar filters."
    )
    query = st.text_input("Search complaint narratives", placeholder="e.g. car accelerates on its own in parking lots")
    if query:
        try:
            with st.spinner("Searching…"):
                hits = S.index.search(query, rows=view.index, top_k=10)
        except CohereError as exc:
            hits = []
            st.error(str(exc))
        if not hits:
            st.caption("No matching complaints.")
        for hit in hits:
            r = hit["row"]
            flags = " · ".join(n for n, on in (("crash", r["crash"]), ("fire", r["fire"])) if on)
            st.markdown(
                f"**{r['model_year']} {r['vehicle']}** · {', '.join(r['components'])} · ODI {r['odi_number']}"
                + (f" · {flags}" if flags else "")
            )
            st.caption(f"relevance {hit['relevance']:.2f} · filed {str(r['date_filed'])[:10]}")
            st.markdown(highlight_html(r["summary"], hit["spans"]), unsafe_allow_html=True)
            st.divider()

# -------------------------------------------------------------------- agent
def render_answer(entry: dict, n: int) -> None:
    st.markdown(entry["text"] or "_No answer returned._")
    for j, spec in enumerate(entry["charts"]):
        try:
            plot(charts.from_spec(spec, S.df, mode, vehicle_colors), f"chat_{n}_{j}")
        except Exception as exc:
            st.caption(f"Could not draw '{spec.get('title')}': {exc}")
    if entry["sources"]:
        with st.expander(f"Sources ({len(entry['sources'])})"):
            for s in entry["sources"]:
                st.markdown(f"**[{s['n']}] {html.escape(s['title'])}**")
                st.caption(s["snippet"])
    if entry["trace"]:
        with st.expander("How the analyst worked"):
            for step in entry["trace"]:
                if step["plan"]:
                    st.write(step["plan"])
                for call in step["calls"]:
                    st.code(f"{call['tool']}({call['args']}) -> {call['results']} result(s)", language=None)


with tab_agent:
    st.write(
        "Command A+ plans its own tool calls (semantic search, aggregation, charts, recalls) and cites "
        "the records behind each claim."
    )
    S.setdefault("chat", [])
    picked = None
    if not S.chat:
        for col, suggestion in zip(st.columns(2) + st.columns(2), SUGGESTIONS):
            if col.button(suggestion, key=f"sg_{suggestion}"):
                picked = suggestion
    for n, entry in enumerate(S.chat):
        with st.chat_message(entry["role"]):
            if entry["role"] == "user":
                st.markdown(entry["text"])
            else:
                render_answer(entry, n)
    question = st.chat_input("Ask about the loaded complaints") or picked
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            status = st.status("Working…")
            history = [{"role": e["role"], "content": e["raw"]} for e in S.chat[-8:] if e.get("raw")]
            try:
                analyst = Analyst(S.co, S.df, S.index, S.recalls)
                result = analyst.ask(
                    question, history=history, scope=scope,
                    on_step=lambda step: [status.write(f"{c['tool']} → {c['results']} result(s)") for c in step["calls"]],
                )
                text, sources = annotate(result.text, result.citations, result.sources)
                entry = {"role": "assistant", "text": text, "raw": result.text, "charts": result.charts,
                         "sources": sources, "trace": result.trace}
                status.update(label="Done", state="complete")
            except CohereError as exc:
                entry = {"role": "assistant", "text": f"The Cohere API returned an error: {exc}", "raw": "",
                         "charts": [], "sources": [], "trace": []}
                status.update(label="Failed", state="error")
            S.chat.append({"role": "user", "text": question, "raw": question})
            S.chat.append(entry)
            render_answer(entry, len(S.chat) - 1)

# ------------------------------------------------------------------ recalls
with tab_recalls:
    recalls: pd.DataFrame = S.recalls
    if recalls.empty:
        st.caption("NHTSA lists no recalls for the loaded vehicles.")
    else:
        grouped = (
            recalls.groupby("campaign")
            .agg(model=("model", lambda s: ", ".join(sorted(set(s)))),
                 model_years=("model_year", lambda s: f"{min(s)}-{max(s)}"),
                 component=("component", "first"), received=("date_received", "first"),
                 summary=("summary", "first"), consequence=("consequence", "first"), remedy=("remedy", "first"))
            .reset_index().sort_values("received", ascending=False)
        )
        stretch(st.dataframe, grouped, hide_index=True)
        pick = st.selectbox("Find complaints that describe the same defect as recall", grouped["campaign"])
        if st.button("Match complaints"):
            row = grouped[grouped["campaign"] == pick].iloc[0]
            with st.spinner("Searching…"):
                hits = S.index.search(f"{row['summary']} {row['consequence']}"[:1500], rows=view.index, top_k=10)
            received = row["received"]
            before = sum(1 for h in hits if pd.notna(received) and pd.notna(h["row"]["date_filed"]) and h["row"]["date_filed"] < received)
            st.caption(f"{len(hits)} closest complaints; {before} were filed before the recall was received ({str(received)[:10]}).")
            for hit in hits:
                r = hit["row"]
                st.markdown(f"**{r['model_year']} {r['vehicle']}** · ODI {r['odi_number']} · filed {str(r['date_filed'])[:10]} · relevance {hit['relevance']:.2f}")
                st.markdown(highlight_html(r["summary"], hit["spans"]), unsafe_allow_html=True)

# --------------------------------------------------------------------- data
with tab_data:
    table = view.drop(columns=["components"]).assign(component=view["components"].map(", ".join))
    stretch(st.dataframe, table, hide_index=True)
    st.download_button("Download filtered complaints (CSV)", table.to_csv(index=False), "complaints.csv", "text/csv")

# ---------------------------------------------------------------- telemetry
with st.sidebar:
    calls = pd.DataFrame(S.co.calls)
    with st.expander(f"API usage ({len(calls)} calls)"):
        if len(calls):
            usage = calls.groupby(["endpoint", "model"]).agg(
                calls=("seconds", "size"), seconds=("seconds", "sum"),
                input_tokens=("input_tokens", "sum"), output_tokens=("output_tokens", "sum"),
            ).reset_index()
            stretch(st.dataframe, usage, hide_index=True)
        else:
            st.caption("No calls yet.")
