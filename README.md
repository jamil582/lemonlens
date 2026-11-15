# 🍋 LemonLens

An agentic dashboard over public US vehicle-safety complaints (NHTSA), built on Cohere's **v2 API** with the latest Command, Embed and Rerank models.

Pick some vehicles, and LemonLens pulls every owner complaint and recall from NHTSA, embeds the narratives, and gives you:

- **Overview**: complaints per month by vehicle, most-reported components, a component × model-year heatmap, and crash share by component.
- **Themes**: failure patterns discovered by clustering the embeddings and named by Command A+.
- **Search**: semantic search with the matching passages highlighted.
- **Ask the analyst**: a tool-using agent that runs its own searches and aggregations, draws charts into the chat, and cites the complaint or recall behind each claim.
- **Recalls**: official recalls, with one-click matching to the complaints that describe the same defect.

## How each Cohere model is used

| Model | Endpoint | Role |
|---|---|---|
| `command-a-plus-05-2026` | `POST /v2/chat` | Agent tool loop (`tools`, `tool_plan`, `tool_calls`), native citations from tool results passed as `document` blocks, per-call `thinking` budget, and JSON-schema structured output for theme naming |
| `embed-v5.0-pro` | `POST /v2/embed` | Indexes complaint narratives (`input_type: search_document`) |
| `embed-v5.0-fast` | `POST /v2/embed` | Embeds queries (`input_type: search_query`); Pro and Fast share one embedding space |
| `rerank-v5.0-pro` | `POST /v2/rerank` | Re-scores the vector shortlist; `relevant_spans` drives the highlighting and trims what the agent reads to the passages that matter |

Everything goes through `lemonlens/cohere_v2.py`, a small client for the v2 REST endpoints (v2 message lists, `embedding_types`, content blocks, `message.citations`). No v1 calls are used.

```
NHTSA API ──► complaints table ──► Embed 5 Pro ──► vector index ─┐
                    │                                            ├─► Rerank 5 (+ spans) ─► search results
                    │                     query ─► Embed 5 Fast ─┘            │
                    └──► pandas aggregations ◄── Command A+ agent ◄───────────┘
                                 │                    │
                              charts            cited answer
```

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then paste your Cohere key into .env
streamlit run app.py
```

Enter vehicles in the sidebar, one per line as `MAKE, MODEL, YEARS` (for example `FORD, ESCAPE, 2020-2022`), and press **Load and index**. NHTSA data needs no key. Responses and embeddings are cached under `data/cache/`.

Without a Cohere key the app still runs: models are replaced by an offline mock and a synthetic dataset loads, both clearly labelled, so you can click through the interface.

## Settings

Environment variables (or `.env`): `COHERE_API_KEY`, `LEMONLENS_EMBED_DIM` (256–2048, default 1024), `LEMONLENS_THINKING_BUDGET` (tokens per agent step, 0 disables), `LEMONLENS_MAX_STEPS`, `LEMONLENS_CANDIDATES`, `LEMONLENS_TOP_K`, `LEMONLENS_MAX_COMPLAINTS`, and `LEMONLENS_*_MODEL` to swap model IDs.

## Tests

```bash
pytest -q
```

The 14 tests cover the v2 request shapes, retries, NHTSA normalisation, aggregations, retrieval, theme discovery, and the agent loop (tool dispatch, message shapes, citations, step limit). They run offline against the mock client.

## Reading the numbers

Complaints are unverified reports filed by owners. Counts are not adjusted for how many vehicles were sold, so a popular model will have more complaints than a rare one. A complaint that lists several components is counted once under each. Use the dashboard to find patterns worth investigating, not to rank vehicles by safety.

## Layout

```
app.py                  Streamlit dashboard
lemonlens/cohere_v2.py  v2 REST client (chat, embed, rerank)
lemonlens/nhtsa.py      NHTSA fetch + normalisation
lemonlens/index.py      Embed index, two-stage search, span highlighting
lemonlens/themes.py     KMeans clusters named with structured outputs
lemonlens/agent.py      Tool-using analyst and citation handling
lemonlens/analytics.py  Filters and aggregations
lemonlens/charts.py     Plotly figures
lemonlens/mock.py       Offline mock client     lemonlens/demo_data.py  Synthetic data
```

## License

MIT. See [LICENSE](LICENSE).
