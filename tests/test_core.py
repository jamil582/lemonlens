import json

import numpy as np
import pandas as pd
import pytest

from lemonlens import analytics, config, demo_data, nhtsa, themes
from lemonlens.agent import TOOLS, Analyst, annotate
from lemonlens.cohere_v2 import CohereError, CohereV2, assistant_turn, text_of, thinking_of
from lemonlens.index import ComplaintIndex, highlight_html, span_text
from lemonlens.mock import MockCohere


@pytest.fixture(scope="module")
def data():
    df, recalls = demo_data.complaints(300)
    index = ComplaintIndex(MockCohere(), df, cache_dir=None, dim=256).build()
    return index.df, recalls, index


# ---------------------------------------------------------------- v2 client
class FakeResp:
    def __init__(self, status, body):
        self.status_code, self._body, self.text = status, body, json.dumps(body)

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, responses):
        self.responses, self.sent = list(responses), []

    def post(self, url, headers, json, timeout):
        self.sent.append((url, headers, json))
        return self.responses.pop(0)


def test_embed_batches_and_v2_payload():
    batch = lambda n: FakeResp(200, {"embeddings": {"float": [[0.1, 0.2]] * n}, "meta": {"billed_units": {"input_tokens": 5}}})
    session = FakeSession([batch(96), batch(4)])
    co = CohereV2("k", session=session)
    out = co.embed(["t"] * 100, model="embed-v5.0-pro", input_type="search_document", output_dimension=256)
    assert out.shape == (100, 2)
    url, headers, payload = session.sent[0]
    assert url.endswith("/v2/embed") and headers["Authorization"] == "Bearer k"
    assert payload["embedding_types"] == ["float"] and payload["input_type"] == "search_document"
    assert payload["output_dimension"] == 256 and len(payload["texts"]) == 96
    assert len(session.sent[1][2]["texts"]) == 4 and co.calls[0]["input_tokens"] == 5


def test_rerank_and_chat_payloads():
    session = FakeSession([FakeResp(200, {"results": []}), FakeResp(200, {"message": {"content": []}})])
    co = CohereV2("k", session=session)
    co.rerank("q", ["a", "b"], model="rerank-v5.0-pro", top_n=1, relevant_spans={"mode": "enabled"})
    co.chat([{"role": "user", "content": "hi"}], model="command-a-plus-05-2026", thinking={"type": "disabled"})
    assert session.sent[0][0].endswith("/v2/rerank")
    assert session.sent[0][2] == {"model": "rerank-v5.0-pro", "query": "q", "documents": ["a", "b"], "top_n": 1,
                                  "relevant_spans": {"mode": "enabled"}}
    assert session.sent[1][0].endswith("/v2/chat")
    assert set(session.sent[1][2]) == {"model", "messages", "thinking"}


def test_errors_and_retry(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    session = FakeSession([FakeResp(429, {"message": "slow down"}), FakeResp(200, {"results": []})])
    assert CohereV2("k", session=session).rerank("q", ["a"], model="m") == {"results": []}
    with pytest.raises(CohereError) as err:
        CohereV2("k", session=FakeSession([FakeResp(401, {"message": "invalid api token"})])).rerank("q", ["a"], model="m")
    assert err.value.status == 401 and "invalid api token" in str(err.value)


def test_content_block_helpers():
    resp = {"message": {"role": "assistant", "content": [{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "4"}],
                        "tool_calls": None, "citations": []}}
    assert text_of(resp) == "4" and thinking_of(resp) == "hmm"
    assert assistant_turn(resp["message"]) == {"role": "assistant", "content": resp["message"]["content"]}


# -------------------------------------------------------------------- NHTSA
def test_normalize_and_parse():
    raw = [{"odiNumber": 1, "crash": True, "fire": False, "numberOfInjuries": 2, "numberOfDeaths": 0,
            "dateOfIncident": "01/05/2021", "dateComplaintFiled": "02/15/2021", "components": "ENGINE,ELECTRICAL SYSTEM",
            "summary": " stalls \n often ",
            "products": [{"type": "Vehicle", "productYear": "2019", "productMake": "Honda", "productModel": "CR-V"}]}]
    df, _ = nhtsa.frames(nhtsa.normalize_complaints(raw, "x", "y", 2000), [])
    row = df.iloc[0]
    assert row["vehicle"] == "HONDA CR-V" and row["model_year"] == 2019
    assert row["components"] == ["ENGINE", "ELECTRICAL SYSTEM"] and row["summary"] == "stalls often"
    assert row["date_filed"] == pd.Timestamp("2021-02-15") and row["crash"] and row["injuries"] == 2
    rec = nhtsa.normalize_recalls([{"NHTSACampaignNumber": "21V001", "ReportReceivedDate": "25/03/2021",
                                    "Component": "engine", "Summary": "s", "ModelYear": "2019"}], "HONDA", "CR-V", 2019)
    assert rec[0]["date_received"] == pd.Timestamp("2021-03-25") and rec[0]["component"] == "ENGINE"
    assert nhtsa.parse_vehicle_lines("honda, cr-v, 2017-2018\nbad line\nFORD, ESCAPE, 2020") == [
        ("HONDA", "CR-V", 2017), ("HONDA", "CR-V", 2018), ("FORD", "ESCAPE", 2020)]


# ---------------------------------------------------------------- analytics
def test_aggregate_and_filters(data):
    df, _, _ = data
    by_vehicle = analytics.aggregate(df, "vehicle")
    assert by_vehicle["complaints"].sum() == len(df)
    assert list(by_vehicle["complaints"]) == sorted(by_vehicle["complaints"], reverse=True)
    brakes = analytics.apply_filters(df, {"component": "brakes", "model": "arrow"})
    assert len(brakes) and all("SERVICE BRAKES" in c for c in brakes["components"]) and set(brakes["model"]) == {"ARROW"}
    months = analytics.aggregate(df, "month_filed", split_by="vehicle")
    assert list(months.columns) == ["month_filed", "vehicle", "complaints"]
    assert list(months["month_filed"]) == sorted(months["month_filed"])
    rate = analytics.aggregate(df, "component", "crash_rate")
    assert rate["crash_rate"].between(0, 100).all()
    assert len(analytics.aggregate(df, "component", top_n=3)) == 3
    assert analytics.aggregate(df, "component", filters={"make": "NOPE"}).empty
    with pytest.raises(ValueError):
        analytics.aggregate(df, "theme")
    assert analytics.kpis(df)["complaints"] == len(df) and "DEMO MOTORS ARROW" in analytics.describe(df)


def test_fold_top():
    d = pd.DataFrame({"x": [1] * 12, "s": list("abcdefghijkl"), "v": range(12)})
    folded = analytics.fold_top(d, "s", "v", keep=7)
    assert folded["s"].nunique() == 8 and folded["v"].sum() == d["v"].sum()
    assert folded.loc[folded["s"] == "Other", "v"].item() == sum(range(5))


# ------------------------------------------------------------------- search
def test_search_respects_filter_and_spans(data):
    df, _, index = data
    rows = analytics.apply_filters(df, {"model": "BISON"}).index
    hits = index.search("battery drains overnight will not start", rows=rows, top_k=5)
    assert hits and all(h["row"]["model"] == "BISON" for h in hits)
    assert "battery" in hits[0]["row"]["summary"].lower()
    assert hits[0]["relevance"] >= hits[-1]["relevance"]
    top = hits[0]
    assert top["spans"] and "battery" in span_text(top["row"]["summary"], top["spans"]).lower()
    assert index.search("anything", rows=pd.Index([])) == []


def test_highlight_escapes_and_merges():
    out = highlight_html("a <b> brake & fail", [{"start": 2, "end": 11}, {"start": 8, "end": 13}])
    assert out == "a <mark>&lt;b&gt; brake &amp;</mark> fail"


# ------------------------------------------------------------------- themes
def test_themes(data):
    df, _, index = data
    series, table = themes.discover(MockCohere(), index.df, index.vectors, k=5)
    assert len(series) == len(df) and table["theme"].is_unique and len(table) == 5
    assert table["complaints"].sum() == len(df) and set(series) == set(table["theme"])
    themed = df.assign(theme=series.values)
    assert analytics.aggregate(themed, "theme")["complaints"].sum() == len(df)


# -------------------------------------------------------------------- agent
def test_tool_schemas_are_valid():
    for tool in TOOLS:
        fn = tool["function"]
        assert tool["type"] == "function" and fn["parameters"]["type"] == "object"
        assert set(fn["parameters"]["required"]) <= set(fn["parameters"]["properties"])


def test_agent_loop_with_citations(data):
    df, recalls, index = data
    result = Analyst(MockCohere(), df, index, recalls).ask("brake pedal goes soft", scope={"model": "ARROW"})
    assert result.text and result.trace[0]["calls"][0]["tool"] == "search_complaints"
    assert result.charts[0]["chart_type"] == "bar" and result.citations
    text, sources = annotate(result.text, result.citations, result.sources)
    assert "[1]" in text and sources[0]["n"] == 1 and sources[0]["kind"] == "complaint"


class ScriptedCo(MockCohere):
    """Replays a fixed list of chat responses and records what it was sent."""
    def __init__(self, script):
        super().__init__()
        self.script, self.seen = list(script), []

    def chat(self, messages, model, **kwargs):
        self.seen.append({"messages": json.loads(json.dumps(messages)), **kwargs})
        return self.script.pop(0)


def _call(name, args, cid):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def test_agent_tool_dispatch_and_message_shapes(data):
    df, recalls, index = data
    script = [
        {"message": {"role": "assistant", "tool_plan": "plan", "content": [{"type": "thinking", "thinking": "t"}], "tool_calls": [
            _call("aggregate_complaints", {"group_by": "component", "metric": "crash_rate", "top_n": 3}, "c1"),
            _call("get_recalls", {"model": "ARROW"}, "c2"),
            _call("plot_chart", {"chart_type": "heatmap", "group_by": "component", "title": "x"}, "c3"),
            _call("nope", {}, "c4")]}},
        {"message": {"role": "assistant", "content": [{"type": "text", "text": "Engines lead."}],
                     "citations": [{"start": 0, "end": 7, "text": "Engines", "sources": [{"type": "tool", "id": "c2:0", "tool_output": {"ref": "recall:DEMO-001"}}]}]}},
    ]
    co = ScriptedCo(script)
    result = Analyst(co, df, index, recalls).ask("q", history=[{"role": "user", "content": "earlier"}])
    sent = co.seen[1]["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "user", "assistant", "tool", "tool", "tool", "tool"]
    assert sent[3]["tool_calls"][0]["id"] == "c1" and sent[3]["tool_plan"] == "plan" and sent[3]["content"][0]["type"] == "thinking"
    agg = json.loads(sent[4]["content"][0]["document"]["data"])
    assert sent[4]["tool_call_id"] == "c1" and sent[4]["content"][0]["type"] == "document" and len(agg["rows"]) == 3
    assert json.loads(sent[5]["content"][0]["document"]["data"])["campaign"] == "DEMO-001"
    assert "heatmap needs split_by" in json.loads(sent[6]["content"][0]["document"]["data"])["error"]
    assert "unknown tool" in json.loads(sent[7]["content"][0]["document"]["data"])["error"]
    assert co.seen[0]["thinking"] == {"type": "enabled", "token_budget": config.THINKING_BUDGET}
    assert result.thinking == "t" and not result.charts
    text, sources = annotate(result.text, result.citations, result.sources)
    assert text == "Engines [1] lead." and sources[0]["ref"] == "recall:DEMO-001"


def test_agent_stops_after_max_steps(data, monkeypatch):
    df, recalls, index = data
    monkeypatch.setattr(config, "MAX_AGENT_STEPS", 2)
    loop = {"message": {"role": "assistant", "tool_calls": [_call("get_recalls", {}, "c")]}}
    co = ScriptedCo([loop, loop, {"message": {"role": "assistant", "content": [{"type": "text", "text": "done"}]}}])
    result = Analyst(co, df, index, recalls).ask("q")
    assert result.text == "done" and len(result.trace) == 2 and co.seen[-1]["tool_choice"] == "NONE"


def test_placeholder_key_is_ignored(monkeypatch):
    monkeypatch.delenv("CO_API_KEY", raising=False)
    monkeypatch.setenv("COHERE_API_KEY", config.PLACEHOLDER_KEY)
    assert config.api_key() is None
    monkeypatch.setenv("COHERE_API_KEY", " real-key ")
    assert config.api_key() == "real-key"
