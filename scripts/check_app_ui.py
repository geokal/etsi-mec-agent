"""Headless smoke check for the Streamlit showcase (app.py).

    uv run python scripts/check_app_ui.py

No ONNX and no network LLM: fastembed is stubbed with deterministic fake embedders and openai is
faked, exactly like check_answer_grounding.py. Queries still hit the LIVE local Qdrant read-only
(localhost:6333), so the UI renders real payloads — in a meaningless order, which is fine here:
this check asserts wiring, not ranking. Nothing is written.
"""
import sys
import types
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

rng = np.random.default_rng(7)
_DENSE = rng.normal(size=384)
_DENSE /= np.linalg.norm(_DENSE)
_COLBERT_Q = rng.normal(size=(4, 128))


class _Dense:
    def __init__(self, *a, **k):
        pass

    def embed(self, texts):
        return (_DENSE for _ in texts)


class _Colbert:
    def __init__(self, *a, **k):
        pass

    def query_embed(self, texts):
        return (_COLBERT_Q for _ in texts)


stub = types.ModuleType("fastembed")
stub.TextEmbedding = _Dense
stub.LateInteractionTextEmbedding = _Colbert
sys.modules["fastembed"] = stub

answer_text = {"content": "placeholder"}
SERVED = "test/free-route-model"


def _chunks(text):
    # A leading choices=[] chunk is what real endpoints send for usage-only deltas;
    # stream_answer must skip it without dying (review-round regression).
    yield types.SimpleNamespace(model=SERVED, choices=[])
    for word in (text.split(" ") if text else []):  # an empty response streams no deltas
        yield types.SimpleNamespace(
            model=SERVED,
            choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content=word + " "))])


class _Completions:
    def create(self, *, model, messages, extra_body, stream):
        assert stream is True, "the UI must stream"
        return _chunks(answer_text["content"])


fake_openai = types.ModuleType("openai")
fake_openai.OpenAI = lambda base_url, api_key: types.SimpleNamespace(
    chat=types.SimpleNamespace(completions=_Completions()))
sys.modules["openai"] = fake_openai
import os
os.environ["OPENROUTER_API_KEY"] = "not-a-real-key"
# A stray APP_PASSWORD in the shell would stop every run at the gate; the gate gets its own
# targeted run at the end of this file.
os.environ.pop("APP_PASSWORD", None)
# Pinned before app.py imports the config: a developer's .env must not change what this check
# exercises (load_dotenv never overrides a variable that is already set).
os.environ["LLM_BASE_URL"] = "https://openrouter.ai/api/v1"
os.environ["LLM_MODEL"] = "openrouter/free"
os.environ.pop("LLM_API_KEY", None)

from streamlit.testing.v1 import AppTest


def click_generate(at):
    """Click the 💬 Generate answer button (the form's submit button is also in at.button)."""
    # at.button also lists the form's submit button, so pick by label, not index.
    gen = [b for b in at.button if b.label.startswith("💬")]
    assert gen, "the Generate answer button did not render"
    gen[0].click()

at = AppTest.from_file(str(root / "app.py"), default_timeout=180)
at.run()
assert not at.exception, at.exception
assert at.tabs and len(at.tabs) == 3, at.tabs
assert at.sidebar.radio and at.sidebar.toggle and len(at.sidebar.slider) == 3

# --- retrieval: the live collection answers, and the UI shapes it -------------------------
at.text_input[0].set_value("What is Mp1?")
at.form_submit_button[0].click()
at.run()
assert not at.exception, at.exception
result = at.session_state["result"]
assert result["display"], "live Qdrant returned nothing — is the collection populated?"
assert result["answer_docs"], "the evidence budget came back empty"
assert at.expander, "no per-hit expanders rendered"
assert at.caption, "no result summary caption rendered"
info = result["info"]
assert info["shown"] == len(result["display"]) and info["elapsed_ms"] >= 0
trace = at.session_state["trace"]
assert "[QUERY]" in trace and "[DEDUP]" in trace or "[HYBRID]" in trace, trace[:200]

# every hit expander is labelled by cover identity, never by filename
for exp in at.expander:
    assert ".pdf" not in exp.label, exp.label

# --- the ColBERT re-rank path reaches the UI through the same full_result shape ---------------
at.sidebar.radio[0].set_value("Dense + ColBERT re-rank")
at.form_submit_button[0].click()
at.run()
assert not at.exception, at.exception
result = at.session_state["result"]
assert result["info"]["path"].startswith("Dense prefetch"), result["info"]["path"]
assert result["display"] and result["answer_docs"], "colbert path returned no UI payload"
assert result["info"]["fetched"] >= len(result["display"])

# --- answer panel: a grounded citation passes the audit through the UI --------------------
at.sidebar.toggle[0].set_value(True)
at.run()
assert not at.exception, at.exception

stamped = [d for d in result["answer_docs"]
           if (d.meta or {}).get("spec_id") and (d.meta or {}).get("edition")
           and isinstance((d.meta or {}).get("page"), int)]
assert stamped, "no edition-stamped excerpt in the evidence budget — run backfill_spec_identity.py"
meta = stamped[0].meta
answer_text["content"] = f"The definition sits in {meta['spec_id']} {meta['edition']} p.{meta['page']}."
click_generate(at)
at.run()
assert not at.exception, at.exception
assert not at.error, at.error
ans = at.session_state["answer"]
assert ans["suspects"] == [], ans["suspects"]
assert ans["model"] == SERVED
assert at.success, "grounded citation did not render the success badge"
assert any("[ANSWER MODEL]" in c.value for c in at.caption), "model caption missing"

# --- and an empty-but-successful stream warns instead of silently doing nothing ------------
answer_text["content"] = ""
click_generate(at)
at.run()
assert not at.exception, at.exception
assert "answer" not in at.session_state, "an empty stream must not store an answer"
assert any("empty answer" in w.value for w in at.warning), [w.value for w in at.warning]

# --- and an over-citation is called out, not printed silently ------------------------------
answer_text["content"] = f"It also appears in {meta['spec_id']} {meta['edition']} p.999."
click_generate(at)
at.run()
assert not at.exception, at.exception
assert not at.error, at.error
assert at.session_state["answer"]["suspects"], "p.999 should be unsupported"
assert at.warning, "unsupported citation did not render a warning"

# --- corpus tab: the scroll-built table reaches the dataframe ------------------------------
assert at.metric and at.dataframe, "corpus metrics/table missing"

# --- the APP_PASSWORD gate: it must render and stop, not crash. Every run above leaves the
# gate unset, and st.password (the first draft's call) is not a Streamlit API at all.
os.environ["APP_PASSWORD"] = "hunter2"
at_gate = AppTest.from_file(str(root / "app.py"), default_timeout=60)
at_gate.run()
assert not at_gate.exception, at_gate.exception
assert not at_gate.error, at_gate.error
assert not at_gate.tabs, "the gate leaked the app UI before authentication"
assert any(w.label == "Password" for w in at_gate.text_input), [w.label for w in at_gate.text_input]

# A session authenticated against the CURRENT gate passes; one carrying the flag of a rotated
# password must be re-challenged.
at_gate.session_state["authed"] = "hunter2"
at_gate.run()
assert not at_gate.exception, at_gate.exception
assert at_gate.tabs, "the gate-flag session did not unlock the app"
os.environ["APP_PASSWORD"] = "hunter3"
at_gate.run()
assert not at_gate.exception, at_gate.exception
assert not at_gate.tabs, "a rotated gate must invalidate the old session"
assert any(w.label == "Password" for w in at_gate.text_input), [w.label for w in at_gate.text_input]
del os.environ["APP_PASSWORD"]

print("OK: the showcase UI wires retrieval, streaming answer and the grounding audit.")
