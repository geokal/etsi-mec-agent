"""Model-free check of the answer grounding in etsi_mec_agent.search.

    uv run python scripts/check_answer_grounding.py

`fastembed` is stubbed, so no ONNX model loads and nothing is embedded; the helpers under test are
pure string work and no Qdrant connection is made. Two things must hold: an excerpt introduces
itself by the identity stamped on it rather than by its filename, and the citation audit reports an
answer only when the excerpts genuinely cannot support it.
"""
import io
import os
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

stub = types.ModuleType("fastembed")
stub.TextEmbedding = object
stub.LateInteractionTextEmbedding = object
sys.modules.setdefault("fastembed", stub)

from etsi_mec_agent.dedup import SimpleDoc
from etsi_mec_agent.search import (
    _build_context,
    _norm_spec,
    _pages_by_spec,
    _source_label,
    _ungrounded_citations,
)

# The corpus hazard this exists for: the filename is a different spec from the text it holds.
docs = [
    SimpleDoc("The UALCMP is a MEC system level functional entity.", {
        "spec_id": "MEC-003", "edition": "V4.1.1", "page": 15,
        "filename": "MEC023.pdf", "heading": "5.3 Functional entities"}),
    SimpleDoc("Mx2 carries the user application LCM API.", {
        "spec_id": "MEC-016", "edition": "V3.1.1", "page": 6, "filename": "MEC016.pdf",
        "diagram_paths": ["data/diagrams/MEC016.pdf-0006-fig-00.png"]}),
    SimpleDoc("One more excerpt from a numbered annex.", {
        "spec_id": "MEC-DEC-032-2", "edition": "V3.2.1", "page": 56, "filename": "MEC032.pdf"}),
]

# Citations: MEC 003 / MEC003 / MEC-003 are the same spec, and so is a DEC part number.
for variant in ("MEC003", "MEC 003", "MEC-003"):
    assert _norm_spec(variant) == "MEC-003", variant
assert _norm_spec("MEC-DEC 032-2") == "MEC-DEC-032-2"
assert _norm_spec("MEC 010-2") == "MEC-010-2"

# An excerpt is labelled by its cover identity, and the label carries no filename.
assert _source_label(docs[0].meta) == "MEC-003 V4.1.1 p.15", _source_label(docs[0].meta)
assert ".pdf" not in _source_label(docs[0].meta)
# Undated/unsigned documents keep a visible fallback instead of a invented identity.
assert _source_label({"filename": "wp-mec.pptx", "page": 3}) == "wp-mec.pptx p.3"

context, labels = _build_context(docs)
assert len(labels) == len(docs), labels
assert labels[0] == "MEC-003 V4.1.1 p.15"
headers = [ln for ln in context.splitlines() if ln.startswith("--- [Source")]
assert len(headers) == len(docs), headers
assert all(label in head for label, head in zip(labels, headers)), headers
assert not any(".pdf" in head.split("—")[0] for head in headers), headers
assert "📐" in headers[1] and "data/diagrams/MEC016.pdf-0006-fig-00.png" in headers[1]
assert docs[0].content in context and docs[2].content in context

assert _pages_by_spec(docs) == {"MEC-003": {15}, "MEC-016": {6}, "MEC-DEC-032-2": {56}}

# Typographic citations. The live answer wrote "ETSI GS MEC 003 V4.1.1 (p.15)" with narrow
# no-break spaces (U+202F) and "MEC‑059" with a non-breaking hyphen (U+2011), and the first
# version of the matcher told the user those two correct citations named no specification.
assert _ungrounded_citations("requests (ETSI\u202fGS\u202fMEC\u202f003\u202fV4.1.1 (p.15))", docs) == []
assert _ungrounded_citations("the broader ecosystem (MEC\u2011016\u202fV3.1.1, p.6)", docs) == []
# Prose page numbers are citations too, and the spec may sit on either side of them.
assert _ungrounded_citations("Per MEC-016 V3.1.1, on page 6 the API is defined.", docs) == []
assert _ungrounded_citations("Defined on page 6 of MEC-016 V3.1.1.", docs) == []
bad = _ungrounded_citations("Defined on page 99 of MEC-016 V3.1.1.", docs)
assert len(bad) == 1 and "p.99" in bad[0], bad

# Unnumbered documents (white papers, decks) are cited by the name stamped on them, and that
# must resolve as well — the live context's 15th excerpt was exactly such a document.
paper = [SimpleDoc("AppD schema walkthrough.", {"spec_id": "AppDevelopmentDocument_v1", "page": 8})]
assert _ungrounded_citations("The schema is drawn in AppDevelopmentDocument_v1 p.8.", paper) == []
bad = _ungrounded_citations("The schema is drawn in AppDevelopmentDocument_v1 p.9.", paper)
assert len(bad) == 1 and "p.9" in bad[0], bad

# Grounded: both spellings, including the long form with a publication date in the middle.
grounded = (
    "The UALCMP is a system-level entity (MEC-003 V4.1.1 p.15).\n\n"
    "Its API runs over Mx2 (ETSI GS MEC 016 V3.1.1 (2024-03) p.6)."
)
assert _ungrounded_citations(grounded, docs) == [], _ungrounded_citations(grounded, docs)

# Right spec, page no excerpt sits on — the fault the audit exists to catch.
bad = _ungrounded_citations("It also relocates apps (MEC-016 V3.1.1 p.28).", docs)
assert len(bad) == 1 and "p.28" in bad[0] and "p.6" in bad[0], bad

# A specification the retrieval never returned.
bad = _ungrounded_citations("See MEC-099 V1.0.1 p.4 for the definition.", docs)
assert len(bad) == 1 and "no excerpt" in bad[0], bad

# A page number with no spec in front of it can't be traced by a reader either.
bad = _ungrounded_citations("Nothing here names a document, and yet it cites p.9.", docs)
assert bad == ["p.9 cited with no specification named"], bad

# The same unsupported citation repeated is one complaint, not several.
bad = _ungrounded_citations("(MEC-016 V3.1.1 p.28) again (MEC-016 V3.1.1 p.28).", docs)
assert len(bad) == 1, bad

# No stamped identity at all: nothing to check against, so nothing is accused.
assert _ungrounded_citations("(MEC-016 V3.1.1 p.28).",
                             [SimpleDoc("t", {"filename": "MEC016.pdf", "page": 6})]) == []

# ---------------------------------------------------------------------------
# The wiring, not just the helpers: generate_answer must label what the model reads and then
# audit what it writes back. `openai` is faked here, so nothing is sent and no real key is used.
# ---------------------------------------------------------------------------
import etsi_mec_agent.search as search

captured: dict = {}
answer_text = {"content": grounded}


SERVED = "test/free-route-model"   # what the router reports as the model that answered


def _chunks(text):
    """A streamed answer arrives as content deltas, one word at a time here."""
    for word in text.split(" "):
        yield types.SimpleNamespace(
            model=SERVED,
            choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content=word + " "))])


class _Completions:
    def create(self, *, model, messages, extra_body, stream):
        captured["model"] = model
        captured["messages"] = messages
        captured["stream"] = stream
        if stream:
            return _chunks(answer_text["content"])
        return types.SimpleNamespace(model=SERVED, choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content=answer_text["content"], reasoning_details=None))])


fake_openai = types.ModuleType("openai")
fake_openai.OpenAI = lambda base_url, api_key: types.SimpleNamespace(
    chat=types.SimpleNamespace(completions=_Completions()))
sys.modules["openai"] = fake_openai
os.environ["OPENROUTER_API_KEY"] = "not-a-real-key"

buf = io.StringIO()
with redirect_stdout(buf):
    returned = search.generate_answer("What is the UALCMP?", docs)
printed = buf.getvalue()

assert returned == grounded
assert captured["model"] == "openrouter/free" and captured["stream"] is False
user_prompt = captured["messages"][1]["content"]
assert "--- [Source 1] MEC-003 V4.1.1 p.15" in user_prompt, user_prompt[:400]
assert "MEC023.pdf" not in user_prompt          # a filename is not a citation
assert "Question: What is the UALCMP?" in user_prompt
assert "Cite only pages that appear on those lines" in captured["messages"][0]["content"]
assert f"[ANSWER MODEL] {SERVED}" in printed, printed
assert "[ANSWER SOURCES] 3 excerpts" in printed, printed
assert "[CITATION CHECK] 2 page citation(s), all present in the excerpts above." in printed, printed

# An answer citing a specification that was never retrieved must be called out, not printed silently.
answer_text["content"] = "It also covers MEC-099 V1.0.1 p.4."
buf = io.StringIO()
with redirect_stdout(buf):
    search.generate_answer("What is the UALCMP?", docs)
printed = buf.getvalue()
assert "[CITATION CHECK] 1 page citation(s), 1 unsupported:" in printed, printed
assert "MEC-099 p.4 — no excerpt from this specification was retrieved" in printed, printed

# --stream: the deltas are reassembled before the audit runs, so the warning must still appear.
answer_text["content"] = "Mx2 is defined in MEC-016 V3.1.1 p.6, not anywhere near p.77."
buf = io.StringIO()
with redirect_stdout(buf):
    streamed = search.generate_answer("What is the UALCMP?", docs, stream=True)
printed = buf.getvalue()
assert captured["stream"] is True
assert streamed.split() == answer_text["content"].split(), streamed
assert "[CITATION CHECK] 2 page citation(s), 1 unsupported:" in printed, printed
assert "MEC-016 p.77 — not in that spec's excerpts (they are p.6)" in printed, printed

# A canned response is called what it is instead of being printed as an answer. This is the literal
# string a live run returned for "What is the MEP in ETSI MEC?" on 2026-10-08.
answer_text["content"] = "User Safety: safe"
buf = io.StringIO()
with redirect_stdout(buf):
    search.generate_answer("What is the MEP in ETSI MEC?", docs)
printed = buf.getvalue()
assert f"[ANSWER MODEL] {SERVED}" in printed, printed
assert "3 word(s) and no citation — that is a canned response from the router" in printed, printed

print("OK: excerpt labels and the citation audit behave.")
