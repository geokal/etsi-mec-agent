"""Streamlit showcase over the ETSI MEC hybrid RAG pipeline.

    uv run streamlit run app.py

Retrieval runs against whichever Qdrant the env selects (QDRANT_URL wins over host/port);
answers go to whichever OpenAI-compatible endpoint LLM_BASE_URL/LLM_MODEL name
(default: the OpenRouter free router, key via LLM_API_KEY or OPENROUTER_API_KEY).
"""
import contextlib
import hmac
import io
import os
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd
import streamlit as st

from etsi_mec_agent.config import settings
from etsi_mec_agent.search import (
    _PAGE_RE,
    _build_context,
    _pages_by_spec,
    _plain,
    _source_label,
    _ungrounded_citations,
    search_specs,
    stream_answer,
)
from etsi_mec_agent.store import get_qdrant_client

st.set_page_config(page_title="ETSI MEC RAG", page_icon="📡", layout="wide")

# redirect_stdout swaps the process-wide sys.stdout, and Streamlit runs each session in its own
# thread of one process — without this lock two concurrent searches capture each other's prints.
_search_lock = threading.Lock()

# Diagram files are rendered by filename under the configured dir only: a payload path is data
# from the (possibly remote) collection, not a local file request.
_diag_root = Path(settings.diagrams_dir).resolve()

# Public-deploy gate: set APP_PASSWORD (env var or Streamlit secret) to require a password.
# Unset = open; on a public deploy the platform's own auth (HF Space login gate) is lazier.
_gate = os.getenv("APP_PASSWORD", "")
if not _gate:
    try:
        _gate = st.secrets.get("APP_PASSWORD", "")
    except Exception:
        pass  # no secrets.toml anywhere — the env var is the only source
if _gate and not st.session_state.get("authed"):
    pw = st.text_input("Password", type="password")
    if pw and hmac.compare_digest(pw.encode(), _gate.encode()):
        st.session_state["authed"] = True
        st.rerun()
    st.stop()

with st.sidebar:
    st.title("📡 ETSI MEC RAG")
    endpoint = "Qdrant Cloud" if settings.qdrant_url else f"{settings.qdrant_host}:{settings.qdrant_port}"
    st.caption(f"Collection `{settings.qdrant_index}` @ {endpoint}  \nLLM `{settings.llm_model}`")
    path_label = st.radio(
        "Retrieval path",
        ["Hybrid RRF (dense + sparse)", "Dense + ColBERT re-rank"],
        help="Hybrid fuses dense recall with exact-term sparse matches server-side (RRF). "
             "ColBERT re-ranks a dense prefetch with token-level MaxSim precision.",
    )
    use_bm25 = path_label.startswith("Hybrid")
    current_only = st.checkbox("Current editions only", value=True,
                               help="Excludes superseded-edition chunks (is_current=false).")
    diagrams_only = st.checkbox("Diagrams only", value=False,
                                help="Only chunks whose page yielded an extracted diagram.")
    top_k = st.slider("Excerpts shown (top_k)", 1, 10, 5)
    answer_context = st.slider("Evidence budget (answer_context)", 5, 25, 15,
                               help="Excerpts handed to the LLM — deliberately more than the display budget.")
    per_doc = st.slider("Per-document cap (per_doc)", 1, 4, 2,
                        help="Quota is spent per content_md5: byte-identical copies of one spec share it.")
    want_answer = st.toggle("LLM answer panel", value=False)

tab_search, tab_corpus, tab_about = st.tabs(["🔎 Search", "📚 Corpus", "🧠 How it works"])

# --------------------------------------------------------------------------- Search
with tab_search:
    with st.form("query_form"):
        query = st.text_input("Query", placeholder="What is Mp1? / Mm4 role / MEP permissions")
        submitted = st.form_submit_button("Search", type="primary")

    if submitted and query.strip():
        trace = io.StringIO()
        try:
            with st.spinner("Retrieving… (first query loads the ONNX embedders: ~30–60 s)"):
                with _search_lock, contextlib.redirect_stdout(trace):
                    result = search_specs(
                        query.strip(), top_k=top_k, diagrams_only=diagrams_only,
                        current_only=current_only, use_bm25=use_bm25,
                        answer_context=answer_context, per_doc=per_doc,
                        generate=False, full_result=True,
                    )
        except Exception as exc:
            st.error(f"Retrieval failed — is Qdrant reachable? {exc}")
            st.stop()
        st.session_state.update(result=result, trace=trace.getvalue(), query=query.strip(),
                                search_settings=(use_bm25, current_only, diagrams_only,
                                                 top_k, answer_context, per_doc))
        st.session_state.pop("answer", None)

    result = st.session_state.get("result")
    if result:
        info = result["info"]
        st.caption(f"**{st.session_state['query']}** — {info['path']} · "
                   f"{info.get('fetched', info['stitched'])} fetched → {info['stitched']} stitched → "
                   f"{info['shown']} shown · {info['elapsed_ms']:.0f} ms · "
                   f"{info['merged_groups']} page-part group(s) merged {info['filter_info']}")
        if not result["display"]:
            st.info("No matching documents found.")
        for i, doc in enumerate(result["display"], 1):
            meta = doc.meta or {}
            score, parts = meta.get("score"), meta.get("stitched_parts", 1)
            title = (f"[{i}] {_source_label(meta)}"
                     + (f" · score {score:.3f}" if isinstance(score, (int, float)) else "")
                     + (f" · 🧵 stitched {parts} parts" if parts > 1 else "")
                     + (" · 🖼️ diagram" if meta.get("has_diagram") else ""))
            with st.expander(title):
                head = " · ".join(str(x) for x in (meta.get("clause"), meta.get("heading")) if x)
                if head:
                    st.caption(f"🔖 {head}")
                st.markdown(doc.content or "")
                for dp in meta.get("diagram_paths", []):
                    p = (_diag_root / Path(dp).name).resolve()
                    if p.is_relative_to(_diag_root) and p.exists():
                        st.image(str(p), caption=p.name)
        with st.expander("Pipeline trace (captured stdout)"):
            st.code(st.session_state.get("trace", ""), language="text")

        if want_answer and result["display"]:
            if st.session_state.get("search_settings") != (
                    use_bm25, current_only, diagrams_only, top_k, answer_context, per_doc):
                st.warning("Sidebar settings changed since the last Search — run Search again "
                           "before generating an answer from them.")
            elif st.button("💬 Generate answer", type="primary"):
                meta_out = {}
                st.session_state.pop("answer", None)
                try:
                    answer = st.write_stream(
                        stream_answer(st.session_state["query"], result["answer_docs"], meta_out))
                except Exception as exc:
                    st.error(f"Answer generation failed — is LLM_API_KEY/OPENROUTER_API_KEY set? {exc}")
                    answer = None
                if answer:
                    st.session_state["answer"] = {
                        "text": answer,
                        "model": meta_out.get("model", ""),
                        "sources": _build_context(result["answer_docs"])[1],
                        "suspects": _ungrounded_citations(answer, result["answer_docs"]),
                        "cited": len(_PAGE_RE.findall(_plain(answer))),
                        "has_identity": bool(_pages_by_spec(result["answer_docs"])),
                    }
                st.session_state["just_streamed"] = True

            ans = st.session_state.get("answer")
            if ans:
                if not st.session_state.pop("just_streamed", False):
                    st.markdown("### Answer")
                    st.markdown(ans["text"])
                st.caption(f"[ANSWER MODEL] {ans['model'] or 'unknown'} · "
                           f"[ANSWER SOURCES] {len(ans['sources'])} excerpts fed to the LLM "
                           f"(~{info['answer_words']} words, up to "
                           f"{info.get('per_doc', per_doc)} per document)")
                with st.expander("Evidence fed to the LLM (answer_context budget)"):
                    for j, lbl in enumerate(ans["sources"], 1):
                        st.write(f"[{j}] {lbl}")
                st.markdown("#### Citation grounding audit")
                if not ans["has_identity"]:
                    st.info("Check skipped — the excerpts carry no stamped spec identity "
                            "(run scripts/backfill_spec_identity.py --apply).")
                elif ans["suspects"]:
                    st.warning(f"{ans['cited']} page citation(s), {len(ans['suspects'])} unsupported:")
                    for s in ans["suspects"]:
                        st.write(f"⚠ {s}")
                elif ans["cited"]:
                    st.success(f"{ans['cited']} page citation(s), all present in the retrieved excerpts.")
                elif len(ans["text"].split()) < 40:
                    st.warning("A short answer with no citation is a canned router response, not "
                               "an answer — re-run, or pin a model with LLM_MODEL.")
                else:
                    st.info("The answer cites no page numbers.")

# --------------------------------------------------------------------------- Corpus
with tab_corpus:
    @st.cache_data(ttl=300, show_spinner="Scrolling the collection…")
    def corpus_stats():
        """(total points, one row per spec_id+edition) from a full read-only scroll."""
        client = get_qdrant_client()
        total = client.count(settings.qdrant_index).count
        rows, offset = {}, None
        while True:
            pts, offset = client.scroll(
                settings.qdrant_index, limit=512, offset=offset,
                with_payload=["spec_id", "edition", "is_current", "has_diagram", "filename"],
                with_vectors=False)
            for p in pts:
                pl = p.payload or {}
                key = (pl.get("spec_id") or pl.get("filename") or "?", pl.get("edition") or "?")
                row = rows.setdefault(key, {"current": True, "chunks": 0, "diagrams": 0})
                row["chunks"] += 1
                row["diagrams"] += bool(pl.get("has_diagram"))
                row["current"] = row["current"] and bool(pl.get("is_current", True))
            if offset is None:
                break
        df = pd.DataFrame(
            [(k[0], k[1], v["current"], v["chunks"], v["diagrams"]) for k, v in sorted(rows.items())],
            columns=["spec_id", "edition", "is_current", "chunks", "diagram chunks"])
        return total, df

    try:
        total, df = corpus_stats()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Chunks (points)", f"{total:,}")
        c2.metric("Specifications", df["spec_id"].nunique())
        c3.metric("Spec editions", len(df))
        c4.metric("Diagram chunks", int(df["diagram chunks"].sum()))
        st.dataframe(df, hide_index=True)
    except Exception as exc:
        st.error(f"Cannot reach Qdrant: {exc}")

    graph = Path("etsi_mec_graph.png")
    if graph.exists():
        st.image(str(graph), caption="Spec relationship graph — regenerate: "
                                     "uv run python scripts/graph_visualize.py")

# --------------------------------------------------------------------------- How it works
with tab_about:
    st.markdown(f"""
### Three vectors per chunk, one hybrid query

- **dense** — `{settings.dense_model}`, 384-d Cosine. Global semantic recall.
- **sparse** — raw term frequencies hashed by crc32 (no shared vocabulary file), IDF weighting
  applied server-side (`Modifier.IDF`). Exact identifiers like `Mm4`, `Mp1`, clause numbers.
- **colbert** — `{settings.colbert_model}`, 128-d multi-vectors, MaxSim late interaction, stored
  on disk. Token-level precision re-rank over the dense prefetch.

The **Hybrid RRF** path sends the dense and sparse prefetches into Qdrant and fuses them
server-side with Reciprocal Rank Fusion — no Haystack, no client-side merging. The **ColBERT**
path re-ranks a dense prefetch by MaxSim. Both then stitch page parts (`chunk_part 1 of 2` +
`2 of 2` become one excerpt, so a table is never cut mid-row) and dedup.

### Display and evidence are two budgets

The list above shows at most `top_k` excerpts, one per document. The LLM reads up to
`answer_context` excerpts with at most `per_doc` per document over the *same* stitched list —
that is what keeps the continuation of a long table in the prompt. The per-document quota is
spent per `content_md5`, so byte-identical copies of one spec share a quota instead of each
claiming one.

### Identity comes from the PDF cover, never the filename

59 of the 106 files in `data/specs/` hold another specification's content, and 52 are
byte-identical copies of another file. `scripts/backfill_spec_identity.py` therefore stamps
`spec_id` / `edition` / `pub_date` / `content_md5` / `is_current` onto every stored point, and
citations are labelled by that identity (`MEC-003 V4.1.1 p.15`) — a filename enters a prompt
only as the visible fallback label of an excerpt the backfill has not stamped yet, never as a
stand-in for the identity its own cover declares. Superseded editions are filtered server-side
unless you untick *Current editions only*.

### Citations are audited, not just requested

Every page citation in an answer (`p.15`, `pp. 15`, prose `page 6`) is matched back to the
excerpts the model actually read, after flattening typographic Unicode (models cite with
non-breaking hyphens U+2011 and narrow no-break spaces U+202F). A citation whose spec was never
retrieved, or whose page no excerpt of that spec sits on, is reported as unsupported. An answer
under 40 words with no citation is called what it is: a canned router response.

*Stack: Qdrant (hybrid vectors + server-side RRF) · FastEmbed ONNX (CPU) · OpenAI-compatible LLM
endpoint (`LLM_BASE_URL` / `LLM_MODEL`, default `openrouter/free`) · Streamlit.*
""")
