"""
ragHumanistica — Unified Multi-Corpus RAG App
=============================================
Launch with:
    streamlit run pipeline/rag_app.py --server.port 8505

Features:
- Corpus-agnostic: auto-loads any indexed corpus
- TEI-aware display: semantic term highlighting, breadcrumbs, rich metadata
- Post-retrieval filters: date range, keywords, chunk type
- Prompt testbed: named presets, side-by-side A/B comparison, export log
"""

import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import streamlit as st
import torch
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "pipeline"))

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

# ── Constants ─────────────────────────────────────────────────────────────────
CORPUS_BASE     = PROJECT_ROOT / "data" / "corpus"
STORE_BASE      = PROJECT_ROOT / "data" / "vector_stores"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

TESTBED_LOG_PATH = PROJECT_ROOT / "data" / "prompt_testbed_log.jsonl"

DEFAULT_PROMPTS = {
    "Generic DH": (
        "Tu es un assistant spécialisé dans les éditions numériques en humanités.\n"
        "Réponds uniquement en te basant sur les documents fournis.\n"
        "Cite toujours tes sources avec la date et le titre du document."
    ),
    "Historical analyst": (
        "You are a historical analyst working with digital editions.\n"
        "Answer precisely based only on the provided documents.\n"
        "Always cite your sources and note the date and document title.\n"
        "If information is absent from the sources, state it explicitly."
    ),
    "Minimal / retrieval": (
        "Answer based on the documents provided. Be concise."
    ),
}

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ragHumanistica",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
mark { background:#fff3cd; padding:2px 4px; border-radius:3px; cursor:help; }
mark:hover { background:#ffc107; }
.metadata-card {
    background:#f8f9fa; padding:1rem; border-radius:.5rem;
    border-left:4px solid #0d6efd; margin-bottom:1rem;
}
.breadcrumb { font-size:.9em; color:#6c757d; padding:.3rem 0; }
.corpus-badge {
    display:inline-block; background:#e9ecef; border-radius:4px;
    padding:2px 8px; font-size:.8em; color:#495057; margin-right:4px;
}
.testbed-box {
    background:#f0f4ff; border:1px solid #c5d3f0; border-radius:.5rem;
    padding:1rem; margin-bottom:1rem;
}
</style>
""", unsafe_allow_html=True)


# ── Utilities ─────────────────────────────────────────────────────────────────

def clean_markup(text: str) -> str:
    return re.sub(r"\s+", " ", BeautifulSoup(text, "lxml").get_text()).strip()


def extract_tei_terms(raw_text: str) -> list:
    """Extract <term ref="#...">label</term> pairs from raw TEI text."""
    return [
        {"ref": ref, "text": txt}
        for ref, txt in re.findall(r'<term[^>]*ref=["\']#([^"\']+)["\'][^>]*>([^<]+)</term>', raw_text)
    ]


def highlight_terms(text: str, terms: list) -> str:
    for t in terms:
        text = text.replace(t["text"], f'<mark title="{t["ref"]}">{t["text"]}</mark>', 1)
    return text


def extract_year(date_str: str) -> Optional[int]:
    if not date_str or date_str == "N/A":
        return None
    try:
        if "-" in date_str:
            return int(date_str.split("-")[0])
        if "/" in date_str:
            parts = date_str.split("/")
            if len(parts) == 3:
                return int(parts[2])
    except Exception:
        pass
    return None


def format_date(date_str: str) -> str:
    if not date_str or date_str == "N/A":
        return "N/A"
    if "-" in date_str:
        try:
            p = date_str.split("-")
            if len(p) == 3:
                return f"{p[2]}/{p[1]}/{p[0]}"
        except Exception:
            pass
    return date_str


def safe_get(obj, *keys, default="N/A"):
    try:
        result = obj
        for key in keys:
            result = result.get(key, default) if isinstance(result, dict) else default
        return result if result is not None else default
    except Exception:
        return default


def trunc(s: str, n: int) -> str:
    return (s[:n] + "…") if len(s) > n else s


# ── Corpus discovery ──────────────────────────────────────────────────────────

def discover_corpora() -> dict:
    result = {}
    if not CORPUS_BASE.exists():
        return result
    for d in sorted(CORPUS_BASE.iterdir()):
        if not d.is_dir():
            continue
        store_dir = STORE_BASE / d.name
        has_index = (store_dir / "docstore.json").exists()
        config = None
        cfg_path = store_dir / "corpus_config.json"
        if cfg_path.exists():
            try:
                with open(cfg_path) as f:
                    config = json.load(f)
            except Exception:
                pass
        result[d.name] = {"has_index": has_index, "config": config}
    return result


# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource
def load_embed_model():
    return HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL, device="cpu", trust_remote_code=True
    )


@st.cache_resource
def load_corpus_index(corpus_name: str):
    Settings.embed_model = load_embed_model()
    Settings.llm = None
    ctx = StorageContext.from_defaults(persist_dir=str(STORE_BASE / corpus_name))
    return load_index_from_storage(ctx)


@st.cache_resource
def load_llm(model_name: str, temperature: float, max_tokens: int, prompt_hash: str):
    system_prompt = st.session_state.get("system_prompt", "")
    with st.spinner(f"Loading {model_name}…"):
        try:
            llm = HuggingFaceLLM(
                model_name=model_name,
                tokenizer_name=model_name,
                model_kwargs={"torch_dtype": torch.float16, "low_cpu_mem_usage": True},
                generate_kwargs={"temperature": temperature, "do_sample": True},
                max_new_tokens=max_tokens,
                device_map="auto",
                system_prompt=system_prompt,
            )
            Settings.llm = llm
            return llm
        except Exception as e:
            st.error(f"LLM load error: {e}")
            return None


# ── Post-retrieval filtering ──────────────────────────────────────────────────

def apply_filters(nodes, year_range=None, kw_filter=None, type_filter=None):
    """Filter retrieved nodes by year range, keyword substring, chunk type."""
    filtered = []
    for node in nodes:
        meta = node.metadata
        # Year filter
        if year_range:
            y = extract_year(safe_get(meta, "doc_date"))
            if y and not (year_range[0] <= y <= year_range[1]):
                continue
        # Keyword filter
        if kw_filter:
            kw_text = safe_get(meta, "keywords", default="").lower()
            doc_text = node.text.lower()
            if kw_filter.lower() not in kw_text and kw_filter.lower() not in doc_text:
                continue
        # Chunk type filter
        if type_filter:
            ct = safe_get(meta, "chunk_type", default="")
            if ct not in type_filter:
                continue
        filtered.append(node)
    return filtered


# ── Display components ────────────────────────────────────────────────────────

def display_metadata_card(metadata: dict, doc_type: str = "", show_file: bool = True):
    st.markdown('<div class="metadata-card">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Temporal**")
        doc_date = safe_get(metadata, "doc_date")
        st.caption(f"Date: {format_date(doc_date)}")
        y = extract_year(doc_date)
        if y:
            st.caption(f"Year: {y}")
        # Correspondence-specific
        sent_date = safe_get(metadata, "sent_date", default="")
        if sent_date and sent_date != "N/A":
            st.caption(f"Sent: {format_date(sent_date)}")

    with col2:
        st.markdown("**Attribution**")
        author = safe_get(metadata, "doc_author")
        st.caption(f"Author: {trunc(author, 40)}")
        # Correspondence
        sender = safe_get(metadata, "sender", default="")
        recipient = safe_get(metadata, "recipient", default="")
        if sender and sender != "N/A":
            st.caption(f"From: {trunc(sender, 35)}")
        if recipient and recipient != "N/A":
            st.caption(f"To: {trunc(recipient, 35)}")
        # Archival
        summary = safe_get(metadata, "summary", default="")
        if summary and summary != "N/A":
            st.caption(f"Summary: {trunc(summary, 60)}")

    with col3:
        st.markdown("**Structure**")
        chunk_type = safe_get(metadata, "chunk_type")
        lang = safe_get(metadata, "language", default="")
        type_label = f"{chunk_type}" + (f" [{lang}]" if lang and lang != "N/A" else "")
        st.caption(f"Type: {type_label}")
        title = safe_get(metadata, "title")
        if title and title != "N/A":
            st.caption(f"Section: {trunc(title, 50)}")
        # Literary metadata
        for field, label in [("author_gender", "Gender"), ("time_slot", "Period"), ("size", "Size")]:
            val = safe_get(metadata, field, default="")
            if val and val != "N/A":
                st.caption(f"{label}: {val}")

    # Document title
    doc_title = safe_get(metadata, "doc_title")
    if doc_title and doc_title != "N/A":
        st.caption(f"Document: {trunc(doc_title, 100)}")

    # File name
    if show_file:
        fname = safe_get(metadata, "file_name", default="")
        if fname and fname != "N/A":
            st.caption(f"File: {fname}")

    # Keywords
    keywords = safe_get(metadata, "keywords", default="")
    if keywords and keywords != "N/A":
        kws = [k.strip() for k in keywords.split(",") if k.strip()][:8]
        st.markdown(" ".join(f"`{k}`" for k in kws))

    # Badges
    badges = []
    if doc_type:
        badges.append(f"doc_type: {doc_type}")
    cs = safe_get(metadata, "corpus_strategy", default="")
    if cs and cs != "N/A":
        badges.append(f"strategy: {cs}")
    pages = safe_get(metadata, "page_numbers", default="")
    if pages and pages != "N/A":
        badges.append(f"pp. {pages}")
    if badges:
        st.markdown(
            " ".join(f'<span class="corpus-badge">{b}</span>' for b in badges),
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


def display_source(node, index: int, doc_type: str = "", expanded: bool = False):
    metadata = node.metadata
    title = safe_get(metadata, "title")
    sender = safe_get(metadata, "sender", default="")
    lang = safe_get(metadata, "language", default="")

    label = f"Source {index} — score: {node.score:.3f}"
    if title and title != "N/A":
        label += f" — {trunc(title, 40)}"
    elif sender and sender != "N/A":
        label += f" — {trunc(sender, 40)}"
    if lang and lang != "N/A":
        label += f" [{lang}]"

    with st.expander(label, expanded=expanded):
        # Breadcrumb if title contains hierarchy
        if title and " > " in title:
            st.markdown(
                f'<div class="breadcrumb">📍 {title.replace(" > ", " › ")}</div>',
                unsafe_allow_html=True,
            )

        display_metadata_card(metadata, doc_type=doc_type)

        st.markdown("**Content**")
        raw = node.text
        terms = extract_tei_terms(raw)
        cleaned = clean_markup(raw)

        if terms:
            highlighted = highlight_terms(cleaned, terms)
            preview = highlighted[:3600] + "…" if len(highlighted) > 3600 else highlighted
            st.markdown(preview, unsafe_allow_html=True)
            with st.expander(f"Semantic terms ({len(terms)})"):
                for t in terms[:15]:
                    st.caption(f"• {t['text']}  `{t['ref']}`")
        else:
            preview = cleaned[:1800] + "…" if len(cleaned) > 1800 else cleaned
            st.text_area("", preview, height=200,
                         key=f"src_{index}_{node.node_id[:8]}", disabled=True)


def display_analytics(nodes):
    if not nodes:
        return
    st.subheader("Results analytics")
    years, types, kws_all = [], [], []
    for node in nodes:
        y = extract_year(safe_get(node.metadata, "doc_date"))
        if y:
            years.append(y)
        types.append(safe_get(node.metadata, "chunk_type", default="unknown"))
        kw = safe_get(node.metadata, "keywords", default="")
        if kw and kw != "N/A":
            kws_all.extend(k.strip() for k in kw.split(",") if k.strip())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Results", len(nodes))
    c2.metric("Time span",
              f"{min(years)}–{max(years)}" if len(years) > 1 else (str(years[0]) if years else "N/A"))
    c3.metric("Main type", Counter(types).most_common(1)[0][0] if types else "N/A")
    c4.metric("Unique keywords", len(set(kws_all)))

    if len(years) > 1:
        with st.expander("Temporal distribution"):
            st.bar_chart(Counter(years))
    if kws_all:
        with st.expander("Top keywords"):
            for kw, n in Counter(kws_all).most_common(10):
                st.caption(f"• {kw} ({n})")


# ── Prompt testbed ────────────────────────────────────────────────────────────

def log_testbed_entry(entry: dict):
    TESTBED_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TESTBED_LOG_PATH, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def render_testbed_sidebar(corpus_cfg: dict) -> dict:
    """
    Renders prompt testbed controls in the sidebar.
    Returns a dict with all testbed settings.
    """
    st.sidebar.markdown("---")
    with st.sidebar.expander("🧪 Prompt testbed", expanded=False):
        st.caption("Design, compare and log prompt variants for the paper.")

        # Preset loader
        preset_name = st.selectbox("Load preset", ["(custom)"] + list(DEFAULT_PROMPTS.keys()))
        if preset_name != "(custom)":
            preset_val = DEFAULT_PROMPTS[preset_name]
        else:
            preset_val = st.session_state.get("system_prompt", list(DEFAULT_PROMPTS.values())[0])

        system_prompt = st.text_area(
            "System prompt (A)",
            value=preset_val,
            height=150,
            key="system_prompt",
        )
        prompt_hash = hashlib.md5(system_prompt.encode()).hexdigest()[:8]

        # A/B mode
        ab_mode = st.checkbox("A/B comparison (two prompts)")
        prompt_b = ""
        hash_b = ""
        if ab_mode:
            prompt_b = st.text_area("System prompt (B)", height=100, key="prompt_b")
            hash_b = hashlib.md5(prompt_b.encode()).hexdigest()[:8]

        # Annotation
        st.markdown("**Log entry**")
        run_label = st.text_input("Run label", placeholder="e.g. exp-01 minimal-prompt")
        annotator_note = st.text_input("Note", placeholder="hypothesis or observation")
        log_enabled = st.checkbox("Auto-log results", value=False)

        # Export
        if TESTBED_LOG_PATH.exists():
            with open(TESTBED_LOG_PATH) as f:
                log_content = f.read()
            st.download_button(
                "⬇ Export log (JSONL)",
                data=log_content,
                file_name="prompt_testbed_log.jsonl",
                mime="application/json",
            )
            entries = [l for l in log_content.strip().split("\n") if l]
            st.caption(f"{len(entries)} entries logged")

    return {
        "system_prompt": system_prompt,
        "prompt_hash": prompt_hash,
        "ab_mode": ab_mode,
        "prompt_b": prompt_b,
        "hash_b": hash_b,
        "run_label": run_label,
        "annotator_note": annotator_note,
        "log_enabled": log_enabled,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.title("📜 ragHumanistica")
    st.caption("Multi-corpus RAG for Digital Editions · ERC ModERN · Sorbonne/ENS Paris")

    corpora  = discover_corpora()
    indexed  = {k: v for k, v in corpora.items() if v["has_index"]}
    unindexed = {k: v for k, v in corpora.items() if not v["has_index"]}

    if not corpora:
        st.error(f"No corpus directories found under {CORPUS_BASE}")
        st.stop()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.subheader("📚 Corpus")

    if not indexed:
        st.sidebar.warning("No indexed corpora. Run corpus_indexer.py --all")
        st.stop()

    selected_corpus = st.sidebar.selectbox(
        "Select corpus", options=list(indexed.keys()), key="selected_corpus"
    )
    corpus_cfg = indexed[selected_corpus].get("config") or {}
    doc_type   = corpus_cfg.get("doc_type", corpus_cfg.get("strategy", ""))

    if corpus_cfg:
        st.sidebar.caption(
            f"Type: `{doc_type}` | "
            f"{corpus_cfg.get('total_chunks','?')} chunks | "
            f"{corpus_cfg.get('total_files','?')} files"
        )
        st.sidebar.caption(f"Indexed: {corpus_cfg.get('indexed_at','?')[:10]}")

    if unindexed:
        with st.sidebar.expander(f"⚠️ {len(unindexed)} corpora not indexed"):
            for name in unindexed:
                st.caption(f"• {name}")

    st.sidebar.markdown("---")

    # Model settings
    with st.sidebar.expander("🤖 Model settings", expanded=True):
        retrieval_only = st.checkbox("Retrieval only (no LLM)", value=True)
        model_choice = st.selectbox(
            "LLM",
            ["mistralai/Mistral-7B-Instruct-v0.3",
             "meta-llama/Llama-3.2-3B-Instruct",
             "HuggingFaceH4/zephyr-7b-beta"],
            disabled=retrieval_only,
        )
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1, disabled=retrieval_only)
        max_tokens  = st.slider("Max tokens", 128, 2048, 512, 128, disabled=retrieval_only)

    # Retrieval settings
    with st.sidebar.expander("🔍 Retrieval", expanded=True):
        top_k          = st.slider("Top-k results", 1, 20, 5)
        show_analytics = st.checkbox("Show analytics", value=True)
        expand_first   = st.checkbox("Expand first result", value=True)

    # Post-retrieval filters
    with st.sidebar.expander("🎛️ Filters (post-retrieval)", expanded=False):
        use_year = st.checkbox("Filter by year")
        year_range = None
        if use_year:
            year_range = st.slider("Year range", 1800, 2024, (1900, 1960))

        kw_filter = st.text_input("Keyword in text/metadata", placeholder="e.g. pension")
        kw_filter = kw_filter.strip() or None

        type_filter = st.multiselect(
            "Chunk type",
            ["chapter", "section", "paragraph", "will", "letter",
             "transcription", "document", "parallel_original", "parallel_translation"],
        ) or None

    # Prompt testbed
    testbed = render_testbed_sidebar(corpus_cfg)

    # Example queries
    st.sidebar.markdown("---")
    st.sidebar.subheader("💡 Examples")
    examples = [
        "Quel est le sujet principal de ce corpus?",
        "Quels personnages sont mentionnés?",
        "Quels événements historiques sont décrits?",
        "Quelle est la période couverte?",
    ]
    for ex in examples:
        if st.sidebar.button(ex, key=f"ex_{ex[:15]}", use_container_width=True):
            st.session_state.query_input = ex
            st.rerun()

    # System info
    st.sidebar.markdown("---")
    with st.sidebar.expander("🖥️ System"):
        if torch.cuda.is_available():
            props    = torch.cuda.get_device_properties(0)
            used_gb  = torch.cuda.memory_allocated(0) / 1e9
            total_gb = props.total_memory / 1e9
            st.metric("GPU", f"{used_gb:.1f}/{total_gb:.1f} GB")
        else:
            st.info("CPU mode")

    # ── Load index ────────────────────────────────────────────────────────────
    with st.spinner(f"Loading '{selected_corpus}'…"):
        try:
            index = load_corpus_index(selected_corpus)
        except Exception as e:
            st.error(f"Failed to load index: {e}")
            st.stop()

    st.success(
        f"✓ **{selected_corpus}** — "
        f"{corpus_cfg.get('total_chunks','?')} chunks · "
        f"doc_type: `{doc_type}`"
    )

    # ── Search interface ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔍 Search")

    query = st.text_input(
        "Query", placeholder="Search the corpus…", key="query_input"
    )
    search_btn = st.button("Search", type="primary", use_container_width=True)

    if not (search_btn and query):
        return

    # ── Retrieve ──────────────────────────────────────────────────────────────
    with st.spinner("Searching…"):
        retriever = index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)

    nodes = apply_filters(nodes, year_range=year_range,
                          kw_filter=kw_filter, type_filter=type_filter)

    if not nodes:
        st.warning("No results after filtering. Try relaxing the filters.")
        return

    # ── Retrieval-only path ───────────────────────────────────────────────────
    if retrieval_only:
        st.markdown("---")
        st.subheader("Retrieved documents")
        if show_analytics:
            display_analytics(nodes)
            st.markdown("---")
        for i, node in enumerate(nodes, 1):
            display_source(node, i, doc_type=doc_type, expanded=(i == 1 and expand_first))

        if testbed["log_enabled"] and testbed["run_label"]:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "corpus": selected_corpus,
                "doc_type": doc_type,
                "query": query,
                "mode": "retrieval_only",
                "top_k": top_k,
                "n_results": len(nodes),
                "run_label": testbed["run_label"],
                "note": testbed["annotator_note"],
                "top_scores": [round(n.score, 4) for n in nodes[:5]],
            }
            log_testbed_entry(entry)
            st.info(f"Logged run: {testbed['run_label']}")
        return

    # ── Full RAG path ─────────────────────────────────────────────────────────
    llm = load_llm(model_choice, temperature, max_tokens, testbed["prompt_hash"])
    if llm is None:
        st.stop()

    query_engine = index.as_query_engine(similarity_top_k=top_k)

    # A/B mode
    if testbed["ab_mode"] and testbed["prompt_b"]:
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown('<div class="testbed-box">', unsafe_allow_html=True)
            st.markdown(f"**Prompt A** `{testbed['prompt_hash']}`")
            with st.spinner("Generating A…"):
                resp_a = query_engine.query(query)
            st.markdown(clean_markup(str(resp_a)))
            st.markdown("</div>", unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="testbed-box">', unsafe_allow_html=True)
            st.markdown(f"**Prompt B** `{testbed['hash_b']}`")
            # Reload LLM with prompt B
            st.session_state["system_prompt"] = testbed["prompt_b"]
            llm_b = load_llm(model_choice, temperature, max_tokens, testbed["hash_b"])
            if llm_b:
                resp_b = query_engine.query(query)
                st.markdown(clean_markup(str(resp_b)))
            st.markdown("</div>", unsafe_allow_html=True)

        # Restore prompt A
        st.session_state["system_prompt"] = testbed["system_prompt"]

        st.markdown("---")
        st.subheader("Sources (Prompt A)")
        for i, node in enumerate(resp_a.source_nodes, 1):
            display_source(node, i, doc_type=doc_type, expanded=(i == 1 and expand_first))

    else:
        with st.spinner("Generating…"):
            response = query_engine.query(query)

        st.markdown("---")
        st.subheader("Response")
        st.markdown(clean_markup(str(response)))

        st.markdown("---")
        st.subheader("Sources")
        if show_analytics:
            display_analytics(response.source_nodes)
            st.markdown("---")
        for i, node in enumerate(response.source_nodes, 1):
            display_source(node, i, doc_type=doc_type, expanded=(i == 1 and expand_first))

        if testbed["log_enabled"] and testbed["run_label"]:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "corpus": selected_corpus,
                "doc_type": doc_type,
                "query": query,
                "mode": "rag",
                "model": model_choice,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "prompt_hash": testbed["prompt_hash"],
                "system_prompt": testbed["system_prompt"],
                "top_k": top_k,
                "n_results": len(response.source_nodes),
                "response_preview": clean_markup(str(response))[:1500],
                "run_label": testbed["run_label"],
                "note": testbed["annotator_note"],
                "top_scores": [round(n.score, 4) for n in response.source_nodes[:5]],
            }
            log_testbed_entry(entry)
            st.info(f"Logged run: {testbed['run_label']}")


if __name__ == "__main__":
    main()
    