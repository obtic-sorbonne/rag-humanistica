"""
TEI Structure Auto-Detector
============================
Phase 1 — Header typing: reads teiHeader to identify document type
           (correspondence, archival, literary, parallel edition, generic)
Phase 2 — Body measurement: samples files to measure avg token count per
           natural unit and decides whether sub-unit splitting is needed

Outputs a corpus_config.json that is human-readable and overridable.
"""

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List
from lxml import etree


TEI_NS  = "http://www.tei-c.org/ns/1.0"
NS      = {"tei": TEI_NS}

# ── Document type identifiers ────────────────────────────────────────────────
DOCTYPE_CORRESPONDENCE   = "correspondence"
DOCTYPE_ARCHIVAL         = "archival"
DOCTYPE_LITERARY         = "literary"
DOCTYPE_PARALLEL_EDITION = "parallel_edition"
DOCTYPE_GENERIC          = "generic"

# ── Div type role table ───────────────────────────────────────────────────────
# chunk_unit     → emit this div as a retrieval unit (before size check)
# recurse        → structural wrapper, look inside for content
# skip           → front/back matter, discard
# parallel_root  → contains parallel versions, iterate children
# parallel_version → one language version within a parallel root
DIV_TYPE_ROLES = {
    "chapter":         "chunk_unit",
    "will":            "chunk_unit",
    "letter":          "chunk_unit",
    "codicil":         "chunk_unit",
    "envelope":        "chunk_unit",

    "section":         "recurse",
    "subsection":      "recurse",
    "subsubsection":   "recurse",
    "book":            "recurse",
    "group":           "recurse",
    "text":            "recurse",
    "main":            "recurse",
    "notes":           "recurse",
    "annex":           "recurse",
    "annexe":          "recurse",
    "foreword":        "recurse",

    "toc":             "skip",
    "titlepage":       "skip",
    "liminal":         "skip",
    "advertisement":   "skip",

    "transcription":   "recurse",      # single-lang wrapper in correspondence; parallel detection is structural
    "original":        "parallel_version",
    "translation":     "parallel_version",
}

TOKEN_CEILING = 400


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_text(element) -> str:
    if element is None:
        return ""
    if not isinstance(element.tag, str):
        return ""
    return re.sub(r"\s+", " ", "".join(element.itertext())).strip()


def _rough_token_count(text: str) -> int:
    """Fast approximation: words * 1.3 ≈ subword tokens."""
    return int(len(text.split()) * 1.3)


def _get_div_role(div_type: str) -> str:
    return DIV_TYPE_ROLES.get(div_type.lower(), "recurse")


# ── Phase 1: Header typing ───────────────────────────────────────────────────

def _detect_doctype(root) -> str:
    # Parallel edition: body has original + translation siblings
    body = root.find(".//tei:text/tei:body", NS)
    if body is not None:
        has_original    = body.find(".//tei:div[@type='original']", NS) is not None
        has_translation = body.find(".//tei:div[@type='translation']", NS) is not None
        if has_original and has_translation:
            return DOCTYPE_PARALLEL_EDITION

    header = root.find(".//tei:teiHeader", NS)
    if header is None:
        return DOCTYPE_GENERIC

    if header.find(".//tei:correspDesc", NS) is not None:
        return DOCTYPE_CORRESPONDENCE

    if (header.find(".//tei:msDesc", NS) is not None or
            header.find(".//tei:msContents", NS) is not None):
        return DOCTYPE_ARCHIVAL

    if header.find(".//tei:textDesc", NS) is not None:
        return DOCTYPE_LITERARY

    return DOCTYPE_GENERIC


# ── Phase 2: Body measurement ─────────────────────────────────────────────────

def _measure_file(root) -> Dict:
    body = root.find(".//tei:text/tei:body", NS)
    if body is None:
        body = root
    paras = body.findall(".//tei:p", NS)
    p_texts = [_get_text(p) for p in paras if len(_get_text(p)) > 20]
    avg_tokens_per_p = (
        sum(_rough_token_count(t) for t in p_texts) / len(p_texts)
        if p_texts else 0
    )
    div_types = {
        d.get("type", "").lower()
        for d in body.findall(".//tei:div[@type]", NS)
        if d.get("type")
    }
    chunk_units = [
        d for d in body.findall(".//tei:div[@type]", NS)
        if _get_div_role(d.get("type", "")) == "chunk_unit"
    ]
    return {
        "p_count": len(p_texts),
        "avg_tokens_per_p": avg_tokens_per_p,
        "chunk_unit_count": len(chunk_units),
        "div_type_vocab": div_types,
    }


def _avg_tokens_per_chunk_unit(root) -> float:
    body = root.find(".//tei:text/tei:body", NS)
    if body is None:
        return 0.0
    units = [
        d for d in body.findall(".//tei:div[@type]", NS)
        if _get_div_role(d.get("type", "")) == "chunk_unit"
    ]
    if not units:
        return 0.0
    return sum(_rough_token_count(_get_text(u)) for u in units) / len(units)


# ── Metadata path table ───────────────────────────────────────────────────────

def _build_metadata_paths(doc_type: str) -> List[Dict]:
    """
    Ordered extraction specs per doc_type.
    {"field": str, "xpath": str, "attr": str|None}
    attr=None → extract text content; attr="when" → get @when attribute.
    Multiple entries for the same field are tried in order; first non-empty wins.
    """
    common = [
        {"field": "doc_title",  "xpath": ".//tei:teiHeader//tei:titleStmt/tei:title",  "attr": None},
        {"field": "doc_author", "xpath": ".//tei:teiHeader//tei:titleStmt/tei:author", "attr": None},
        {"field": "doc_date",   "xpath": ".//tei:teiHeader//tei:creation/tei:date",    "attr": "when"},
        {"field": "doc_date",   "xpath": ".//tei:teiHeader//tei:date[@when]",          "attr": "when"},
        {"field": "keywords",   "xpath": ".//tei:teiHeader//tei:keywords/tei:term",    "attr": None},
    ]
    type_specific = {
        DOCTYPE_CORRESPONDENCE: [
            {"field": "sender",       "xpath": ".//tei:correspDesc//tei:correspAction[@type='sent']//tei:persName",     "attr": None},
            {"field": "sender_ref",   "xpath": ".//tei:correspDesc//tei:correspAction[@type='sent']//tei:persName",     "attr": "ref"},
            {"field": "recipient",    "xpath": ".//tei:correspDesc//tei:correspAction[@type='received']//tei:persName", "attr": None},
            {"field": "sent_date",    "xpath": ".//tei:correspDesc//tei:correspAction[@type='sent']/tei:date",          "attr": "when"},
            {"field": "sent_place",   "xpath": ".//tei:correspDesc//tei:correspAction[@type='sent']//tei:placeName",    "attr": None},
        ],
        DOCTYPE_ARCHIVAL: [
            {"field": "summary",      "xpath": ".//tei:teiHeader//tei:msContents/tei:summary",         "attr": None},
            {"field": "collection",   "xpath": ".//tei:teiHeader//tei:msIdentifier/tei:collection",    "attr": None},
        ],
        DOCTYPE_LITERARY: [
            {"field": "author_gender","xpath": ".//tei:teiHeader//tei:textDesc//tei:authorGender",     "attr": "key"},
            {"field": "time_slot",    "xpath": ".//tei:teiHeader//tei:textDesc//tei:timeSlot",         "attr": "key"},
            {"field": "size",         "xpath": ".//tei:teiHeader//tei:textDesc//tei:size",             "attr": "key"},
        ],
        DOCTYPE_PARALLEL_EDITION: [
            {"field": "origin_place", "xpath": ".//tei:teiHeader//tei:creation//tei:placeName",        "attr": None},
            {"field": "origin_ref",   "xpath": ".//tei:teiHeader//tei:creation//tei:placeName",        "attr": "ref"},
            {"field": "origin_date",  "xpath": ".//tei:teiHeader//tei:creation/tei:date",              "attr": "when"},
            {"field": "organisation", "xpath": ".//tei:teiHeader//tei:creation//tei:orgName",          "attr": None},
            {"field": "org_ref",      "xpath": ".//tei:teiHeader//tei:creation//tei:orgName",          "attr": "ref"},
        ],
        DOCTYPE_GENERIC: [],
    }
    return common + type_specific.get(doc_type, [])


# ── Main detection entry point ────────────────────────────────────────────────

def detect_corpus_strategy(corpus_dir: str, sample_size: int = 10) -> Dict:
    corpus_path = Path(corpus_dir)
    xml_files = list(corpus_path.glob("**/*.xml"))
    if not xml_files:
        raise ValueError(f"No XML files found in {corpus_dir}")

    sample = random.sample(xml_files, min(sample_size, len(xml_files)))
    print(f"\n  Probing {len(sample)} files from '{corpus_path.name}'...")

    doc_types: List[str] = []
    measurements: List[Dict] = []
    avg_chunk_tokens_list: List[float] = []
    all_div_vocab: set = set()

    for f in sample:
        try:
            tree = etree.parse(str(f))
            root = tree.getroot()
        except Exception as e:
            print(f"    [probe] Could not parse {f.name}: {e}")
            continue
        doc_types.append(_detect_doctype(root))
        m = _measure_file(root)
        measurements.append(m)
        all_div_vocab.update(m["div_type_vocab"])
        avg_chunk_tokens_list.append(_avg_tokens_per_chunk_unit(root))

    if not doc_types:
        raise ValueError(f"No parseable XML files in {corpus_dir}")

    n = len(measurements)
    doc_type = Counter(doc_types).most_common(1)[0][0]

    mean_avg_p_tokens = sum(m["avg_tokens_per_p"] for m in measurements) / n
    nonzero_chunks = [t for t in avg_chunk_tokens_list if t > 0]
    mean_chunk_tokens = sum(nonzero_chunks) / len(nonzero_chunks) if nonzero_chunks else 0.0
    has_chunk_units = any(m["chunk_unit_count"] > 0 for m in measurements)
    has_parallel = any(
        "original" in m["div_type_vocab"] or "translation" in m["div_type_vocab"]
        for m in measurements
    )

    observed_chunk_units = sorted(t for t in all_div_vocab if _get_div_role(t) == "chunk_unit")
    observed_recurse     = sorted(t for t in all_div_vocab if _get_div_role(t) == "recurse")

    # Splitting decision
    if has_chunk_units:
        needs_paragraph_split = mean_chunk_tokens > TOKEN_CEILING
    else:
        needs_paragraph_split = True

    metadata_paths = _build_metadata_paths(doc_type)

    print(f"\n  Doc type (majority vote, n={n}): {doc_type}")
    print(f"  Chunk-unit div types: {observed_chunk_units or '(none, paragraph mode)'}")
    print(f"  Recurse div types: {observed_recurse}")
    print(f"  Has parallel versions: {has_parallel}")
    print(f"  Mean tokens/paragraph: {mean_avg_p_tokens:.0f}")
    print(f"  Mean tokens/chunk-unit: {mean_chunk_tokens:.0f}  (ceiling: {TOKEN_CEILING})")
    print(f"  Needs paragraph split within chunk units: {needs_paragraph_split}")

    return {
        "corpus_name":           corpus_path.name,
        "corpus_dir":            str(corpus_path),
        "doc_type":              doc_type,
        "chunk_unit_types":      observed_chunk_units,
        "recurse_types":         observed_recurse,
        "needs_paragraph_split": needs_paragraph_split,
        "parallel_versions":     has_parallel,
        "token_ceiling":         TOKEN_CEILING,
        "metadata_paths":        metadata_paths,
        "features_summary": {
            "n_sampled":             n,
            "mean_tokens_per_para":  round(mean_avg_p_tokens, 1),
            "mean_tokens_per_chunk": round(mean_chunk_tokens, 1),
            "div_type_vocab":        sorted(all_div_vocab),
            "doc_type_votes":        dict(Counter(doc_types)),
        },
        "sample_files": [f.name for f in sample],
    }


# ── Config persistence ─────────────────────────────────────────────────────────

def save_corpus_config(config: Dict, vector_store_path: str):
    config_path = Path(vector_store_path) / "corpus_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"  Saved corpus config → {config_path}")


def load_corpus_config(vector_store_path: str) -> Dict:
    config_path = Path(vector_store_path) / "corpus_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No corpus_config.json at {config_path}")
    with open(config_path) as f:
        return json.load(f)


# Legacy constants (kept for rag_app.py compatibility)
STRATEGY_SECTION   = "tei_section"
STRATEGY_PARAGRAPH = "tei_paragraph"
STRATEGY_NAIVE     = "naive"