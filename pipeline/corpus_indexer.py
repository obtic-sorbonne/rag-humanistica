"""
Multi-Corpus Indexer for ragHumanistica
========================================
Adaptive chunking pipeline driven by tei_detector.py config.

Chunking logic (in priority order):
  1. Walk div hierarchy using DIV_TYPE_ROLES
  2. At each chunk_unit div: if text > TOKEN_CEILING tokens, split into paragraphs
  3. If a paragraph still exceeds TOKEN_CEILING, hand off to SentenceSplitter
  4. Leaf recurse-divs with no chunk_unit children are treated as chunk_units
  5. Parallel editions: emit one chunk per language version with lang metadata

Usage:
    python pipeline/corpus_indexer.py --corpus AVH_corpus
    python pipeline/corpus_indexer.py --all
    python pipeline/corpus_indexer.py --corpus fra-eltec --force
    python pipeline/corpus_indexer.py --status
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "pipeline"))

from lxml import etree
from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from tei_detector import (
    detect_corpus_strategy,
    save_corpus_config,
    DIV_TYPE_ROLES,
    TOKEN_CEILING,
    NS,
    _get_text,
    _rough_token_count,
    _get_div_role,
)

CORPUS_BASE     = PROJECT_ROOT / "data" / "corpus"
STORE_BASE      = PROJECT_ROOT / "data" / "vector_stores"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
TEI_NS          = NS

# Metadata field truncation limits (prevent LlamaIndex node budget overflow)
TRUNCATE_FIELDS = {
    "doc_title":  200,
    "abstract":   300,
    "keywords":   300,
    "doc_author": 150,
    "summary":    400,
}

# Sentence splitter for oversized paragraphs — reused across files
_sentence_splitter = SentenceSplitter(chunk_size=TOKEN_CEILING, chunk_overlap=50)

_embed_model = None


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
        _embed_model = HuggingFaceEmbedding(
            model_name=EMBEDDING_MODEL,
            device="cuda",
            trust_remote_code=True,
        )
        Settings.embed_model = _embed_model
        Settings.llm = None
    return _embed_model


# ── Metadata extraction ───────────────────────────────────────────────────────

def extract_metadata(root, metadata_paths: List[Dict]) -> Dict:
    """
    Apply ordered metadata_paths from corpus config.
    Multiple entries with the same field name: first non-empty wins.
    Multi-valued xpath (e.g. keywords): join with ', '.
    """
    meta: Dict[str, str] = {}
    for spec in metadata_paths:
        field = spec["field"]
        if field in meta:
            continue  # already filled
        xpath = spec["xpath"]
        attr  = spec.get("attr")
        try:
            elements = root.findall(xpath, NS)
        except Exception:
            continue
        if not elements:
            continue
        if attr:
            values = [e.get(attr, "") for e in elements if e.get(attr)]
        else:
            values = [_get_text(e) for e in elements]
        values = [v for v in values if v.strip()]
        if values:
            meta[field] = ", ".join(values)

    # Ensure standard fields always present (even if empty)
    for f in ("doc_title", "doc_author", "doc_date", "keywords"):
        meta.setdefault(f, "")

    return meta


def _truncate_metadata(meta: Dict) -> Dict:
    for field, limit in TRUNCATE_FIELDS.items():
        if field in meta and len(meta[field]) > limit:
            meta[field] = meta[field][:limit] + "…"
    return meta


# ── Text splitting ────────────────────────────────────────────────────────────

def _split_text_if_large(text: str, base_id: str, meta: Dict,
                          ceiling: int = TOKEN_CEILING) -> List[Document]:
    """
    If text fits within ceiling, return one Document.
    Otherwise split at sentence boundaries and return multiple Documents.
    """
    if _rough_token_count(text) <= ceiling:
        return [Document(text=text, metadata=dict(meta), id_=base_id)]

    # Use SentenceSplitter via its text-splitting method
    nodes = _sentence_splitter.get_nodes_from_documents(
        [Document(text=text, metadata=dict(meta), id_=base_id)]
    )
    docs = []
    for i, node in enumerate(nodes):
        m = dict(meta)
        m["chunk_index"] = i
        docs.append(Document(
            text=node.get_content(),
            metadata=m,
            id_=f"{base_id}_s{i}",
        ))
    return docs


# ── Paragraph extraction within a div ────────────────────────────────────────

def _paragraphs_from_div(div, base_meta: Dict, file_stem: str,
                          prefix: str, section_title: str = "") -> List[Document]:
    """
    Extract <p> elements from div, splitting oversized paragraphs.
    prefix is used for stable chunk IDs.
    """
    docs = []
    paras = div.findall(".//tei:p", NS)
    for i, p in enumerate(paras):
        text = _get_text(p)
        if len(text.strip()) < 30:
            continue
        full_text = f"[{section_title}]\n\n{text}" if section_title else text
        xml_id = p.get("{http://www.w3.org/XML/1998/namespace}id", "")
        meta = dict(base_meta)
        meta.update({
            "chunk_type": "paragraph",
            "title":      section_title,
            "xml_id":     xml_id,
        })
        chunk_id = xml_id or f"{prefix}_p{i}"
        docs.extend(_split_text_if_large(full_text, chunk_id, meta))
    return docs


# ── Div tree walker ───────────────────────────────────────────────────────────

def _walk_div(div, base_meta: Dict, file_stem: str,
              needs_paragraph_split: bool, depth: int = 0) -> List[Document]:
    """
    Recursively walk the div tree according to DIV_TYPE_ROLES.
    Returns a flat list of Documents.
    """
    div_type = div.get("type", "").lower()
    role = _get_div_role(div_type) if div_type else "recurse"
    xml_id = div.get("{http://www.w3.org/XML/1998/namespace}id", "")
    head = div.find("tei:head", NS)
    title = _get_text(head) if head is not None else ""

    if role == "skip":
        return []

    if role == "parallel_root":
        # Recurse into each child version div
        docs = []
        for child in div:
            if not isinstance(child.tag, str):
                continue
            child_role = _get_div_role(child.get("type", "").lower())
            if child_role == "parallel_version":
                lang = child.get("{http://www.w3.org/XML/1998/namespace}lang", "")
                meta = dict(base_meta)
                meta["language"] = lang
                meta["chunk_type"] = f"parallel_{child.get('type','version')}"
                docs.extend(_walk_div(child, meta, file_stem,
                                      needs_paragraph_split, depth + 1))
        return docs

    if role == "parallel_version":
        # Reached directly (not via parallel_root) — treat as chunk unit
        role = "chunk_unit"

    if role == "chunk_unit":
        meta = dict(base_meta)
        meta.update({
            "chunk_type": div_type or "section",
            "title":      title,
            "xml_id":     xml_id,
        })
        prefix = xml_id or f"{file_stem}_{div_type}_{depth}"

        if needs_paragraph_split:
            # Split into paragraphs (and further if any para is oversized)
            docs = _paragraphs_from_div(div, meta, file_stem, prefix, title)
            if docs:
                return docs
            # No paragraphs found — fall back to whole-div text
            text = _get_text(div)
            if len(text.strip()) < 30:
                return []
            full_text = f"[{title}]\n\n{text}" if title else text
            return _split_text_if_large(full_text, prefix, meta)
        else:
            # Try whole-div text first
            text = _get_text(div)
            if len(text.strip()) < 30:
                return []
            full_text = f"[{title}]\n\n{text}" if title else text
            return _split_text_if_large(full_text, prefix, meta)

    # role == "recurse": look for child divs to recurse into
    child_divs = [c for c in div if isinstance(c.tag, str)
                  and etree.QName(c.tag).localname == "div"]

    if not child_divs:
        # Leaf recurse-div: no child divs, extract text-bearing children
        # (handles <opener>, <p>, <closer>, <postscript> etc. in correspondence)
        meta = dict(base_meta)
        meta.update({
            "chunk_type": div_type or "section",
            "title":      title,
            "xml_id":     xml_id,
        })
        prefix = xml_id or f"{file_stem}_{div_type}_{depth}"

        # Try paragraph-level extraction first
        text_children = [
            c for c in div.iter()
            if isinstance(c.tag, str)
            and etree.QName(c.tag).localname in
                ("p", "opener", "closer", "postscript", "dateline", "salute", "signed", "ab")
        ]
        if text_children and needs_paragraph_split:
            docs = []
            for i, child in enumerate(text_children):
                text = _get_text(child)
                if len(text.strip()) < 30:
                    continue
                m = dict(meta)
                m["chunk_type"] = etree.QName(child.tag).localname
                child_id = child.get("{http://www.w3.org/XML/1998/namespace}id", "")
                docs.extend(_split_text_if_large(text, child_id or f"{prefix}_c{i}", m))
            if docs:
                return docs

        # Fall back to whole-div text
        text = _get_text(div)
        if len(text.strip()) < 30:
            return []
        full_text = f"[{title}]\n\n{text}" if title else text
        return _split_text_if_large(full_text, prefix, meta)

    docs = []
    for child in child_divs:
        docs.extend(_walk_div(child, base_meta, file_stem,
                              needs_paragraph_split, depth + 1))
    return docs


# ── File-level extractor ──────────────────────────────────────────────────────

def extract_documents(filepath: Path, config: Dict) -> List[Document]:
    """
    Extract Documents from one TEI file using the corpus config.
    """
    try:
        tree = etree.parse(str(filepath))
        root = tree.getroot()
    except Exception as e:
        print(f"    [skip] Parse error {filepath.name}: {e}")
        return []

    base_meta = extract_metadata(root, config["metadata_paths"])
    base_meta["file_name"]       = filepath.name
    base_meta["doc_type"]        = config["doc_type"]
    base_meta["corpus_strategy"] = "adaptive"
    _truncate_metadata(base_meta)

    # Namespace-agnostic body search: try TEI NS, then bare namespace, then no NS
    body = root.find(".//tei:text/tei:body", NS)
    if body is None:
        body = root.find(".//{http://www.tei-c.org/ns/1.0}body")
    if body is None:
        body = root.find(".//body")
    if body is None:
        # No body element found: fall back to full-text single document
        text_el = (root.find(".//tei:text", NS) or
                   root.find(".//{http://www.tei-c.org/ns/1.0}text") or
                   root.find(".//text") or root)
        text = _get_text(text_el)
        if text.strip():
            base_meta["chunk_type"] = "document"
            return [Document(text=text, metadata=base_meta,
                             id_=f"{filepath.stem}_doc")]
        return []

    needs_split = config.get("needs_paragraph_split", False)
    docs = []

    # localname-based div matching handles both namespaced and bare files
    top_divs = [c for c in body if isinstance(c.tag, str)
                and etree.QName(c.tag).localname == "div"]

    if not top_divs:
        # No divs: paragraph fallback, also localname-based
        paras = [e for e in body.iter()
                 if isinstance(e.tag, str) and etree.QName(e.tag).localname == "p"]
        for i, p in enumerate(paras):
            text = _get_text(p)
            if len(text.strip()) < 30:
                continue
            meta = dict(base_meta)
            meta["chunk_type"] = "paragraph"
            docs.extend(_split_text_if_large(text, f"{filepath.stem}_p{i}", meta))
        return docs

    for div in top_divs:
        docs.extend(_walk_div(div, base_meta, filepath.stem,
                              needs_split, depth=0))

    return docs


# ── Corpus index builder ──────────────────────────────────────────────────────

def build_corpus_index(corpus_name: str, force: bool = False) -> bool:
    corpus_dir = CORPUS_BASE / corpus_name
    store_dir  = STORE_BASE / corpus_name

    if not corpus_dir.exists():
        print(f"[ERROR] Corpus directory not found: {corpus_dir}")
        return False

    if store_dir.exists() and not force:
        if (store_dir / "docstore.json").exists():
            print(f"[SKIP] Index exists for '{corpus_name}'. Use --force to rebuild.")
            return False

    store_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Building index: {corpus_name}")
    print(f"{'='*60}")

    config = detect_corpus_strategy(str(corpus_dir))

    xml_files = sorted(corpus_dir.glob("**/*.xml"))
    print(f"\n  Found {len(xml_files)} XML files")

    all_docs = []
    for i, f in enumerate(xml_files, 1):
        docs = extract_documents(f, config)
        all_docs.extend(docs)
        if i % 20 == 0 or i == len(xml_files):
            print(f"    Processed {i}/{len(xml_files)} files → {len(all_docs)} chunks")

    if not all_docs:
        print(f"[ERROR] No documents extracted from {corpus_name}")
        return False

    print(f"\n  Total chunks: {len(all_docs)}")

    get_embed_model()
    print(f"\n  Generating embeddings...")

    # SentenceSplitter as safety net only — our extractor already handles splitting.
    # Use a generous chunk_size so metadata length never triggers the ValueError.
    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=50)
    index = VectorStoreIndex.from_documents(
        all_docs,
        transformations=[splitter],
        show_progress=True,
    )
    index.storage_context.persist(str(store_dir))

    config["indexed_at"]   = datetime.now().isoformat()
    config["total_chunks"] = len(all_docs)
    config["total_files"]  = len(xml_files)
    save_corpus_config(config, str(store_dir))

    print(f"\n  Index saved → {store_dir}")
    print(f"  Strategy: adaptive / doc_type: {config['doc_type']}")
    print(f"  Paragraph split: {config['needs_paragraph_split']}")
    print(f"  Chunks: {len(all_docs)} from {len(xml_files)} files")
    return True


# ── CLI helpers ───────────────────────────────────────────────────────────────

def get_all_corpus_names() -> List[str]:
    return sorted(d.name for d in CORPUS_BASE.iterdir() if d.is_dir())


def print_status():
    corpora = get_all_corpus_names()
    print(f"\n{'Corpus':<45} {'Index':<10} {'DocType':<22} {'Split':<7} {'Chunks':<8} {'Indexed'}")
    print("-" * 115)
    for name in corpora:
        store_dir   = STORE_BASE / name
        has_index   = (store_dir / "docstore.json").exists()
        config_path = store_dir / "corpus_config.json"
        if has_index and config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            doc_type = cfg.get("doc_type", cfg.get("strategy", "?"))
            chunks   = cfg.get("total_chunks", "?")
            indexed  = cfg.get("indexed_at", "?")[:10]
            split    = "yes" if cfg.get("needs_paragraph_split") else "no"
            status   = "✓"
        else:
            doc_type = "-"; chunks = "-"; indexed = "-"; split = "-"; status = "✗ missing"
        print(f"{name:<45} {status:<10} {doc_type:<22} {split:<7} {str(chunks):<8} {indexed}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ragHumanistica adaptive corpus indexer")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--corpus", type=str, help="Corpus folder name under data/corpus/")
    group.add_argument("--all",    action="store_true", help="Build all missing indexes")
    group.add_argument("--status", action="store_true", help="Print index status table")
    parser.add_argument("--force", action="store_true", help="Rebuild even if index exists")

    args = parser.parse_args()

    if args.status:
        print_status()
    elif args.all:
        corpora = get_all_corpus_names()
        print(f"Processing {len(corpora)} corpora...")
        results = {name: "✓" if build_corpus_index(name, force=args.force) else "skipped/error"
                   for name in corpora}
        print("\nSummary:")
        for name, s in results.items():
            print(f"  {name}: {s}")
    else:
        build_corpus_index(args.corpus, force=args.force)