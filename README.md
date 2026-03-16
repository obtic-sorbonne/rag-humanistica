# ragHumanistica

Multi-corpus RAG for TEI-encoded digital editions.  
ERC ModERN · Sorbonne/ENS Paris

---

## What this is

A retrieval-augmented generation pipeline that uses the structural and semantic markup of TEI-XML editions as the basis for chunking and metadata-aware retrieval. Instead of naive fixed-size splitting, the system detects the document type of each corpus automatically and applies a three-level chunking cascade driven by TEI structure.

Built for the [Humanistica 2026](https://humanistica2026.fr/) paper:  
**"Faire du neuf avec du balisé : Quand une édition TEI devient la mémoire d'un RAG"**

---

## Architecture

```
TEI XML files
     │
     ▼
tei_detector.py        ← auto-detects corpus type and chunking strategy
     │
     ▼
corpus_indexer.py      ← walks div hierarchy, extracts chunks + metadata,
     │                    embeds with multilingual-e5-large, persists to FAISS
     ▼
data/vector_stores/<corpus_name>/
     │
     ▼
rag_app.py             ← Streamlit interface: retrieval, filters, A/B prompt testbed
```

### Chunking cascade

1. `tei_detector` samples the corpus, votes on document type (correspondence / literary / parallel edition / archival / generic), and measures mean token count per natural structural unit (`div[@type]` classified as `chunk_unit`).
2. If mean tokens per unit exceed the ceiling (400), units are split into `<p>` elements.
3. Paragraphs still exceeding the ceiling are split at sentence boundaries.
4. `teiHeader` metadata (author, date, section title, div type, language) is propagated to every chunk and available as post-retrieval filters.

---

## Corpora

Tested on six TEI corpora:

| Corpus | Type | Files |
|--------|------|-------|
| AVH_corpus | archival / periodicals | bulletins 1919–1957 |
| Paul_d_Estournelles_de_Constant | correspondence | letters WWI |
| editionTestamentsDePoilus | archival | wills |
| ehri-online-editions-main | parallel edition | Holocaust docs |
| fra-eltec | literary | ELTeC French novels |
| eng-eltec | literary | ELTeC English novels |

Corpus XML files are not included in this repo (too large). Place them under `data/corpus/<corpus_name>/`.

---

## Setup

### Prerequisites

- Python 3.10–3.12
- NVIDIA GPU recommended (tested on A40 48GB); CPU works for retrieval-only mode
- CUDA 12.x (for GPU embedding)

### Install

```bash
git clone https://github.com/obtic-sorbonne/ragHumanistica.git
cd ragHumanistica

python -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA (adjust to your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

### Data layout

```
data/
  corpus/
    AVH_corpus/          ← put TEI XML files here
    fra-eltec/
    ...
  vector_stores/         ← auto-created by corpus_indexer.py
```

---

## Usage

### 1. Index a corpus

```bash
python pipeline/corpus_indexer.py --corpus AVH_corpus
```

Index all corpora at once:

```bash
python pipeline/corpus_indexer.py --all
```

Check index status:

```bash
python pipeline/corpus_indexer.py --status
```

Force rebuild:

```bash
python pipeline/corpus_indexer.py --corpus fra-eltec --force
```

### 2. Launch the app

```bash
streamlit run pipeline/rag_app.py --server.port 8505
```

The interface lets you:
- Select any indexed corpus
- Run retrieval-only queries (default) or full RAG with a local LLM
- Filter results by date range, keyword, or chunk type
- Use the prompt testbed for A/B comparison and JSONL logging

---

## Models

| Component | Model |
|-----------|-------|
| Embeddings | `intfloat/multilingual-e5-large` |
| LLM (optional) | `mistralai/Mistral-7B-Instruct-v0.3`, `HuggingFaceH4/zephyr-7b-beta`, `meta-llama/Llama-3.2-3B-Instruct` |

Models are downloaded automatically by HuggingFace on first use. LLM is only loaded when retrieval-only mode is disabled.

---

## Project structure

```
ragHumanistica/
├── pipeline/
│   ├── tei_detector.py       # corpus type detection + chunking strategy
│   ├── corpus_indexer.py     # indexing pipeline (TEI → chunks → embeddings)
│   └── rag_app.py            # Streamlit app
├── data/
│   ├── corpus/               # TEI XML files (not in git)
│   └── vector_stores/        # FAISS indexes (not in git)
├── requirements.txt
├── setup.sh
└── README.md
```

---

## Citation

```bibtex
@inproceedings{raghumanistica2026,
  title     = {Faire du neuf avec du balisé : Quand une édition TEI devient la mémoire d'un RAG},
  author    = {Castellon, Clément, Chiffoleau, Floriane et Miasnikova, Alina}
  booktitle = {Humanistica 2026},
  year      = {2026}
}
```

---

## License

[To be specified]