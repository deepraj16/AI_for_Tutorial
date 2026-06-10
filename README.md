# RAG Pipeline for MSETCL Technical Manuals

## Project Overview

Build a **Retrieval-Augmented Generation (RAG)** system for querying MSETCL (Maharashtra State Electricity Transmission Company Ltd) technical manuals. The system uses **offline local embeddings** (zero API cost) paired with an online LLM API (Mistral / GPT) for answering user queries.

---

## 1. PDF Source Documents

| # | PDF File | Pages | Size | Type | Content Description |
|:-:|:---|:---:|:---:|:---:|:---|
| 1 | `Document from Deepraj.pdf` | 288 | 13.3 MB | **Digital** | Transmission Lines Construction Manual |
| 2 | `Document from Deepraj (3).pdf` | 325 | 34.3 MB | **Digital** | EHV Sub-station Construction Manual Vol I |
| 3 | `Document from Deepraj (1).pdf` | 227 | 68.0 MB | **Scanned** | Maintenance Procedure Manual for EHV Sub-stations |
| 4 | `Document from Deepraj (2).pdf` | 98 | 159.0 MB | **Scanned** | Environment, Health and Safety (EHS) Manual |
| 5 | `ETM STAMP VOL I.pdf` | 390 | 33.7 MB | **Scanned** | Equipment Testing Manual – Power Transformers, CTs, PTs etc. |

**Total: 1,328 pages**

---

## 2. Text Extraction Strategy

### Digital PDFs (613 pages) → Direct Extraction
- **Tool:** `pypdf` (PdfReader)
- **Method:** `page.extract_text()` — extracts selectable text directly
- **Speed:** Very fast (~1 second per 100 pages)
- **Quality:** High fidelity, no noise

### Scanned PDFs (715 pages) → OCR Pipeline
- **Rendering:** `PyMuPDF` (`fitz`) renders each page to a 150 DPI PNG image
- **OCR Engine:** `Tesseract OCR v5.5.0` installed at `C:\Program Files\Tesseract-OCR\`
- **Python Wrapper:** `pytesseract`
- **Speed:** ~2-5 seconds per page (CPU dependent)
- **Quality:** Good for printed text; minor noise on diagrams/tables

> **Why PyMuPDF + Tesseract instead of just pypdf?**
> Scanned PDFs store pages as embedded images (TIFF/JPEG). `pypdf` can extract these raw image layers, but they are often split into multiple fragments (background, foreground, masks) which don't OCR correctly. PyMuPDF renders the complete visual page as a single image, giving Tesseract a clean input.

---

## 3. Data Analysis (Page 12 Samples)

Before designing the chunking strategy, page 12 was extracted from each PDF and analyzed:

### Key Observations

| Observation | Details |
|:---|:---|
| **Document structure** | All 5 manuals use numbered hierarchical sections (e.g., `1.1`, `1.1A`, `3.2.1`, `56.3`) |
| **Content type** | Dense technical paragraphs, regulatory clauses, equipment specifications, procedures |
| **Avg line length** | Digital: ~75 chars/line · OCR: ~49 chars/line |
| **Tables/Figures** | Present but OCR converts them to messy text; acceptable for retrieval |
| **Headers/Footers** | Repeating headers like `MSETCL/EHSM/June25/Version 1.0`, standalone page numbers |
| **OCR noise** | Stray 1-2 char lines (`aN`, `BY`, `$`), `&amp;` artifacts, Unicode replacement chars |

### Sample Extracted Content

**Digital PDF — `Document from Deepraj.pdf` (Page 12):**
```
Chapter 1 — Philosophy and Methodology of Projects Planning and Execution For Msetcl Network
...is one of the most important component in ARR. In MSETCL, there are only two types schemes
by which the capitalization can be achieved i.e. by Projects Schemes (for creation of new assets)
and Life Extension or Renovation & Modernization schemes...
```

**Scanned PDF — `ETM STAMP VOL I.pdf` (Page 12):**
```
1.1 Principle information
1.1A- Principle of Operation
Transformer is a static device which transfers a.c. electrical power from one circuit to the other
at the same frequency through magnetic flux linking both the circuit, by electro-magnetic induction...
```

---

## 4. Preprocessing Pipeline

The following cleaning steps are applied to all extracted text before chunking:

| Step | What it does | Why |
|:---|:---|:---|
| **Remove OCR noise lines** | Drops lines with ≤ 2 characters (non-digit) | Eliminates stray chars like `aN`, `BY`, `$` from OCR |
| **Remove page headers/footers** | Regex strips `MSETCL/...`, `MAHATRANSCO...`, standalone page numbers | These repeat on every page and pollute search results |
| **Fix Unicode artifacts** | Replaces curly quotes `''"" → '"`, em/en dashes `—– → -`, `■ → ""` | Normalizes text for consistent tokenization |
| **Fix HTML entities** | `&amp; → &` | Common OCR misread |
| **Normalize whitespace** | Multiple spaces → single, 3+ newlines → 2 | Prevents empty chunks and wasted tokens |
| **Strip** | Trim leading/trailing whitespace | Clean chunk boundaries |

---

## 5. Chunking Strategy

### Chosen: Recursive Character Text Splitting with Overlap

```
Chunk Size:    500 characters
Chunk Overlap: 100 characters
Separators:    ['\n\n', '\n', '. ', ' ']  (tried in order)
```

### Why these parameters?

| Parameter | Value | Rationale |
|:---|:---:|:---|
| **Chunk Size = 500 chars** | 500 | `bge-small-en-v1.5` has a 512-token context window. English technical text averages ~1 token per 4-5 chars, so 500 chars ≈ 100-125 tokens — safely within limits while capturing a full paragraph of context |
| **Overlap = 100 chars** | 100 | Technical documents reference previous sentences heavily ("as mentioned in section 1.1A above..."). Overlap ensures no context is lost at chunk boundaries |
| **Separator priority** | `\n\n` first | Splits at paragraph breaks first (preserving logical sections), then line breaks, then sentences, then words as last resort |

### How Recursive Splitting Works

```
Input Text (1500 chars)
    │
    ├─ Try split by '\n\n' (paragraph breaks)
    │   ├─ Paragraph 1 (400 chars) → ✅ Chunk 1
    │   ├─ Paragraph 2 (800 chars) → Too long, try next separator
    │   │   ├─ Try split by '\n' (line breaks)
    │   │   │   ├─ Lines fit → ✅ Chunk 2, Chunk 3
    │   └─ Paragraph 3 (300 chars) → ✅ Chunk 4
    │
    └─ Apply 100-char overlap between consecutive chunks
        ├─ Chunk 1: [original]
        ├─ Chunk 2: [last 100 chars of Chunk 1] + [Chunk 2]
        ├─ Chunk 3: [last 100 chars of Chunk 2] + [Chunk 3]
        └─ Chunk 4: [last 100 chars of Chunk 3] + [Chunk 4]
```

---

## 6. Embedding Model

### Model: `BAAI/bge-small-en-v1.5`

| Property | Value |
|:---|:---|
| **Library** | `sentence-transformers` (v5.5.1) |
| **Model size** | ~130 MB |
| **Vector dimension** | 384 |
| **Max tokens** | 512 |
| **Runs on** | CPU (no GPU required) |
| **Cost** | **$0.00** — fully offline, no API calls |
| **Cached at** | `C:\Users\raj\.cache\huggingface\hub\models--BAAI--bge-small-en-v1.5` |

### Why this model?

- Top-ranked on the MTEB (Massive Text Embedding Benchmark) for retrieval tasks in its size class
- 384-dimensional vectors are compact → fast FAISS search, low memory usage
- Excellent for English technical/domain-specific text

---

## 7. Vector Store (FAISS)

### Index Type: `IndexFlatIP` (Flat Inner Product)

| Property | Value |
|:---|:---|
| **Library** | `faiss-cpu` (v1.11.0) |
| **Index type** | `IndexFlatIP` — exact inner product search |
| **Similarity metric** | Cosine similarity (vectors are L2-normalized before indexing) |
| **Search speed** | Exact brute-force — fast enough for <50K vectors |
| **Persistence** | Saved as `vector_store/index.faiss` |

### Stored Metadata (per chunk)

Each chunk in the index has associated metadata stored in `vector_store/chunks_metadata.pkl`:

```python
{
    "source": "ETM STAMP VOL I.pdf",   # source PDF filename
    "page": 12,                         # original page number
    "chunk_length": 487                 # character count of this chunk
}
```

---

## 8. Output Files

After running `build_vector_store.py`, the following files are created:

```
vector_store/
├── index.faiss            # FAISS binary index (all vectors)
├── chunks_metadata.pkl    # Python pickle: list of chunk texts + metadata dicts
└── store_info.json        # Human-readable summary (chunk count, model info, etc.)
```

---

## 9. Query Flow (Future)

```
User Query
    │
    ▼
┌─────────────────────────────┐
│ Embed query with             │
│ BAAI/bge-small-en-v1.5      │  ← Offline, $0.00
│ (same model used for index)  │
└─────────────┬───────────────┘
              │ 384-dim vector
              ▼
┌─────────────────────────────┐
│ FAISS Similarity Search      │
│ index.search(query_vec, k=5) │  ← Offline, $0.00
└─────────────┬───────────────┘
              │ Top-5 relevant chunks
              ▼
┌─────────────────────────────┐
│ Build prompt:                │
│ "Context: [chunks]"         │
│ "Question: [user query]"    │
│                             │
│ Send to LLM API             │  ← Only API cost (per query)
│ (GPT-4o / Mistral)          │
└─────────────┬───────────────┘
              │
              ▼
         Final Answer
```

---

## 10. How to Run

### Prerequisites
```
pip install sentence-transformers faiss-cpu pymupdf pypdf pytesseract Pillow
```
Tesseract OCR must be installed at `C:\Program Files\Tesseract-OCR\`

### Build the Vector Store
```
python build_vector_store.py
```
> ⚠️ First run takes **15-30 minutes** due to OCR on 715 scanned pages. Subsequent runs can skip OCR if extracted text is cached.

---

## 11. Installed Packages Summary

| Package | Version | Purpose |
|:---|:---|:---|
| `pypdf` | — | Digital PDF text extraction |
| `pymupdf` (fitz) | 1.27.2 | Render scanned PDF pages to images |
| `pytesseract` | — | Python wrapper for Tesseract OCR |
| `Pillow` (PIL) | — | Image processing for OCR pipeline |
| `sentence-transformers` | 5.5.1 | Load & run embedding model locally |
| `faiss-cpu` | 1.11.0 | Vector similarity search index |
| `numpy` | 1.26.4 | Numerical operations |
| `torch` | — | Backend for sentence-transformers |
