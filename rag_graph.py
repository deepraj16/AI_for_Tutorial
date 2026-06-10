

"""
RAG Pipeline using LangGraph
- Loads FAISS vector store from disk
- Embeds user query locally using BAAI/bge-small-en-v1.5
- Retrieves top-k relevant chunks + their neighbors (prev/next)
- Merges consecutive chunks before sending to LLM
- Generates answer using Mistral LLM via API
"""

import os
import pickle
import numpy as np
import faiss
from typing import TypedDict, List
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────
VECTOR_STORE_DIR  = "vector_store"
FAISS_INDEX_PATH  = os.path.join(VECTOR_STORE_DIR, "index.faiss")
METADATA_PATH     = os.path.join(VECTOR_STORE_DIR, "chunks_metadata.pkl")
EMBEDDING_MODEL   = "BAAI/bge-small-en-v1.5"
TOP_K             = 5          # number of chunks to retrieve via FAISS
MISTRAL_API_KEY   = "lHcwga2vJ6yyjV470WdMIFn5hRgtMbcc"
MISTRAL_MODEL     = "mistral-large-latest"

# ──────────────────────────────────────────
# PDF FILENAME → PROPER DOCUMENT NAME MAP
# ──────────────────────────────────────────
PDF_NAME_MAP = {
    "Document from Deepraj.pdf":     "Transmission Lines Construction Manual",
    "Document from Deepraj (1).pdf": "Maintenance Procedure Manual (EHV Sub-stations)",
    "Document from Deepraj (2).pdf": "Environment, Health and Safety (EHS) Manual",
    "Document from Deepraj (3).pdf": "EHV Sub-station Construction Manual Vol I",
    "ETM STAMP VOL I.pdf":           "Equipment Testing Manual Vol I (ETM)",
}

# ──────────────────────────────────────────
# LOAD RESOURCES ONCE AT STARTUP
# ──────────────────────────────────────────
print("Loading embedding model...")
embedder = SentenceTransformer(EMBEDDING_MODEL)

print("Loading FAISS index...")
faiss_index = faiss.read_index(FAISS_INDEX_PATH)

print("Loading chunk metadata...")
with open(METADATA_PATH, "rb") as f:
    store_data = pickle.load(f)
chunks   = store_data["chunks"]
metadata = store_data["metadata"]

print(f"Vector store ready: {faiss_index.ntotal} vectors loaded.")

llm = ChatMistralAI(
    model=MISTRAL_MODEL,
    api_key=MISTRAL_API_KEY,
)

# ──────────────────────────────────────────
# LANGGRAPH STATE
# ──────────────────────────────────────────
class RAGState(TypedDict):
    question:       str
    retrieved_docs: List[dict]   # [{text, source, page, score, is_neighbor}]
    context:        str
    answer:         str

# ──────────────────────────────────────────
# NODE 1: Retrieve relevant chunks + neighbors
# ──────────────────────────────────────────
def retrieve(state: RAGState) -> RAGState:
    question = state["question"]

    # Embed query offline
    query_vec = embedder.encode(
        [question],
        normalize_embeddings=True
    ).astype("float32")

    # Search FAISS index
    scores, indices = faiss_index.search(query_vec, TOP_K)

    retrieved_docs = []
    seen_indices   = set()  # avoid duplicate chunks

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        # Expand: previous chunk, current chunk, next chunk
        neighbor_indices = [idx - 1, idx, idx + 1]

        for nidx in neighbor_indices:
            # Bounds check + dedup
            if nidx < 0 or nidx >= len(chunks) or nidx in seen_indices:
                continue

            # Only include neighbors from the SAME source PDF
            # (prevents bleeding across document boundaries)
            if metadata[nidx]["source"] != metadata[idx]["source"]:
                continue

            seen_indices.add(nidx)
            raw_filename = metadata[nidx]["source"]
            retrieved_docs.append({
                "text":        chunks[nidx],
                "source":      PDF_NAME_MAP.get(raw_filename, raw_filename),
                "filename":    raw_filename,
                "page":        metadata[nidx]["page"],
                "score":       float(score) if nidx == idx else 0.0,
                "is_neighbor": nidx != idx,
                "chunk_idx":   nidx,
            })

    return {**state, "retrieved_docs": retrieved_docs}

# ──────────────────────────────────────────
# NODE 2: Build context string
#         Merges consecutive chunks from the same source
#         so the LLM sees complete, unbroken passages
# ──────────────────────────────────────────
def build_context(state: RAGState) -> RAGState:
    docs = state["retrieved_docs"]

    if not docs:
        return {**state, "context": "No relevant documents found."}

    # Sort by (filename, chunk_index) for coherent reading order
    docs_sorted = sorted(docs, key=lambda d: (d["filename"], d["chunk_idx"]))

    # Merge consecutive chunks from the same source into single passage blocks
    merged_blocks = []
    i = 0
    while i < len(docs_sorted):
        current      = docs_sorted[i]
        merged_text  = current["text"]
        start_page   = current["page"]

        # Greedily absorb the next chunk if it's consecutive & same source
        while i + 1 < len(docs_sorted):
            nxt = docs_sorted[i + 1]
            if (nxt["filename"] == current["filename"] and
                    nxt["chunk_idx"] == current["chunk_idx"] + 1):
                merged_text += " " + nxt["text"]
                current      = nxt
                i           += 1
            else:
                break

        merged_blocks.append({
            "source":     current["source"],
            "filename":   current["filename"],
            "page":       f"{start_page}–{current['page']}" if start_page != current["page"] else str(start_page),
            "text":       merged_text,
            "score":      docs_sorted[i]["score"],
        })
        i += 1

    # Build final context string for the LLM
    parts = []
    for j, block in enumerate(merged_blocks, 1):
        parts.append(
            f"[Source {j}: {block['source']}, Page {block['page']}]\n{block['text']}"
        )

    context = "\n\n---\n\n".join(parts)
    return {**state, "context": context}

# ──────────────────────────────────────────
# NODE 3: Generate answer with Mistral
# ──────────────────────────────────────────
def generate(state: RAGState) -> RAGState:
    system_prompt = (
        "You are an expert assistant for MSETCL (Maharashtra State Electricity "
        "Transmission Company Ltd) technical manuals. Answer the user's question "
        "using ONLY the provided context. Be precise and concise. "
        "Always mention the source document and page number when relevant. "
        "If the context does not contain the answer, say 'I could not find this "
        "information in the provided documents.'"
    )

    user_prompt = (
        f"Context from documents:\n\n{state['context']}\n\n"
        f"Question: {state['question']}"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    response = llm.invoke(messages)
    return {**state, "answer": response.content}

# ──────────────────────────────────────────
# BUILD LANGGRAPH
# ──────────────────────────────────────────
def build_rag_graph():
    graph = StateGraph(RAGState)

    graph.add_node("retrieve",      retrieve)
    graph.add_node("build_context", build_context)
    graph.add_node("generate",      generate)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve",      "build_context")
    graph.add_edge("build_context", "generate")
    graph.add_edge("generate",      END)

    return graph.compile()

rag_graph = build_rag_graph()

# ──────────────────────────────────────────
# PUBLIC FUNCTION for Streamlit to call
# ──────────────────────────────────────────
def ask(question: str) -> dict:
    """
    Run the full RAG graph for a given question.
    Returns:
        {
            "answer":         str,
            "retrieved_docs": [{text, source, page, score, is_neighbor}, ...]
        }
    """
    initial_state: RAGState = {
        "question":       question,
        "retrieved_docs": [],
        "context":        "",
        "answer":         "",
    }
    final_state = rag_graph.invoke(initial_state)
    return {
        "answer":         final_state["answer"],
        "retrieved_docs": final_state["retrieved_docs"],
    }


if __name__ == "__main__":
    # Quick CLI test
    q = "What is the daily log sheet procedure for EHV lines?"
    print(f"\nQuestion: {q}\n")
    result = ask(q)
    print("Answer:\n", result["answer"])
    print("\nSources:")
    for doc in result["retrieved_docs"]:
        neighbor_tag = " [neighbor]" if doc.get("is_neighbor") else ""
        print(f"  - {doc['source']} | Page {doc['page']} | Score: {doc['score']:.4f}{neighbor_tag}")