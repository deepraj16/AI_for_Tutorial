"""
MSETCL RAG Chatbot · Streamlit UI (Standard Native & Mobile-Optimized)
Run with: streamlit run app.py
"""

import streamlit as st

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="MSETCL Assistant",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="auto",
)

# ─────────────────────────────────────────
# LOAD RAG (cached)
# ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_rag():
    from rag_graph import ask, chunks, metadata, faiss_index
    return ask, len(chunks), metadata, faiss_index.ntotal

# ─────────────────────────────────────────
# LOADING SCREEN
# ─────────────────────────────────────────
if "rag_ready" not in st.session_state:
    st.title("⚡ MSETCL Assistant")
    st.info("Initializing knowledge base... Please wait.")
    with st.status("Loading RAG System Components...", expanded=True) as status:
        st.write("Connecting to Mistral AI and loading embeddings...")
        ask_fn, total_chunks, meta, total_vectors = load_rag()
        status.update(label="System Ready!", state="complete", expanded=False)

    st.session_state["rag_ready"]     = True
    st.session_state["ask_fn"]        = ask_fn
    st.session_state["total_chunks"]  = total_chunks
    st.session_state["total_vectors"] = total_vectors
    st.rerun()

else:
    ask_fn        = st.session_state["ask_fn"]
    total_chunks  = st.session_state["total_chunks"]
    total_vectors = st.session_state["total_vectors"]

# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
if "messages" not in st.session_state: 
    st.session_state.messages = []

# ═══════════════════════════════
#  LEFT SIDEBAR
# ═══════════════════════════════
with st.sidebar:
    st.title("⚡ MSETCL Assistant")
    st.caption("RAG-powered document Q&A")
    st.divider()

    # ── Indexed documents ──
    st.subheader("Indexed Documents")
    docs_info = [
        ("Transmission Lines Construction Manual",         "~320 pages"),
        ("Maintenance Procedure Manual (EHV Sub-stations)","~280 pages"),
        ("Environment, Health & Safety (EHS) Manual",      "~210 pages"),
        ("EHV Sub-station Construction Manual Vol I",      "~310 pages"),
        ("Equipment Testing Manual Vol I (ETM)",           "~208 pages"),
    ]
    for name, pages in docs_info:
        st.markdown(f"📄 **{name}**\n*{pages}*")

    st.divider()

    # ── Stats + clear ──
    st.caption(f"● {total_chunks:,} chunks · {total_vectors:,} vectors")
    st.caption("Model: mistral-large")

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ═══════════════════════════════
#  MAIN CHAT AREA
# ═══════════════════════════════

# ── Header bar ──
st.title("⚡ MSETCL Document Assistant")
st.write("Ask technical questions from MSETCL transmission, EHV, EHS, and testing manuals.")

# ── Empty state with suggestion buttons ──
if not st.session_state.messages:
    suggestions = [
        "What is the daily log sheet procedure for EHV lines?",
        "Explain how a power transformer works.",
        "What PPE is required for MSETCL employees?",
        "How is a Permit to Work issued?",
        "How is the earth mat designed for a substation?",
        "What tests are done at transformer pre-commissioning?",
    ]
    
    st.info("Select a suggestion below or type your question in the chat box to begin:")
    cols = st.columns(2)
    for i, s in enumerate(suggestions):
        if cols[i % 2].button(s, key=f"sug_{i}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": s})
            st.rerun()

# ── Render conversation history ──
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        if msg["role"] == "assistant" and "sources" in msg:
            direct = [d for d in msg["sources"] if not d.get("is_neighbor")]
            d_ct = len(direct)
            if d_ct > 0:
                with st.expander(f"📎 View {d_ct} sources", expanded=False):
                    for doc in direct:
                        preview = doc["text"][:180]
                        if len(doc["text"]) > 180:
                            preview += "…"
                        st.markdown(f"**{doc['source']}** (pg {doc['page']})")
                        st.caption(f"Score: {doc['score']:.3f}")
                        st.write(preview)
                        st.divider()

# ── Chat input ──
if prompt := st.chat_input("Ask anything about MSETCL technical manuals…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

# ── Handle query execution ──
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    user_query = st.session_state.messages[-1]["content"]
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base…"):
            result = ask_fn(user_query)
        st.markdown(result["answer"])
        
        # Render the expander for sources immediately
        direct = [d for d in result["retrieved_docs"] if not d.get("is_neighbor")]
        d_ct = len(direct)
        if d_ct > 0:
            with st.expander(f"📎 View {d_ct} sources", expanded=False):
                for doc in direct:
                    preview = doc["text"][:180]
                    if len(doc["text"]) > 180:
                        preview += "…"
                    st.markdown(f"**{doc['source']}** (pg {doc['page']})")
                    st.caption(f"Score: {doc['score']:.3f}")
                    st.write(preview)
                    st.divider()
        
    st.session_state.messages.append({
        "role":    "assistant",
        "content": result["answer"],
        "sources": result["retrieved_docs"],
    })
    st.rerun()