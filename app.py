"""
MSETCL RAG Chatbot · Streamlit UI (Standard Native & Mobile-Optimized)
Run with: streamlit run app.py
"""

import streamlit as st
import base64
import os

def get_image_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""


# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="TransGuru GPT",
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
    st.title("⚡ TransGuru GPT")
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
    st.image("Mahatransco Logo-01.png", use_container_width=True)
    st.title("⚡ TransGuru GPT")
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
pslc_logo_b64 = get_image_base64("PSLC logo.png")

header_html = f"""
<div style="display: flex; align-items: center; justify-content: space-between; gap: 15px; margin-bottom: 20px; width: 100%;">
    <div style="flex: 0 0 auto; visibility: hidden;">
        <img src="data:image/png;base64,{pslc_logo_b64}" style="height: 60px; max-width: 100%; object-fit: contain;">
    </div>
    <div style="flex: 1 1 auto; text-align: center;">
        <h1 style="margin: 0; font-size: 2.2rem; font-weight: 700; font-family: 'Inter', sans-serif; color: inherit;">⚡TransGuru</h1>
        <p style="margin: 5px 0 0 0; font-size: 1rem; color: inherit; opacity: 0.7;">Ask questions in the chatbox below.</p>
    </div>
    <div style="flex: 0 0 auto;">
        <img src="data:image/png;base64,{pslc_logo_b64}" style="height: 60px; max-width: 100%; object-fit: contain; filter: drop-shadow(0px 2px 4px rgba(0,0,0,0.15));">
    </div>
</div>
"""

st.markdown(header_html, unsafe_allow_html=True)
st.write("")


# ── Empty state with suggestion buttons ──
if not st.session_state.messages:
    col1, col2, col3 = st.columns([3, 2, 3])
    with col2:
        st.image("WhatsApp Image 2026-06-12 at 8.27.06 AM (1).png", use_container_width=True)
        
    suggestions = [
        "What is the daily log sheet procedure for EHV lines?",
        "What PPE is required for MSETCL employees?",
        "How is a Permit to Work issued?",
    ]
    
    st.markdown("<p style='text-align: center; font-size: 0.95rem; margin-top: 10px;'>Or select a suggestion to begin:</p>", unsafe_allow_html=True)
    cols = st.columns(3)
    for i, s in enumerate(suggestions):
        if cols[i].button(s, key=f"sug_{i}", use_container_width=True):
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
