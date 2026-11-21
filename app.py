import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from brain import ask_gemini_stream
from rag import (
    load_embedding_model,
    load_collection,
    process_pdf,
    retrieve_context,
    build_context,
    get_indexed_documents,
    delete_document,
    clear_database
)
from prompts import SYSTEM_PROMPT


# ----------------------------------------------------
# 1. PAGE CONFIGURATION & THEME STYLING
# ----------------------------------------------------

st.set_page_config(
    page_title="MechMate AI — Mechanical Engineering Study Companion",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional UI & Clean Typography
st.markdown("""
<style>
    /* Global tweaks */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }
    
    /* Header branding */
    .mechmate-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
    }
    .mechmate-title {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .mechmate-subtitle {
        font-size: 0.95rem;
        color: #718096;
        margin: 0.2rem 0 0 0;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.12);
        color: #10b981;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    /* Welcome / Starter Cards */
    .hero-card {
        background: rgba(248, 250, 252, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .capability-tag {
        display: inline-block;
        background: rgba(99, 102, 241, 0.1);
        color: #818cf8;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }

    /* Source Citation Cards */
    .source-box {
        background: rgba(128, 128, 128, 0.06);
        border-left: 3px solid #3b82f6;
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 8px;
    }
    .source-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-file { background: rgba(59, 130, 246, 0.15); color: #60a5fa; }
    .badge-subject { background: rgba(168, 85, 247, 0.15); color: #c084fc; }
    .badge-page { background: rgba(234, 179, 8, 0.15); color: #facc15; }

    /* Custom button tweaks */
    div[data-testid="stExpander"] {
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------
# 2. API KEY VALIDATION
# ----------------------------------------------------

load_dotenv()

try:
    api_key = st.secrets.get("GEMINI_API_KEY", "")
except Exception:
    api_key = ""

api_key = api_key or os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("🔑 **Gemini API Key Missing**: Please set `GEMINI_API_KEY` in your `.env` file or Streamlit secrets to activate MechMate AI.")
    st.info("Tip: Copy `.env.example` to `.env` and paste your Google Gemini API key.")
    st.stop()


# ----------------------------------------------------
# 3. AI & RAG COMPONENTS INITIALIZATION
# ----------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return load_embedding_model()


@st.cache_resource(show_spinner=False)
def get_database():
    col = load_collection()
    # Strict isolation: ensure test fixture is never present in knowledge base
    try:
        delete_document(col, "fluid_mechanics_fixed.pdf")
    except Exception:
        pass
    return col


embedding_model = get_embedding_model()
collection = get_database()


# ----------------------------------------------------
# 4. SIDEBAR CONFIGURATION & CONTROLS
# ----------------------------------------------------

with st.sidebar:
    st.markdown("### ⚙️ **MechMate Workspace**")
    st.caption("Custom study materials & subject focus")

    SUBJECT_OPTIONS = [
        ("🌐 All subjects", "All subjects"),
        ("🌊 Fluid Mechanics", "Fluid Mechanics"),
        ("🔥 Thermodynamics", "Engineering Thermodynamics"),
        ("🧱 Strength of Materials", "Strength of Materials"),
        ("⚙️ Theory of Machines", "Theory of Machines"),
        ("📐 Machine Design", "Machine Design"),
        ("🏭 Manufacturing Processes", "Manufacturing Processes"),
        ("🌡️ Heat Transfer", "Heat Transfer"),
        ("⚖️ Engineering Mechanics", "Engineering Mechanics"),
    ]

    subject_labels = [label for label, _ in SUBJECT_OPTIONS]
    subject_values = {label: val for label, val in SUBJECT_OPTIONS}

    selected_label = st.selectbox(
        "Current Subject Context:",
        subject_labels,
        index=0,
        help="Select a subject to filter document search, or choose 'All subjects' to search your full library."
    )
    selected_subject = subject_values[selected_label]

    st.markdown("---")

    # Document Upload Section
    st.markdown("#### 📤 **Upload Study Materials**")
    uploaded_files = st.file_uploader(
        "Drag & drop PDF lecture notes or textbook chapters:",
        type=["pdf"],
        accept_multiple_files=True,
        help="Supported: Digital lecture PDFs & scanned notes (Gemini Vision OCR enabled)."
    )

    if st.button("📥 Add Notes to Knowledge Base", use_container_width=True, type="primary"):
        if not uploaded_files:
            st.warning("Please select at least one PDF file first.")
        else:
            upload_target_subject = (
                "General" if selected_subject == "All subjects" else selected_subject
            )

            progress_bar = st.progress(0)
            status_placeholder = st.empty()

            total_added = 0
            for idx, uploaded_file in enumerate(uploaded_files):
                status_placeholder.text(f"Indexing {uploaded_file.name}...")
                try:
                    chunks, message = process_pdf(
                        uploaded_file.getvalue(),
                        uploaded_file.name,
                        upload_target_subject,
                        collection,
                        embedding_model,
                        api_key=api_key
                    )
                    total_added += chunks
                except Exception as err:
                    st.error(f"Error indexing {uploaded_file.name}: {err}")
                progress_bar.progress((idx + 1) / len(uploaded_files))

            status_placeholder.empty()
            progress_bar.empty()

            if total_added > 0:
                st.success(f"Successfully added {total_added} knowledge chunks!")
                st.rerun()

    st.markdown("---")

    # Knowledge Base Management Expander
    with st.expander("📁 **Knowledge Base Library**", expanded=False):
        all_docs = get_indexed_documents(collection)
        user_docs = [d for d in all_docs if d["source"] != "fluid_mechanics_fixed.pdf"]

        if user_docs:
            st.markdown(f"**Indexed Documents ({len(user_docs)}):**")
            for doc in user_docs:
                col1, col2 = st.columns([4, 1])
                col1.markdown(f"📄 **{doc['source']}**  \n<span style='font-size:0.75rem; color:#888;'>{doc['subject']} • {doc['chunks']} chunks</span>", unsafe_allow_html=True)
                if col2.button("🗑️", key=f"del_{doc['source']}", help=f"Delete {doc['source']}"):
                    delete_document(collection, doc["source"])
                    st.toast(f"Removed {doc['source']}")
                    st.rerun()

            st.markdown("---")
            if st.button("🧹 Clear All Notes", use_container_width=True):
                clear_database(collection)
                st.toast("Knowledge Base reset successfully.")
                st.rerun()
        else:
            st.caption("No notes uploaded yet. Upload your PDF materials above to activate RAG search.")

    # Status Metric
    chunk_count = collection.count()
    st.metric(
        label="Indexed Knowledge Chunks",
        value=f"{chunk_count:,}",
        help="Total vector embeddings stored in local ChromaDB."
    )

    if st.button("🔄 Clear Chat Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ----------------------------------------------------
# 5. HEADER & HERO BRANDING
# ----------------------------------------------------

active_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

st.markdown(f"""
<div class="mechmate-header">
    <div>
        <h1 class="mechmate-title">⚙️ MechMate AI</h1>
        <p class="mechmate-subtitle">Intelligent Study & Numerical Companion for Mechanical Engineering</p>
    </div>
    <div style="text-align: right;">
        <span class="status-pill">
            <span style="font-size: 0.6rem;">🟢</span> {active_model}
        </span>
    </div>
</div>
""", unsafe_allow_html=True)


# ----------------------------------------------------
# 6. STARTER PROMPT HANDLER & EMPTY STATE
# ----------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# If no messages yet, display clean Welcome Hero with quick action cards
if len(st.session_state.messages) == 0:
    st.markdown("""
    <div class="hero-card">
        <h3 style="margin-top: 0; font-size: 1.15rem;">👋 Welcome, Engineer! How can I assist your study today?</h3>
        <p style="color: #888; font-size: 0.9rem; margin-bottom: 1rem;">
            MechMate AI provides step-by-step derivations, numerical problem verification, and exam preparation using your uploaded notes.
        </p>
        <div>
            <span class="capability-tag">📐 Step-by-Step Derivations</span>
            <span class="capability-tag">🧮 Numerical Verification</span>
            <span class="capability-tag">📚 RAG Notes Search</span>
            <span class="capability-tag">🧪 Viva & Lab Prep</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### 💡 **Quick Exploration Starters**")
    col1, col2 = st.columns(2)

    starter_prompts = [
        ("🌊 Derive Bernoulli's Equation", "Derive Bernoulli's Equation along a streamline. Clearly state all assumptions, step-by-step derivation, and final formula with SI units."),
        ("🔥 Rankine Cycle 4 Processes", "Explain the 4 key processes of an ideal Rankine Cycle with T-s coordinates and thermal efficiency formula."),
        ("🧱 Bending Stress Equation", "State and derive the Bending Equation for a pure beam: M/I = sigma/y = E/R with its underlying assumptions."),
        ("🧪 Cavitation in Pumps (Viva Prep)", "Viva Exam Prep: What is cavitation in centrifugal pumps, why does it occur, and what precautions prevent it?")
    ]

    with col1:
        if st.button(starter_prompts[0][0], use_container_width=True):
            st.session_state.pending_question = starter_prompts[0][1]
            st.rerun()
        if st.button(starter_prompts[1][0], use_container_width=True):
            st.session_state.pending_question = starter_prompts[1][1]
            st.rerun()

    with col2:
        if st.button(starter_prompts[2][0], use_container_width=True):
            st.session_state.pending_question = starter_prompts[2][1]
            st.rerun()
        if st.button(starter_prompts[3][0], use_container_width=True):
            st.session_state.pending_question = starter_prompts[3][1]
            st.rerun()


# ----------------------------------------------------
# 7. RENDER CHAT HISTORY
# ----------------------------------------------------

for msg in st.session_state.messages:
    avatar = "🧑‍🎓" if msg["role"] == "user" else "⚙️"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])


# ----------------------------------------------------
# 8. PROCESS USER QUESTION & STREAMING
# ----------------------------------------------------

# Check if a starter prompt was clicked
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None
else:
    question = st.chat_input("Ask any Mechanical Engineering concept, numerical problem, or question...")

if question:
    # 1. Append & render user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(question)

    # 2. Extract recent conversation context (last 6 turns)
    previous_messages = st.session_state.messages[:-1][-6:]
    history = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in previous_messages
    )

    # 3. Retrieve relevant study materials & Stream response
    with st.chat_message("assistant", avatar="⚙️"):
        retrieved_documents = []
        context = ""

        # Fast vector search
        with st.spinner("🔍 MechMate is consulting your study notes..."):
            try:
                retrieved_documents = retrieve_context(
                    question=question,
                    collection=collection,
                    embedding_model=embedding_model,
                    subject=selected_subject,
                    top_k=4
                )
                context = build_context(retrieved_documents)
            except Exception as rag_err:
                st.warning(f"Note: Vector retrieval skipped ({rag_err}). Relying on engineering fundamentals.")
                context = ""

        # Stream LLM Response
        try:
            stream_generator = ask_gemini_stream(
                question=question,
                context=context,
                history=history,
                system_prompt=SYSTEM_PROMPT,
                api_key=api_key
            )
            answer = st.write_stream(stream_generator)

            # Styled Source Citation Cards
            if retrieved_documents:
                with st.expander(f"📚 **Verified Study Sources ({len(retrieved_documents)} citations)**", expanded=False):
                    for item in retrieved_documents:
                        st.markdown(f"""
                        <div class="source-box">
                            <span class="source-badge badge-file">📄 {item['source']}</span>
                            <span class="source-badge badge-subject">🏷️ {item['subject']}</span>
                            <span class="source-badge badge-page">📖 Page {item['page']}</span>
                            <div style="margin-top: 6px; font-size: 0.85rem; color: #aaa;">
                                <em>"{item['text'][:220]}..."</em>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.caption("ℹ️ *Answer generated from Mechanical Engineering knowledge base (no specific uploaded PDF matched).*")

        except Exception as error:
            answer = (
                "⚠️ Something went wrong while connecting to the Gemini API. "
                "Please check your API key, model permissions, and internet connection."
            )
            st.error(f"{answer}\n\n**Details:** `{error}`")

    # 4. Save response to session state
    st.session_state.messages.append({"role": "assistant", "content": answer})
