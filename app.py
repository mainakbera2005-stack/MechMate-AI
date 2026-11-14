
import os
import hmac
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from brain import ask_gemini
from rag import (
    build_context,
    load_collection,
    load_embedding_model,
    process_pdf,
    retrieve_context,
)
from prompts import SYSTEM_PROMPT


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="MechMate AI | Engineering Workspace",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 2. CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")


def get_config_value(name, default=""):
    """Read configuration from Streamlit Secrets or environment."""
    try:
        value = st.secrets.get(name, default)
    except Exception:
        value = default

    return value or os.getenv(name, default)


api_key = get_config_value("GEMINI_API_KEY")
access_password = get_config_value("APP_ACCESS_PASSWORD")


# =========================================================
# 3. OPTIONAL ACCESS PROTECTION
# =========================================================

if access_password:

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:

        st.title("⚙️ MechMate AI")
        st.subheader("Engineering Workspace Access")

        with st.form("access_form"):

            entered_password = st.text_input(
                "Enter access password",
                type="password",
            )

            submitted = st.form_submit_button(
                "Enter Workspace",
                use_container_width=True,
            )

            if submitted:

                if hmac.compare_digest(
                    entered_password,
                    str(access_password),
                ):
                    st.session_state.authenticated = True
                    st.rerun()

                else:
                    st.error("Incorrect password.")

        st.stop()


# =========================================================
# 4. API KEY VALIDATION
# =========================================================

if not api_key:

    st.error(
        "Gemini API key not found. Add GEMINI_API_KEY "
        "to your local .env file or Streamlit Secrets."
    )

    st.stop()


# =========================================================
# 5. SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []


# =========================================================
# 6. LOAD AI COMPONENTS
# =========================================================

@st.cache_resource(show_spinner="Loading MechMate AI...")
def get_embedding_model():

    return load_embedding_model()


@st.cache_resource
def get_database():

    return load_collection()


try:

    embedding_model = get_embedding_model()
    collection = get_database()

except Exception as error:

    st.error(
        "MechMate could not initialize its knowledge system. "
        "Please check dependencies and deployment logs."
    )

    print(
        f"Initialization error: {type(error).__name__}: {error}"
    )

    st.stop()


# =========================================================
# 7. CUSTOM INTERFACE STYLING
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 0;
    }

    .main-subtitle {
        font-size: 15px;
        opacity: 0.75;
        margin-top: 5px;
        margin-bottom: 25px;
    }

    .welcome-card {
        padding: 25px;
        border-radius: 15px;
        background: linear-gradient(
            120deg,
            #142b45,
            #193c5c
        );
        border: 1px solid #294b6b;
        margin-bottom: 20px;
    }

    .welcome-card h2 {
        font-size: 25px;
        margin-bottom: 10px;
        color: white;
    }

    .welcome-card p {
        color: #d7e7f8;
        font-size: 15px;
        line-height: 1.7;
    }

    .section-heading {
        font-size: 23px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 8. SIDEBAR
# =========================================================

st.sidebar.markdown("## ⚙️ MechMate AI")

st.sidebar.caption("ENGINEERING WORKSPACE")

st.sidebar.divider()

st.sidebar.markdown("### 📚 Learning Environment")

subjects = [
    "All subjects",
    "Engineering Mechanics",
    "Engineering Thermodynamics",
    "Fluid Mechanics",
    "Strength of Materials",
    "Manufacturing Processes",
    "Theory of Machines",
    "Heat Transfer",
    "Machine Design",
]

selected_subject = st.sidebar.selectbox(
    "Select Engineering Subject",
    subjects,
)

st.sidebar.divider()


# =========================================================
# 9. PDF KNOWLEDGE MANAGEMENT
# =========================================================

st.sidebar.markdown("### 📂 Knowledge Management")

st.sidebar.write("Upload your study materials")

MAX_PDF_SIZE_MB = 20
MAX_PDF_SIZE_BYTES = MAX_PDF_SIZE_MB * 1024 * 1024

uploaded_files = st.sidebar.file_uploader(
    "Upload Mechanical Engineering PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    help="Maximum 20 MB per PDF. Scanned PDFs require OCR.",
)

if st.sidebar.button(
    "➕ Add to Knowledge Base",
    use_container_width=True,
    type="primary",
):

    if not uploaded_files:

        st.sidebar.warning("Please upload at least one PDF.")

    else:

        subject_to_store = (
            "General"
            if selected_subject == "All subjects"
            else selected_subject
        )

        total_files = len(uploaded_files)

        progress = st.sidebar.progress(0)

        for index, uploaded_file in enumerate(
            uploaded_files,
            start=1,
        ):

            if uploaded_file.size > MAX_PDF_SIZE_BYTES:

                st.sidebar.error(
                    f"{uploaded_file.name}: "
                    f"exceeds {MAX_PDF_SIZE_MB} MB."
                )

                progress.progress(index / total_files)
                continue

            try:

                with st.spinner(
                    f"Processing {uploaded_file.name}..."
                ):

                    chunks, message = process_pdf(
                        pdf_bytes=uploaded_file.getvalue(),
                        filename=uploaded_file.name,
                        subject=subject_to_store,
                        collection=collection,
                        embedding_model=embedding_model,
                    )

                if chunks > 0:

                    st.sidebar.success(
                        f"{uploaded_file.name}: "
                        f"{chunks} chunks indexed."
                    )

                    if uploaded_file.name not in (
                        st.session_state.indexed_files
                    ):

                        st.session_state.indexed_files.append(
                            uploaded_file.name
                        )

                else:

                    st.sidebar.info(
                        f"{uploaded_file.name}: {message}"
                    )

            except Exception as error:

                st.sidebar.error(
                    f"Could not process {uploaded_file.name}: "
                    f"{type(error).__name__}"
                )

                print(
                    f"PDF processing error: "
                    f"{type(error).__name__}: {error}"
                )

            progress.progress(index / total_files)

        st.sidebar.success("PDF processing completed.")


st.sidebar.caption(
    f"Maximum PDF size: {MAX_PDF_SIZE_MB} MB"
)

st.sidebar.divider()


# =========================================================
# 10. SYSTEM STATUS
# =========================================================

try:

    chunk_count = collection.count()

except Exception:

    chunk_count = 0


st.sidebar.markdown("### 🖥️ System Status")

st.sidebar.metric(
    "Indexed Knowledge Chunks",
    chunk_count,
)

st.sidebar.success("Gemini API key detected.")

st.sidebar.caption(
    "API access is checked when you send a question."
)

if access_password:

    if st.sidebar.button(
        "Log Out",
        use_container_width=True,
    ):

        st.session_state.authenticated = False
        st.session_state.messages = []
        st.rerun()

else:

    st.sidebar.warning(
        "No access password configured. "
        "Anyone with the public URL can access this workspace."
    )

st.sidebar.divider()


# =========================================================
# 11. CHAT MANAGEMENT
# =========================================================

if st.sidebar.button(
    "🗑️ Clear Conversation",
    use_container_width=True,
):

    st.session_state.messages = []
    st.rerun()


st.sidebar.caption("MechMate AI | Prototype v1.1")


# =========================================================
# 12. MAIN DASHBOARD
# =========================================================

st.markdown(
    '<p class="main-title">⚙️ MechMate AI</p>',
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="main-subtitle">'
    'Your Intelligent Companion for Mechanical Engineering'
    '</p>',
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="welcome-card">
        <h2>Welcome to your Engineering Workspace</h2>
        <p>
            Learn complex engineering concepts, explore your
            study materials, solve numerical problems and
            prepare for examinations with your AI-powered
            academic companion.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 13. DASHBOARD METRICS
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Knowledge Chunks",
        chunk_count,
    )

with col2:

    st.metric(
        "Conversation Messages",
        len(st.session_state.messages),
    )

with col3:

    st.metric(
        "Selected Subject",
        selected_subject,
    )


st.divider()


# =========================================================
# 14. EXPLORE ENGINEERING CONCEPTS
# =========================================================

st.markdown(
    '<p class="section-heading">'
    'Explore Engineering Concepts'
    '</p>',
    unsafe_allow_html=True,
)


concepts = {
    "Engineering Mechanics": "Explain Newton's laws of motion.",
    "Thermodynamics": "Explain the first law of thermodynamics.",
    "Fluid Mechanics": "What is Bernoulli's theorem?",
    "Strength of Materials": "Explain stress and strain.",
    "Heat Transfer": "What is the difference between conduction, convection and radiation?",
    "Theory of Machines": "Explain the working principle of a four-bar mechanism.",
    "Manufacturing Processes": "Explain casting and its applications.",
    "Machine Design": "What is the factor of safety?",
}

concept_columns = st.columns(2)

for index, (subject, question_text) in enumerate(
    concepts.items()
):

    with concept_columns[index % 2]:

        if st.button(
            f"📘 {subject}",
            use_container_width=True,
        ):

            st.session_state.pending_question = question_text
            st.rerun()


st.divider()


# =========================================================
# 15. DISPLAY CHAT HISTORY
# =========================================================

st.markdown(
    '<p class="section-heading">Engineering AI Assistant</p>',
    unsafe_allow_html=True,
)


for message in st.session_state.messages:

    with st.chat_message(
        message["role"],
        avatar="👨‍🎓" if message["role"] == "user" else "⚙️",
    ):

        st.markdown(message["content"])

        sources = message.get("sources", [])

        if sources:

            with st.expander("📚 Inspect Retrieved Sources"):

                for source_index, source in enumerate(
                    sources,
                    start=1,
                ):

                    st.markdown(
                        f"**Source {source_index}: "
                        f"{source['source']}**"
                    )

                    st.caption(
                        f"Subject: {source['subject']} | "
                        f"Page: {source['page']} | "
                        f"Cosine distance: "
                        f"{source['distance']:.4f}"
                    )

                    st.markdown(source["text"])
                    st.divider()


# =========================================================
# 16. QUESTION INPUT
# =========================================================

typed_question = st.chat_input(
    "Ask any Mechanical Engineering question..."
)

question = typed_question or st.session_state.pop(
    "pending_question",
    None,
)


# =========================================================
# 17. PROCESS STUDENT QUESTION
# =========================================================

if question:

    previous_messages = st.session_state.messages[-6:]

    history = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in previous_messages
    )

    # Save the user question.

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user", avatar="👨‍🎓"):

        st.markdown(question)


    # Generate AI response.

    answer = ""
    retrieved_documents = []

    with st.chat_message("assistant", avatar="⚙️"):

        try:

            with st.spinner(
                "MechMate is studying your notes..."
            ):

                retrieved_documents = retrieve_context(
                    question=question,
                    collection=collection,
                    embedding_model=embedding_model,
                    subject=selected_subject,
                    top_k=4,
                )

                context = build_context(
                    retrieved_documents
                )

                citation_instruction = """

                SOURCE CITATION REQUIREMENTS:

                1. Use uploaded study materials as your primary source.
                2. Cite retrieved sources using their exact identifiers : [S1], [S2], [S3], etc.
                3. Never invent a source identifier, filename or page number.
                4. Only cite sources that actually support the statement.
                5. Clearly distinguish uploaded PDF information from
                   general engineering knowledge.
                6. 6. If no relevant material is retrieved, disclose this.
                7. Do not claim that information came from a PDF
                   unless the retrieved context supports it.

                """

                enhanced_prompt = (
                    SYSTEM_PROMPT
                    + "\n\n"
                    + citation_instruction
                )

                answer = ask_gemini(
                    question=question,
                    context=context,
                    history=history,
                    system_prompt=enhanced_prompt,
                    api_key=api_key,
                )

            st.markdown(answer)

            if retrieved_documents:

                st.success(
                    f"{len(retrieved_documents)} "
                    "study material chunk(s) retrieved."
                )

                with st.expander(
                    "📚 Inspect Retrieved Sources"
                ):

                    for index, item in enumerate(
                        retrieved_documents,
                        start=1,
                    ):

                        st.markdown(
                            f"**Source {index}: {item['source']}**"
                        )

                        st.caption(
                            f"Subject: {item['subject']} | "
                            f"Page: {item['page']} | "
                            f"Cosine distance: "
                            f"{item['distance']:.4f}"
                        )

                        st.markdown(item["text"])
                        st.divider()

            else:

                st.info(
                    "No relevant study material was retrieved. "
                    "This answer may rely on general Gemini knowledge."
                )

        except Exception as error:

            answer = (
                "I could not generate a response right now. "
                "Please check your internet connection, Gemini "
                "configuration, model availability and API quota."
            )

            st.error(answer)

            print(
                f"MechMate request error: "
                f"{type(error).__name__}: {error}"
            )


    # Save assistant response and retrieved source metadata.

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": retrieved_documents,
        }
    )


# =========================================================
# 18. FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; padding:15px;">

        <p style="font-weight:600;">
            ⚙️ MechMate AI | Intelligent Engineering Learning
        </p>

        <p style="font-size:12px; opacity:0.7;">
            Built with Python, Streamlit, Gemini AI,
            Sentence Transformers and ChromaDB.
        </p>

        <p style="font-size:12px; opacity:0.7;">
            Learn. Understand. Engineer.
        </p>

    </div>
    """,
    unsafe_allow_html=True,
)
