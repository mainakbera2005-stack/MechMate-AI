
import os
from pathlib import Path

import streamlit as st

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from brain import ask_gemini, ask_gemini_stream
from rag import (
    load_embedding_model,
    load_collection,
    process_pdf,
    retrieve_context,
    build_context
)
from prompts import SYSTEM_PROMPT


# --------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="MechMate AI",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ MechMate AI")
st.caption(
    "Your Personal Mechanical Engineering Companion"
)


# --------------------------------
# 2. API KEY
# --------------------------------

load_dotenv()

try:
    api_key = st.secrets.get("GEMINI_API_KEY", "")
except Exception:
    api_key = ""

api_key = api_key or os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("Please configure your Gemini API key.")
    st.stop()


# --------------------------------
# 3. LOAD AI COMPONENTS
# --------------------------------

@st.cache_resource
def get_embedding_model():

    return load_embedding_model()


@st.cache_resource
def get_database():

    return load_collection()


embedding_model = get_embedding_model()
collection = get_database()


# --------------------------------
# 4. SIDEBAR
# --------------------------------

st.sidebar.title("⚙️ MechMate Settings")

subjects = [
    "All subjects",
    "Engineering Mechanics",
    "Engineering Thermodynamics",
    "Fluid Mechanics",
    "Strength of Materials",
    "Manufacturing Processes",
    "Theory of Machines",
    "Heat Transfer",
    "Machine Design"
]

selected_subject = st.sidebar.selectbox(
    "Choose your subject",
    subjects
)

st.sidebar.divider()

st.sidebar.subheader("Upload Study Materials")

uploaded_files = st.sidebar.file_uploader(
    "Upload your Mechanical Engineering PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if st.sidebar.button(
    "Add PDFs to Knowledge Base",
    use_container_width=True
):

    if not uploaded_files:

        st.sidebar.warning("Upload a PDF first.")

    else:

        subject_to_store = (
            "General"
            if selected_subject == "All subjects"
            else selected_subject
        )

        for uploaded_file in uploaded_files:

            try:

                with st.spinner(
                    f"Processing {uploaded_file.name}..."
                ):

                    chunks, message = process_pdf(
                        uploaded_file.getvalue(),
                        uploaded_file.name,
                        subject_to_store,
                        collection,
                        embedding_model
                    )

                if chunks > 0:
                    st.sidebar.success(
                        f"{uploaded_file.name}: {chunks} chunks added."
                    )
                else:
                    st.sidebar.info(message)

            except Exception as error:

                st.sidebar.error(
                    f"Error processing {uploaded_file.name}: {error}"
                )

st.sidebar.divider()

st.sidebar.metric(
    "Knowledge Base Chunks",
    collection.count()
)

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()


# --------------------------------
# 5. CHAT HISTORY
# --------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# --------------------------------
# 6. CHAT INPUT
# --------------------------------

question = st.chat_input(
    "Ask any Mechanical Engineering question..."
)


# --------------------------------
# 7. PROCESS QUESTION
# --------------------------------

if question:

    # Display student question

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):

        st.markdown(question)


    # Get previous conversation context

    previous_messages = st.session_state.messages[:-1][-6:]

    history = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in previous_messages
    )


    # Search relevant study materials

    with st.chat_message("assistant"):

        retrieved_documents = []
        context = ""

        with st.spinner("MechMate is studying your notes..."):
            try:
                retrieved_documents = retrieve_context(
                    question,
                    collection,
                    embedding_model,
                    selected_subject,
                    top_k=4
                )
                context = build_context(retrieved_documents)
            except Exception as rag_err:
                st.warning(f"Could not retrieve notes: {rag_err}")
                context = ""

        try:
            stream_generator = ask_gemini_stream(
                question=question,
                context=context,
                history=history,
                system_prompt=SYSTEM_PROMPT,
                api_key=api_key
            )

            answer = st.write_stream(stream_generator)

            # Show retrieved sources
            if retrieved_documents:
                with st.expander("📚 View retrieved sources"):
                    for item in retrieved_documents:
                        st.markdown(
                            f"""
                            **File:** {item['source']}  
                            **Subject:** {item['subject']}  
                            **Page:** {item['page']}  

                            ---
                            """
                        )
            else:
                st.info(
                    "No study material has been retrieved. "
                    "The answer may rely on general knowledge."
                )

        except Exception as error:
            answer = (
                "Something went wrong while generating the answer. "
                "Please check your Gemini API key, model access, and internet connection."
            )
            st.error(f"{answer}\n\nDetails: {error}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
