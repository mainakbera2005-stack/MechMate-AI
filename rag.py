import hashlib
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from pypdf import PdfReader, PdfWriter
from sentence_transformers import SentenceTransformer


# Database location
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "database" / "chroma_db"

DB_PATH.mkdir(parents=True, exist_ok=True)


# Load the embedding model
def load_embedding_model():
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    return model


# Connect to ChromaDB
def load_collection():
    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    collection = client.get_or_create_collection(
        name="mechmate_knowledge",
        metadata={"hnsw:space": "cosine"}
    )

    return collection


# Divide text into structure-preserving chunks
def split_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    """
    Recursively splits text using structural separators (paragraphs, newlines,
    sentences, spaces) to avoid breaking mathematical equations, tables, or words.
    """
    if not text or not text.strip():
        return []

    separators = ["\n\n", "\n", ". ", "; ", " ", ""]

    def _split(s: str, seps: List[str]) -> List[str]:
        if not s.strip():
            return []
        if len(s) <= chunk_size or not seps:
            return [s.strip()] if s.strip() else []

        sep = seps[0]
        remaining_seps = seps[1:]

        if sep == "":
            chunks = []
            step = max(1, chunk_size - overlap)
            for i in range(0, len(s), step):
                sub = s[i:i + chunk_size].strip()
                if sub:
                    chunks.append(sub)
            return chunks

        parts = s.split(sep)
        splits = []
        current_chunk = ""

        for part in parts:
            candidate = f"{current_chunk}{sep}{part}" if current_chunk else part
            if len(candidate) <= chunk_size:
                current_chunk = candidate
            else:
                if current_chunk.strip():
                    splits.append(current_chunk.strip())
                if len(part) > chunk_size:
                    sub_splits = _split(part, remaining_seps)
                    splits.extend(sub_splits)
                    current_chunk = ""
                else:
                    current_chunk = part

        if current_chunk.strip():
            splits.append(current_chunk.strip())

        return splits

    raw_chunks = _split(text, separators)

    if overlap <= 0 or len(raw_chunks) <= 1:
        return raw_chunks

    final_chunks = []
    for i, chunk in enumerate(raw_chunks):
        if i > 0:
            prev_tail = raw_chunks[i - 1][-overlap:].strip()
            if prev_tail and not chunk.startswith(prev_tail):
                chunk = f"... {prev_tail}\n{chunk}"
        final_chunks.append(chunk)

    return final_chunks


def render_page_to_pdf_bytes(page) -> bytes:
    """Extract a single PDF page into in-memory PDF bytes."""
    writer = PdfWriter()
    writer.add_page(page)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def transcribe_page_with_gemini(
    page_bytes: bytes,
    api_key: str,
    model: str = "gemini-2.5-flash"
) -> str:
    """
    Uses Gemini Multimodal Document Understanding to transcribe scanned
    PDF pages into text and KaTeX-compatible LaTeX equations.
    """
    if not api_key:
        return ""
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        prompt = (
            "Transcribe all text, numbers, formulas, tables, and notes from this engineering document page verbatim. "
            "Convert all mathematical equations and derivations into standard LaTeX format ($...$ for inline, $$...$$ for block). "
            "Do not summarize or omit anything. Return only the extracted content."
        )
        response = client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=page_bytes, mime_type="application/pdf"),
                prompt
            ]
        )
        return response.text or ""
    except Exception as e:
        print(f"Gemini OCR warning: {e}")
        return ""


# Extract PDF text and save it in ChromaDB
def process_pdf(
    pdf_bytes: bytes,
    filename: str,
    subject: str,
    collection,
    embedding_model,
    api_key: Optional[str] = None,
    ocr_model: str = "gemini-2.5-flash"
):
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()

    # Avoid indexing the same PDF repeatedly
    existing = collection.get(
        where={"file_hash": file_hash},
        limit=1
    )

    if existing["ids"]:
        return 0, "This PDF is already indexed."

    reader = PdfReader(BytesIO(pdf_bytes))

    all_chunks = []
    all_ids = []
    all_metadata = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        # If no digital text found, attempt Gemini Vision OCR for scanned page
        if not text.strip() and api_key:
            page_pdf_bytes = render_page_to_pdf_bytes(page)
            text = transcribe_page_with_gemini(
                page_pdf_bytes,
                api_key=api_key,
                model=ocr_model
            )

        if not text.strip():
            continue

        chunks = split_text(text)

        for chunk_number, chunk in enumerate(chunks):
            unique_id = hashlib.sha256(
                f"{file_hash}_{page_number}_{chunk_number}".encode()
            ).hexdigest()

            all_chunks.append(chunk)
            all_ids.append(unique_id)

            all_metadata.append({
                "source": filename,
                "subject": subject,
                "page": page_number,
                "file_hash": file_hash
            })

    if not all_chunks:
        raise ValueError(
            "No readable text found. If this is a scanned PDF, verify that your "
            "Gemini API key is configured to enable Gemini Vision OCR."
        )

    # Convert chunks into embeddings
    embeddings = embedding_model.encode(
        all_chunks,
        normalize_embeddings=True
    ).tolist()

    # Store everything in ChromaDB
    collection.upsert(
        ids=all_ids,
        documents=all_chunks,
        embeddings=embeddings,
        metadatas=all_metadata
    )

    return len(all_chunks), "PDF indexed successfully!"


# Search for relevant information
def retrieve_context(
    question: str,
    collection,
    embedding_model,
    subject: str = "All subjects",
    top_k: int = 4,
    max_distance: Optional[float] = None
) -> List[Dict[str, Any]]:
    if collection.count() == 0:
        return []

    # Convert question into an embedding
    question_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    # Apply subject filter: if specific subject selected, include subject + General
    query_filter = None
    if subject != "All subjects":
        query_filter = {
            "$or": [
                {"subject": subject},
                {"subject": "General"}
            ]
        }

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(top_k, collection.count()),
        where=query_filter,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    retrieved = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for document, metadata, distance in zip(
        documents, metadatas, distances
    ):
        if max_distance is not None and distance > max_distance:
            continue

        retrieved.append({
            "text": document,
            "source": metadata["source"],
            "subject": metadata["subject"],
            "page": metadata["page"],
            "distance": distance
        })

    return retrieved


# Format retrieved material for Gemini
def build_context(retrieved_documents: List[Dict[str, Any]]) -> str:
    if not retrieved_documents:
        return "No relevant study material was retrieved."

    context = ""

    for index, item in enumerate(retrieved_documents, start=1):
        context += f"""
        SOURCE {index}
        File: {item['source']}
        Subject: {item['subject']}
        Page: {item['page']}

        Content:
        {item['text']}

        --------------------------
        """

    return context
