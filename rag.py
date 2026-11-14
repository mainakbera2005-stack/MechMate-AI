import hashlib
import re
from io import BytesIO
from pathlib import Path

import chromadb
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# 1. DATABASE LOCATION
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "database" / "chroma_db"
DB_PATH.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 2. EMBEDDING MODEL
# ---------------------------------------------------------

def load_embedding_model():
    """Load the local sentence-transformer embedding model."""
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


# ---------------------------------------------------------
# 3. CHROMADB COLLECTION
# ---------------------------------------------------------

def load_collection():
    """Connect to the persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=str(DB_PATH))

    return client.get_or_create_collection(
        name="mechmate_knowledge",
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------
# 4. TEXT CHUNKING
# ---------------------------------------------------------

def split_text(text, chunk_size=900, overlap=150):
    """
    Split text into overlapping character chunks.

    Character-based chunking is retained from the original prototype
    to avoid changing the existing embedding/retrieval design abruptly.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size.")

    cleaned_text = re.sub(r"\s+", " ", text).strip()
    if not cleaned_text:
        return []

    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(cleaned_text), step):
        chunk = cleaned_text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)

    return chunks


# ---------------------------------------------------------
# 5. PDF PROCESSING AND INDEXING
# ---------------------------------------------------------

def process_pdf(
    pdf_bytes,
    filename,
    subject,
    collection,
    embedding_model,
):
    """Extract PDF text, create embeddings, and store them in ChromaDB."""

    if not pdf_bytes:
        raise ValueError("The uploaded PDF is empty.")

    if not pdf_bytes.startswith(b"%PDF"):
        raise ValueError("The uploaded file does not appear to be a valid PDF.")

    file_hash = hashlib.sha256(pdf_bytes).hexdigest()

    # Avoid indexing the same PDF more than once.
    existing = collection.get(
        where={"file_hash": file_hash},
        limit=1,
    )

    if existing.get("ids"):
        return 0, "This PDF is already indexed."

    try:
        reader = PdfReader(BytesIO(pdf_bytes), strict=False)
    except (PdfReadError, ValueError, OSError) as error:
        raise ValueError(
            "The PDF could not be read. It may be damaged or protected."
        ) from error

    if reader.is_encrypted:
        try:
            decrypted = reader.decrypt("")
        except Exception:
            decrypted = 0

        if not decrypted:
            raise ValueError(
                "This PDF is password-protected. Upload an unlocked PDF."
            )

    if len(reader.pages) == 0:
        raise ValueError("The PDF contains no pages.")

    all_chunks = []
    all_ids = []
    all_metadata = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""

        if not page_text.strip():
            continue

        chunks = split_text(page_text)

        for chunk_number, chunk in enumerate(chunks):
            unique_id = hashlib.sha256(
                f"{file_hash}_{page_number}_{chunk_number}".encode("utf-8")
            ).hexdigest()

            all_chunks.append(chunk)
            all_ids.append(unique_id)
            all_metadata.append(
                {
                    "source": Path(filename).name,
                    "subject": subject,
                    "page": page_number,
                    "file_hash": file_hash,
                }
            )

    if not all_chunks:
        raise ValueError(
            "No selectable text was found in this PDF. "
            "It may be scanned or image-based; OCR is not included yet."
        )

    embeddings = embedding_model.encode(
        all_chunks,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    collection.upsert(
        ids=all_ids,
        documents=all_chunks,
        embeddings=embeddings,
        metadatas=all_metadata,
    )

    return len(all_chunks), "PDF indexed successfully."


# ---------------------------------------------------------
# 6. SEMANTIC RETRIEVAL
# ---------------------------------------------------------

def retrieve_context(
    question,
    collection,
    embedding_model,
    subject="All subjects",
    top_k=4,
):
    """Retrieve the most relevant chunks for a student's question."""

    if not question or not question.strip():
        return []

    if top_k <= 0:
        return []

    if collection.count() == 0:
        return []

    question_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    query_args = {
        "query_embeddings": [question_embedding],
        "n_results": min(top_k, collection.count()),
        "include": ["documents", "metadatas", "distances"],
    }

    if subject and subject != "All subjects":
        query_args["where"] = {"subject": subject}

    results = collection.query(**query_args)

    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    retrieved = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):
        if not document or not metadata:
            continue

        retrieved.append(
            {
                "text": document,
                "source": metadata.get("source", "Unknown file"),
                "subject": metadata.get("subject", "Unspecified"),
                "page": metadata.get("page", "Unknown"),
                "distance": float(distance),
            }
        )

    return retrieved



# ---------------------------------------------------------
# 7. FORMAT CONTEXT WITH SOURCE IDENTIFIERS
# ---------------------------------------------------------

def build_context(retrieved_documents):
    """
    Format retrieved PDF chunks with unique source IDs.

    Example:
    [S1] -> Document A, Page 5
    [S2] -> Document B, Page 12
    """

    if not retrieved_documents:
        return (
            "No relevant study material was retrieved. "
            "Use general engineering knowledge and clearly "
            "disclose that no uploaded source supports the answer."
        )

    sections = []

    for index, item in enumerate(retrieved_documents, start=1):

        source_id = f"S{index}"

        sections.append(
            f"""
SOURCE ID: [{source_id}]

File: {item.get('source', 'Unknown file')}

Subject: {item.get('subject', 'Unspecified')}

Page: {item.get('page', 'Unknown')}

Content:
{item.get('text', '')}
""".strip()
        )

    return "\n\n--------------------------\n\n".join(sections)


# ---------------------------------------------------------
# 8. CITATION VALIDATION
# ---------------------------------------------------------

def audit_citations(answer, retrieved_documents):
    """
    Check whether Gemini used valid retrieved source IDs.

    Invalid citations are replaced with an explicit
    warning marker rather than being displayed as valid.
    """

    import re

    if not answer:
        answer = ""

    valid_ids = {
        f"S{index}"
        for index in range(1, len(retrieved_documents) + 1)
    }

    citation_pattern = r"\[(S\d+)\]"

    found_ids = {
        match.upper()
        for match in re.findall(
            citation_pattern,
            answer,
            flags=re.IGNORECASE
        )
    }

    unknown_ids = found_ids - valid_ids
    valid_citations = found_ids & valid_ids

    def replace_invalid(match):

        source_id = match.group(1).upper()

        if source_id not in valid_ids:
            return "[unverified citation]"

        return f"[{source_id}]"

    checked_answer = re.sub(
        citation_pattern,
        replace_invalid,
        answer,
        flags=re.IGNORECASE
    )

    report = {
        "valid_citations": sorted(valid_citations),
        "unknown_citations": sorted(unknown_ids),
        "missing_citations": (
            bool(retrieved_documents)
            and not bool(valid_citations)
        ),
    }

    return checked_answer, report
