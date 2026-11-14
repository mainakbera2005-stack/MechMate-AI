
import hashlib
from io import BytesIO
from pathlib import Path

import chromadb
from pypdf import PdfReader
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


# Divide text into smaller chunks
def split_text(text, chunk_size=900, overlap=150):

    chunks = []

    step = chunk_size - overlap

    for start in range(0, len(text), step):

        chunk = text[start:start + chunk_size].strip()

        if chunk:
            chunks.append(chunk)

    return chunks


# Extract PDF text and save it in ChromaDB
def process_pdf(
    pdf_bytes,
    filename,
    subject,
    collection,
    embedding_model
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
            "No readable text found. This may be a scanned PDF."
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
    question,
    collection,
    embedding_model,
    subject="All subjects",
    top_k=4
):

    if collection.count() == 0:
        return []

    # Convert question into an embedding
    question_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    # Apply subject filter when needed
    query_filter = None

    if subject != "All subjects":
        query_filter = {
            "subject": subject
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

        retrieved.append({
            "text": document,
            "source": metadata["source"],
            "subject": metadata["subject"],
            "page": metadata["page"],
            "distance": distance
        })

    return retrieved


# Format retrieved material for Gemini
def build_context(retrieved_documents):

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

