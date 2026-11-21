import pytest
from rag import split_text, EXCLUDED_SOURCES, get_indexed_documents, delete_document, clear_database


def test_split_text_preserves_equations():
    sample = (
        "Thermodynamics first law:\n"
        "$$ dQ = dU + dW $$\n\n"
        "Where $dQ$ is heat added and $dW$ is work done by the system."
    )
    chunks = split_text(sample, chunk_size=100, overlap=20)
    assert len(chunks) >= 1
    # Ensure equation block $$ dQ = dU + dW $$ was not broken in half
    assert any("$$ dQ = dU + dW $$" in chunk for chunk in chunks)


def test_fluid_mechanics_pdf_in_excluded_sources():
    # Enforce that fluid_mechanics_fixed.pdf is registered in EXCLUDED_SOURCES
    assert "fluid_mechanics_fixed.pdf" in EXCLUDED_SOURCES


def test_db_management_functions():
    import chromadb
    test_client = chromadb.EphemeralClient()
    col = test_client.create_collection("mgmt_test")

    col.add(
        ids=["id1", "id2"],
        documents=["text1", "text2"],
        metadatas=[
            {"source": "test_notes.pdf", "subject": "Thermodynamics"},
            {"source": "test_notes.pdf", "subject": "Thermodynamics"}
        ]
    )

    docs = get_indexed_documents(col)
    assert len(docs) == 1
    assert docs[0]["source"] == "test_notes.pdf"
    assert docs[0]["chunks"] == 2

    # Test delete
    deleted = delete_document(col, "test_notes.pdf")
    assert deleted == 2
    assert col.count() == 0

    # Test clear
    col.add(ids=["id3"], documents=["text3"], metadatas=[{"source": "fluid_mechanics_fixed.pdf"}])
    assert col.count() == 1
    cleared = clear_database(col)
    assert cleared == 1
    assert col.count() == 0
