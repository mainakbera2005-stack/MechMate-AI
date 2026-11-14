
from rag import (
    load_embedding_model,
    load_collection,
    process_pdf,
    retrieve_context
)

# Load the embedding model
print("Loading embedding model...")

model = load_embedding_model()

# Connect to database
collection = load_collection()

print("RAG engine loaded successfully!")

# Enter your PDF path
pdf_path = "data/fluid_mechanics_fixed.pdf"

with open(pdf_path, "rb") as file:
    pdf_bytes = file.read()

# Process PDF
chunks, message = process_pdf(
    pdf_bytes,
    "fluid_mechanics_fixed.pdf",
    "Fluid Mechanics",
    collection,
    model
)

print(message)
print("Chunks created:", chunks)

# Test retrieval
question = "Explain Bernoulli's theorem"

results = retrieve_context(
    question,
    collection,
    model,
    subject="Fluid Mechanics"
)

print("\nRETRIEVED INFORMATION:\n")

for item in results:
    print("Source:", item["source"])
    print("Page:", item["page"])
    print("Content:", item["text"][:300])
    print("------------------------")

