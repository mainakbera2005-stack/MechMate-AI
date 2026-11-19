import os
from pathlib import Path
from dotenv import load_dotenv

from rag import (
    load_embedding_model,
    load_collection,
    process_pdf,
    retrieve_context,
    split_text
)

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

print("--- 1. Testing Equation-Aware Smart Chunking ---")
sample_text = (
    "In fluid dynamics, Bernoulli's principle states that an increase in the speed of a fluid "
    "occurs simultaneously with a decrease in static pressure or a decrease in the fluid's potential energy.\n\n"
    "The characteristic equation along a streamline is given by:\n"
    "$$ P + \\frac{1}{2} \\rho v^2 + \\rho g z = \\text{constant} $$\n\n"
    "Where:\n"
    "- $P$ is the static pressure (Pa)\n"
    "- $\\rho$ is the fluid density (kg/m^3)\n"
    "- $v$ is the flow velocity (m/s)\n"
    "- $g$ is the acceleration due to gravity (m/s^2)\n"
    "- $z$ is the elevation above reference plane (m)"
)

chunks = split_text(sample_text, chunk_size=300, overlap=50)
print(f"Generated {len(chunks)} smart chunk(s).")
for idx, ch in enumerate(chunks, 1):
    print(f"Chunk {idx}:\n{ch}\n---")

print("\n--- 2. Loading RAG Components ---")
model = load_embedding_model()
collection = load_collection()
print("RAG engine loaded successfully!")

pdf_path = Path("data/fluid_mechanics_fixed.pdf")
if pdf_path.exists():
    with open(pdf_path, "rb") as file:
        pdf_bytes = file.read()

    chunks_count, message = process_pdf(
        pdf_bytes,
        "fluid_mechanics_fixed.pdf",
        "Fluid Mechanics",
        collection,
        model,
        api_key=api_key
    )
    print(f"Index status: {message} ({chunks_count} chunks)")

    print("\n--- 3. Testing Context Retrieval ---")
    question = "Explain Bernoulli's theorem"
    results = retrieve_context(
        question,
        collection,
        model,
        subject="Fluid Mechanics"
    )

    print(f"Retrieved {len(results)} context chunks:")
    for item in results:
        print(f"Source: {item['source']} (Page {item['page']})")
        print(f"Subject: {item['subject']} | Distance: {item.get('distance', 0):.4f}")
        print(f"Content:\n{item['text'][:200]}...")
        print("------------------------")
