import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from pypdf import PdfReader
from rag import render_page_to_pdf_bytes, transcribe_page_with_gemini

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
pdf_path = Path("data/fluid_mechanics_fixed.pdf")

if not api_key:
    print("Warning: GEMINI_API_KEY is not set. Please set it in .env to test Gemini Vision OCR.")
    sys.exit(1)

if not pdf_path.exists():
    print(f"Error: {pdf_path} does not exist.")
    sys.exit(1)

print(f"Testing Gemini Vision OCR on: {pdf_path}")
reader = PdfReader(str(pdf_path))
print(f"Total pages: {len(reader.pages)}")

# Test on page 1
page_1 = reader.pages[0]
page_bytes = render_page_to_pdf_bytes(page_1)
print("Sending page 1 to Gemini Multimodal Document Understanding...")

transcription = transcribe_page_with_gemini(page_bytes, api_key=api_key)
print("\n--- EXTRACTED OCR & FORMULA CONTENT ---")
print(transcription[:1500])
print("\nGemini Vision OCR test completed successfully!")
