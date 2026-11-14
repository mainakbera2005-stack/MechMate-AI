# MechMate AI

**An AI-powered study companion for Mechanical Engineering students.**

## 1. Project Overview

MechMate AI is a Mechanical Engineering study assistant designed to help students understand engineering concepts, study lecture notes, and ask questions using uploaded PDF materials.

The application combines a language model with a Retrieval-Augmented Generation (RAG) system to provide answers based on uploaded study resources.

## 2. Technology Stack

- **Python** — Main programming language
- **Streamlit** — Web application interface
- **Google Gemini API** — AI answer generation
- **Sentence Transformers** — Text embeddings
- **ChromaDB** — Vector database for document retrieval
- **PyPDF** — PDF text extraction
- **Python-dotenv** — Environment variable management

## 3. Current Working Features

- Streamlit-based chat interface.
- Gemini-powered Mechanical Engineering answers.
- PDF study material ingestion.
- Text extraction from supported PDFs.
- Semantic search using Sentence Transformers.
- Vector storage and retrieval using ChromaDB.
- Answers using retrieved study material.
- Display of source filenames and available page numbers.
- Beginner-friendly explanations and mathematical derivations.

## 4. Project Setup

### Requirements

- Windows
- Python
- Internet connection
- Google Gemini API key

### Activate the virtual environment

Open PowerShell in the project folder and run:

```powershell
.\.venv\Scripts\Activate.ps1
```

Alternatively, use the virtual environment's Python executable directly.

### Run the application

```powershell
& ".\.venv\Scripts\python.exe" -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

```text
http://localhost:8501
```

The port may change if another Streamlit instance is already running.

## 5. Important Project Files

| File / Folder | Purpose |
|---|---|
| `app.py` | Main Streamlit application |
| `brain.py` | Gemini integration and answer generation |
| `rag.py` | Document processing and retrieval |
| `.env` | Environment variables and API key |
| `.venv/` | Python virtual environment |
| `database/chroma_db/` | Persistent ChromaDB storage |
| `data/` | Study material storage, if used by the application |

## 6. Current AI Configuration

The Gemini model configured in `brain.py` is:

`gemini-3.5-flash-lite`

The application uses a built-in engineering instruction prompt to encourage:
- Step-by-step explanations.
- Explicit assumptions.
- Correct equation usage.
- Unit checking.
- Clear identification of uncertainty.

These instructions improve answer quality but do not guarantee mathematical or physical correctness.

## 7. Known Limitations

1. Scanned PDFs may not be readable because OCR is not yet integrated.
2. Engineering equations and physical assumptions are not independently verified by a dedicated calculation engine.
3. Ambiguous problem statements may still require a diagram or clarification.
4. Retrieval quality and duplicate source handling need further testing.
5. Subject filtering and document organization should be tested with multiple uploaded PDFs.

## 8. Recommended Future Improvements

### High Priority
- Integrate OCR for scanned PDFs.
- Improve engineering equation verification.
- Improve physical assumption and boundary-condition checking.
- Test subject filtering and source retrieval.
- Improve API error handling.

### Additional Improvements
- Add a numerical calculation verification module.
- Add more Mechanical Engineering subjects.
- Improve the user interface.
- Add automated tests.
- Prepare the application for deployment.

## 9. Important Security Notes

- Never upload or share the `.env` file.
- Never expose the Gemini API key in source code or screenshots.
- Do not commit `.venv/`, database files, or private credentials to a public repository without checking what they contain.

## 10. Current Project Status

**Working prototype — further testing and development required.**

The application currently launches successfully, connects to Gemini, retrieves information from uploaded PDFs, and generates Mechanical Engineering explanations.

The next development phase should focus on reliability, OCR, technical answer verification, and testing.

---

**Project:** MechMate AI  
**Purpose:** Mechanical Engineering AI Study Companion
