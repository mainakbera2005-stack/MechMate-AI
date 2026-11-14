# MechMate AI

An AI-powered study companion for Mechanical Engineering students.

## Milestone 1: Prototype stabilization

This version preserves the original Streamlit + Gemini + Sentence Transformers
+ ChromaDB architecture and focuses on safer setup, PDF processing, session
state handling, and clearer error handling.

## Requirements

- Windows 10/11
- Python 3.11 or 3.12 recommended for this dependency stack
- Internet connection for Gemini API access and initial model download
- Gemini API key

## 1. Create and activate a virtual environment

Open a terminal in this folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

## 2. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The first installation can take time because Sentence Transformers,
PyTorch, and related packages are sizeable.

## 3. Configure the Gemini API key

Copy `.env.example` to `.env`:

```powershell
Copy-Item .env.example .env
```

Open `.env` and replace the placeholder with your own key:

```text
GEMINI_API_KEY=your_actual_key
```

Do not share `.env` or commit it to GitHub.

The default Gemini model name remains the one from the original project.
If that model is unavailable to your API account, set `GEMINI_MODEL` in
`.env` to a model name currently available to your account.

## 4. Run the app

```powershell
python -m streamlit run app.py
```

Open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## 5. First test

1. Select Fluid Mechanics.
2. Upload a text-based Fluid Mechanics PDF.
3. Click "Add PDFs to Knowledge Base".
4. Ask a question that is covered in that PDF.
5. Expand "View retrieved sources" and check the filename and page.
6. Ask an unrelated question and confirm that the app says no study material
   was retrieved when no matching chunk is returned.

## What this milestone improves

- Initializes chat session state before the Clear Chat button is used.
- Reads the API key from `.env` or Streamlit secrets.
- Removes unused imports.
- Adds PDF size, validity, encryption, and empty-text checks.
- Preserves duplicate-PDF detection.
- Handles empty Gemini responses.
- Displays retrieved text, source filename, subject, page, and cosine distance.
- Uses a clean UTF-8 requirements file containing direct dependencies.

## Known limitations

- OCR is not included. Scanned/image-only PDFs are rejected with a clear message.
- Retrieval still uses the original character-based chunking and top-4 semantic
  search. A relevance threshold has not yet been calibrated.
- Numerical answers are not independently verified by a calculation engine.
- The Gemini model name must be available to the API key being used.
- This is a local prototype, not a production-secured multi-user application.

## Project structure

```text
for_U_Mainak/
├── app.py
├── brain.py
├── prompts.py
├── rag.py
├── requirements.txt
├── .env.example
├── README.md
└── data/
    └── fluid_mechanics_fixed.pdf
```
