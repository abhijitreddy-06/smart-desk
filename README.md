# Smart Desk

Smart Desk is a Python-based Streamlit app that routes questions between a direct LLM response path and a policy-focused RAG path backed by ChromaDB.

## What it does

- Answers general questions with a direct Gemini chat response.
- Detects policy-related questions and sends them through a retrieval-augmented workflow.
- Ingests PDF files into a Chroma vector store for policy search.

## Project Structure

- `app.py` - Streamlit UI.
- `agent.py` - Routes queries between direct and policy modes.
- `retrieval_service.py` - Policy retrieval and RAG answer generation.
- `ingestion_service.py` - PDF ingestion pipeline that builds the Chroma database.
- `pdfs/` - Source PDFs used for ingestion.
- `chroma_db/` - Persisted vector database.

## Requirements

- Python 3.11 or newer, but below 3.14.
- Poetry.
- A Gemini API key.
- PDF documents to ingest for policy search.

## Setup

1. Install dependencies:

   ```powershell
   poetry install --no-root
   ```

2. Copy `.env.example` to `.env` and fill in the values you need.

3. Activate the Poetry environment if needed:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

## Environment Variables

The app reads configuration with `python-dotenv` and `os.getenv`.

- `GEMINI_API_KEY` - required for the LLM.
- `GEMINI_MODEL` - optional model name, for example `gemini-2.5-flash`.
- `EMBEDDING_MODEL` - embedding model for ingestion and retrieval.
- `UNSTRUCTURED_PARTITION_STRATEGY` - PDF parsing strategy.
- `PDF_FOLDER_PATH` - folder containing PDF files for ingestion.
- `CHROMA_DB_PATH` - folder where the Chroma database is stored.
- `RAG_TOP_K` - number of retrieved chunks to include.
- `DEFAULT_USER_QUERY` - optional placeholder text for the UI.
- `STREAMLIT_PORT` - optional Streamlit port.

## Run the App

Start the Streamlit UI:

```powershell
streamlit run app.py
```

Open the local URL shown in the terminal and ask either a general question or a policy question.

## Build the Vector Store

Run ingestion after placing PDF files in the configured folder:

```powershell
python ingestion_service.py "path\to\pdfs" "path\to\chroma_db"
```

Or rely on `.env` defaults:

```powershell
python ingestion_service.py
```

## How It Works

1. `app.py` sends user input to `handle_query()`.
2. `agent.py` classifies the query as policy or non-policy.
3. Non-policy queries go to a direct Gemini response.
4. Policy queries go to `answer_policy_query()` in `retrieval_service.py`.
5. Retrieved policy context is used to generate the final answer.

## Notes

- Keep secrets in `.env` and do not commit them.
- Re-run ingestion whenever the source PDFs change.
- If you add new generated folders, update `.gitignore` as needed.
