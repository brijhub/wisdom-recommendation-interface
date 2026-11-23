# Wisdom Recommendation Interface

A small RAG application that reviews an internal conversation and returns practical, source-cited wisdom from the supplied SP01 SME material. FastAPI serves the API, Streamlit provides the UI, OpenAI supplies embeddings and the recommendation model, and ChromaDB stores knowledge-base chunks locally. Conversation transcripts are processed per request and are not saved.

## Project structure

```text
data/               supplied Excel knowledge files
src/app.py          Streamlit interface
src/backend.py      FastAPI endpoint
src/config.py       environment settings
src/ingest.py       extraction, ingestion, and ChromaDB search
src/rag.py          evidence-grounded recommendation prompt
validation/         manual transcripts, labelled retrieval cases, evaluator
requirements.txt
```

## Setup

The project uses the existing `venv` folder and Python 3.13.

```powershell
.\venv\Scripts\Activate.ps1
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Create `.env` in the project root and add:

```text
OPENAI_API_KEY=your-key-here
OPENAI_CHAT_MODEL=gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_EMBEDDING_DIMENSIONS=1536
CHROMA_COLLECTION=wisdom_knowledge
BACKEND_URL=http://127.0.0.1:8000
```

Ingest the knowledge base into local ChromaDB:

```powershell
.\venv\Scripts\python.exe src\ingest.py
```

This creates `storage/chroma_db/`. The folder is local, persistent, and excluded from Git; no external database service is required.

Start the backend:

```powershell
.\venv\Scripts\python.exe -m uvicorn backend:app --app-dir src --reload
```

In a second terminal, start the UI:

```powershell
.\venv\Scripts\python.exe -m streamlit run src\app.py
```

Open `http://localhost:8501`, paste a transcript, and select **Get wisdom**. API documentation is available at `http://127.0.0.1:8000/docs`.

## Ingestion scope

- **Pack feedback workbook:** only `Situation`, `Persona`, `Hidden state`, `Objection arcs`, and `Win conditions` are embedded.
- **Calibration workbooks:** only `Transcript 1`, `Transcript 2`, `Transcript 3`, `Line flags`, and `Part 2 single moments` are embedded.

Every other sheet, the context-feedback workbook, Word transcripts, and unsupported files are ignored.
Blank cells are ignored. Additional SME calibration workbooks are discovered automatically when
placed in `data/`; names following `... Calibration - SME Name (Blind Grading template).xlsx`
also preserve the SME name as metadata. Rerun `src/ingest.py` after any data change to replace the
collection with a fresh index.

### Metadata versus embedding vector

Each ChromaDB record has three useful parts:

- `content`: the meaningful natural-language record—for example a verdict/rewrite/reason row, a calibration grade with its coaching explanation, or several consecutive transcript turns. This exact text is sent to the OpenAI embedding model and becomes the vector used for semantic similarity.
- `metadata`: provenance and filters, not semantic content. It includes source filename, sheet, Excel row range, record type, scenario ID, transcript, topic, and reviewer when available.
- `embedding` (1536 numbers by default): the numerical representation returned by `text-embedding-3-small`. The incoming transcript is embedded with the same model, and ChromaDB cosine distance selects the closest knowledge records.

Keeping provenance as metadata makes every recommendation traceable without polluting the semantic vector with filenames and row numbers.

## API

`POST /recommend`

```json
{
  "transcript": "Customer: ...\nOfficer: ..."
}
```

The response contains Markdown coaching and the retrieved evidence records. There is deliberately no history endpoint or conversation database.

Recommendations are capped at 220 words and focus on the specific interaction: up to two things
that worked, the two highest-impact improvements with ready-to-use replacement language, and one
next action. Timestamps are used only when supplied in the transcript.

The external OpenAI calls used by ingestion and recommendations require the API key in `.env`.

## Retrieval validation

Paste-ready examples are in `validation/sample_transcripts.md`. The labelled retrieval set contains
14 additional interaction cases covering strong and weak examples of all eight SP01 skills.

Run retrieval evaluation after ingestion:

```powershell
.\venv\Scripts\python.exe validation\evaluate.py
```

The evaluator embeds all validation transcripts in one OpenAI request, searches ChromaDB, prints
Hit Rate@k, Precision@k, Recall@k, and MRR@k for k = 3, 5, and 6, and writes detailed per-case
retrievals to `validation/results.json`. This measures retrieval against labelled evidence; it does
not grade the final generated coaching text.
