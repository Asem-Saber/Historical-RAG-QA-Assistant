# Ancient Egypt RAG 🏛️

A Retrieval-Augmented Generation (RAG) chatbot powered by the **Encyclopedia of Ancient Egypt**. Ask questions about Ancient Egyptian civilization and get accurate, source-backed answers.

## Project Structure

```
AncientEgyptRAG/
├── app/                    # Backend application
│   ├── core/               # RAG business logic
│   │   ├── config.py       # Centralized settings (Pydantic BaseSettings)
│   │   ├── schemas.py      # Request/response Pydantic models
│   │   └── pipeline.py     # RAG pipeline (retrieve → format → generate)
│   │
│   └── api/                # FastAPI backend
│       ├── main.py         # App factory, CORS, lifespan
│       ├── dependencies.py # Dependency injection
│       └── routes/
│           └── chat.py     # POST /api/chat  |  GET /api/health
│
├── frontend/               # Streamlit chat UI
│   └── app.py              # Calls FastAPI over HTTP
│
├── evaluation/             # LangSmith evaluation
│   ├── create_dataset.py   # Upload eval QA pairs to LangSmith
│   └── run_experiment.py   # Run correctness experiments
│
├── scripts/                # Utility scripts
│   └── create_vectordb.py  # PDF → Markdown → Chunks → Chroma
│
├── data/                   # Data assets
│   ├── ancient_egypt.pdf
│   ├── ancient_egypt.md
│   └── eval_dataset.json
│
├── vectorstore/            # Chroma persistent database
├── .env                    # Environment variables
└── requirements.txt        # Python dependencies
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file in the project root:

```env
GITHUB_API_KEY=your_github_models_api_key
HF_API_KEY=your_huggingface_token
LLAMAPARSE_API_KEY=your_llamaparse_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
```

### 3. Create Vector Database (if needed)

```bash
python -m scripts.create_vectordb
```

## Running

### Start the API Server

```bash
uvicorn app.api.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### Start the Frontend

```bash
streamlit run frontend/app.py
```

> **Note:** The FastAPI server must be running before starting the Streamlit frontend.

## API Endpoints

| Method | Endpoint       | Description                          |
|--------|----------------|--------------------------------------|
| GET    | `/api/health`  | Health check                         |
| POST   | `/api/chat`    | Send a question, get a RAG answer    |

### Example Request

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the a'\''akh in ancient Egyptian belief?"}'
```

## Evaluation

```bash
# Upload dataset to LangSmith
python -m evaluation.create_dataset

# Run evaluation experiment
python -m evaluation.run_experiment
```
