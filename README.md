# Ancient Egypt RAG

A Retrieval-Augmented Generation (RAG) chatbot powered by the **Encyclopedia of Ancient Egypt**. Ask questions about Ancient Egyptian civilization and get accurate, source-backed answers with inline citations.

## Features

- **Hybrid Search** -- combines BM25 keyword retrieval with vector similarity search (60/40 weighting) via EnsembleRetriever
- **Query Decomposition** -- automatically detects multi-hop questions and breaks them into focused sub-queries
- **Re-Ranking** -- retrieved documents are re-ranked using a CrossEncoder (`BAAI/bge-reranker-v2-m3`) for higher relevance
- **Inline Citations** -- answers reference numbered sources (`[1]`, `[2]`, ...) so every claim is traceable
- **Streaming** -- real-time token-by-token responses via Server-Sent Events
- **LangSmith Tracing** -- every pipeline stage is traced for observability and debugging
- **LLM-as-Judge Evaluation** -- automated correctness evaluation using LangSmith and OpenEvals

## Architecture

```
User Question
    |
    v
Query Decomposition (simple / multi-hop detection)
    |
    v
Hybrid Retrieval (BM25 + Vector via Ollama embeddings)
    |
    v
Re-Ranking (CrossEncoder)
    |
    v
Citation Formatting ([1], [2], ...)
    |
    v
LLM Answer Generation (with source-grounded system prompt)
```

## Project Structure

```
├── app/
│   ├── core/
│   │   ├── config.py           # Centralized settings (Pydantic BaseSettings)
│   │   ├── schemas.py          # Request/response Pydantic models
│   │   └── pipeline.py         # RAG pipeline (decompose → retrieve → rerank → cite → generate)
│   │
│   └── api/
│       ├── main.py             # App factory, CORS, lifespan
│       ├── dependencies.py     # Dependency injection
│       └── routes/
│           └── chat.py         # POST /api/chat  |  POST /api/chat/stream  |  GET /api/health
│
├── frontend/
│   └── app.py                  # Streamlit chat UI (Amoon Chatbot)
│
├── evaluation/
│   ├── create_dataset.py       # Upload eval QA pairs to LangSmith
│   └── run_experiment.py       # Run correctness experiments (LLM-as-judge)
│
├── scripts/
│   └── create_vectordb.py      # PDF → Markdown (LlamaParse) → Chunks → Chroma
│
├── data/
│   ├── ancient_egypt.pdf       # Source PDF
│   ├── ancient_egypt.md        # Parsed markdown
│   └── eval_dataset.json       # Evaluation QA pairs
│
├── vectorstore/                # Chroma persistent database
├── .env.example                # Environment variable template
└── requirements.txt            # Python dependencies
```

## Prerequisites

- **Python 3.11+**
- **Ollama** running locally with an embedding model pulled (default: `qwen3-embedding:8b`)
- An **OpenAI-compatible LLM endpoint** (GitHub Models, OpenRouter, etc.)
- **LlamaParse** API key (for PDF parsing, only needed when rebuilding the vector store)

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Variables

Copy the example and fill in your keys:

```bash
cp .env.example .env
```

```env
# LLM (any OpenAI-compatible endpoint)
API_KEY=your-api-key
ENDPOINT=https://models.inference.ai.azure.com
MODEL_ID=openai/gpt-4o

# PDF Parsing
LLAMAPARSE_API_KEY=your-llamaparse-key

# Embeddings (Ollama)
OLLAMA_ENDPOINT=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=qwen3-embedding:8b

# Re-Ranking
RERANKER_MODEL_NAME=BAAI/bge-reranker-v2-m3

# LangSmith (observability)
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your-langsmith-key
LANGSMITH_PROJECT=your-project-name
```

### 3. Pull the Ollama Embedding Model

```bash
ollama pull qwen3-embedding:8b
```

### 4. Create Vector Database (if needed)

```bash
python -m scripts.create_vectordb
```

This parses the PDF via LlamaParse (agentic tier), splits the markdown by headers and character size, and persists the chunks to a Chroma database.

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

| Method | Endpoint            | Description                                       |
|--------|---------------------|---------------------------------------------------|
| GET    | `/api/health`       | Health check                                      |
| POST   | `/api/chat`         | Send a question, get a JSON response with sources |
| POST   | `/api/chat/stream`  | Stream answer tokens via Server-Sent Events       |

### Example Request

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the akh in ancient Egyptian belief?"}'
```

### Streaming Response Format

The `/api/chat/stream` endpoint emits SSE events with these types:

- `decomposition` -- sub-queries (only for multi-hop questions)
- `chunk` -- individual answer tokens
- `sources` -- cited source documents
- `done` -- stream complete

## Evaluation

```bash
# Upload dataset to LangSmith
python -m evaluation.create_dataset

# Run correctness experiment (LLM-as-judge)
python -m evaluation.run_experiment
```
