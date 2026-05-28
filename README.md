# RAG Project

Production-ready Retrieval-Augmented Generation pipeline.

## Architecture

```
rag_project/
├── src/
│   ├── ingestion/      Load from PDFs, CSVs, Confluence, SharePoint, URLs
│   ├── chunking/       Split text into overlapping chunks
│   ├── embeddings/     Generate embeddings via Anthropic / OpenAI
│   ├── vectordb/       ChromaDB persistent vector store
│   ├── retrieval/      Similarity search + reranking
│   ├── prompts/        Prompt templates
│   ├── llm/            Claude API client
│   ├── api/            FastAPI routes
│   └── utils/          Logging, helpers
├── tests/
├── logs/
├── .env
├── config.yaml
├── requirements.txt
└── main.py
```

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env          # add your API keys
uvicorn main:app --reload
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/ingest/file` | Upload & embed a file |
| POST | `/ingest/confluence` | Sync Confluence space |
| POST | `/ingest/sharepoint` | Sync SharePoint site |
| POST | `/chat` | RAG chat query |
| GET  | `/collections` | List ChromaDB collections |
| DELETE | `/collections/{name}` | Clear a collection |

## Tip

Keep `.env` out of git. Never hardcode API keys.