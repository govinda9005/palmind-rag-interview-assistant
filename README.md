# PalmMind AI – Document-Based Conversational RAG & Interview Booking System

An enterprise-grade, modular Python backend providing document ingestion, custom retrieval-augmented generation (RAG), multi-turn conversational memory, and LLM-driven structured interview booking.

---

## 🏗️ System Architecture

The application is structured into clean, decoupled layers adhering to **SOLID** principles and separation of concerns:

```
palmind-rag-interview-assistant/
├── app/
│   ├── main.py                     # FastAPI application entrypoint & lifespan
│   ├── core/
│   │   ├── config.py               # Pydantic v2 BaseSettings configuration
│   │   └── logging.py              # Structured, secure logging configuration
│   ├── api/
│   │   ├── routes/
│   │   │   ├── documents.py        # Document ingestion endpoints
│   │   │   └── chat.py             # Conversational RAG & Booking chat endpoint
│   │   └── schemas/
│   │       ├── document.py         # Ingestion request/response models
│   │       ├── chat.py             # Chat and retrieved source chunk models
│   │       └── booking.py          # Structured interview booking models
│   ├── services/
│   │   ├── extraction_service.py   # PDF and TXT text extraction
│   │   ├── chunking_service.py     # Fixed-size and Recursive chunking strategies
│   │   ├── embedding_service.py    # Local SentenceTransformer embeddings
│   │   ├── vector_service.py       # Qdrant collection, indexing, and search
│   │   ├── memory_service.py       # Redis multi-turn session memory
│   │   ├── rag_service.py          # Custom prompt construction and retrieval flow
│   │   ├── booking_service.py      # LLM structured extraction & booking validation
│   │   └── document_service.py     # Orchestration for document ingestion pipeline
│   ├── repositories/
│   │   ├── document_repository.py  # SQLite CRUD for documents and chunks
│   │   └── booking_repository.py   # SQLite CRUD for interview bookings
│   ├── models/
│   │   ├── document.py             # SQLAlchemy Document model
│   │   ├── chunk.py                # SQLAlchemy Chunk model
│   │   └── booking.py              # SQLAlchemy Booking model
│   ├── database/
│   │   ├── connection.py           # Engine & SessionLocal setup
│   │   └── init_db.py              # Schema initialization and migration
│   └── utils/
│       ├── exceptions.py           # Custom exception hierarchy & HTTP mapping
│       └── validators.py           # File size, extension, and content validators
├── tests/
│   ├── test_health.py              # Operational health & root endpoint tests
│   ├── test_chunking.py            # Unit tests for chunking strategies
│   ├── test_documents.py           # Unit tests for text extraction & upload
│   ├── test_chat.py                # Unit tests for custom RAG & Redis memory
│   └── test_booking.py             # Unit tests for interview booking logic
├── .env.example                    # Environment variable template
├── .gitignore                      # Git ignore rules
├── docker-compose.yml              # Qdrant and Redis services
├── Dockerfile                      # Application container definition
├── pytest.ini                      # Pytest async and warning configuration
├── requirements.txt                # Pinned production and test dependencies
└── README.md                       # Comprehensive system documentation
```

---

## 🛠️ Technology Stack

| Component | Technology | Description |
|---|---|---|
| **Web Framework** | FastAPI | High-performance, async Python web API framework |
| **Validation** | Pydantic v2 / Settings | Strict schema definition and type-safe environment management |
| **Server** | Uvicorn | ASGI server implementation |
| **Vector Database** | Qdrant | Fast vector search engine with payload filtering |
| **Embeddings** | Sentence Transformers | Local `all-MiniLM-L6-v2` model (384 dimensions) |
| **Conversational Memory** | Redis | Session-isolated multi-turn conversation caching |
| **Metadata & Bookings** | SQLite + SQLAlchemy 2.x | Relational persistence with ORM mappings |
| **LLM Provider** | Google Gemini API | Custom RAG grounding and structured booking extraction |
| **PDF Extraction** | PyPDF | Robust PDF text extraction |
| **Testing** | pytest + pytest-asyncio + HTTPX | Automated unit and integration testing suite |

---

## 🚀 Quickstart & Setup Guide

### 1. Prerequisites
- **Python**: 3.11+
- **Docker & Docker Compose**: (Required for running Qdrant and Redis locally)
- **Git**

### 2. Environment Configuration
Create your `.env` configuration file from `.env.example`:
```bash
cp .env.example .env
```
Edit `.env` to configure your keys (e.g. `GEMINI_API_KEY`).

### 3. Spin Up Auxiliary Services (Qdrant & Redis)
Launch the vector database and cache services using Docker Compose:
```bash
docker compose up -d
```
Verify services:
- **Qdrant REST API**: `http://localhost:6333`
- **Qdrant Web Dashboard**: `http://localhost:6333/dashboard`
- **Redis Server**: `localhost:6379`

### 4. Create Virtual Environment & Install Dependencies
```bash
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 5. Run the Application
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The application exposes:
- **Swagger Interactive API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc Documentation**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Operational Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

---

## 🧪 Running Automated Tests

Run the complete test suite:
```bash
pytest
```
To run tests with verbose output:
```bash
pytest -v
```

---

## 🗺️ Project Implementation Roadmap

- [x] **Phase 1: Project Setup and Architecture** (Directory structure, dependencies, configuration, Docker Compose, health endpoint)
- [ ] **Phase 2: Text Extraction and Two Chunking Strategies** (PDF & TXT parsers, Fixed-size chunking, Recursive character chunking)
- [ ] **Phase 3: Embedding Generation and Qdrant Integration** (Local `all-MiniLM-L6-v2`, collection setup, indexing, similarity search)
- [ ] **Phase 4: SQL Database and Document Ingestion API** (SQLAlchemy models, metadata persistence, `POST /api/v1/documents/upload`)
- [ ] **Phase 5: Custom Conversational RAG with Gemini** (Prompt assembly, semantic retrieval, context grounding without LangChain chains)
- [ ] **Phase 6: Redis-Based Multi-Turn Memory** (Session isolation, chronological message history, follow-up query context)
- [ ] **Phase 7: LLM-Based Interview Booking** (Intent recognition, structured validation for name/email/date/time, SQLite persistence)
- [ ] **Phase 8: Comprehensive Testing, Refactoring, Documentation, and GitHub Submission**

