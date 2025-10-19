Voici un README complet pour votre projet :

# RAG Pipeline with Agno & PostgreSQL

A complete Retrieval-Augmented Generation (RAG) pipeline that processes PDF documents, generates embeddings, and enables intelligent question-answering using Agno framework and PostgreSQL with pgvector.

## 🚀 Quick Start

### 1. Configure the Environment

Update the `.env` file with your configuration:

```bash
# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=admin_user
POSTGRES_PASSWORD=my_strong_password
POSTGRES_DB=rag_db

# Table names for Agno compatibility
POSTGRES_KNOWLEDGE_TABLE_NAME=knowledge_contents
POSTGRES_VECTOR_TABLE_NAME=documents

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:1b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# Chunking Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### 2. Install Dependencies

Using pip:
```bash
pip install -r requirements.txt
```

Or using uv (recommended):
```bash
uv sync
```

### 3. Launch PostgreSQL with Docker

```bash
docker run -d \
  -e POSTGRES_DB=rag_db \
  -e POSTGRES_USER=admin_user \
  -e POSTGRES_PASSWORD=my_strong_password \
  -e PGDATA=/var/lib/postgresql/data/pgdata \
  -v pgvolume:/var/lib/postgresql/data \
  -p 5432:5432 \
  --name pgvector \
  agnohq/pgvector:16
```

### 4. Add PDF Files and Process

Place your PDF documents in the `knowledge/` directory, then run:

```bash
python start.py
```

### 5. Launch the API Server

```bash
fastapi dev main.py
```

## 📁 Project Structure

```
BISMILLAH/
├── knowledge/                 # Directory for PDF documents
│   ├── document1.pdf
│   └── document2.pdf
├── utils/                    # Utility modules
│   ├── __init__.py
│   ├── pg_database_utils.py  # Database operations
│   ├── embedding_utils.py    # Embedding generation
│   ├── pdf_utils.py          # PDF processing
│   ├── ollama_utils.py       # Ollama configuration
│   └── agent_utils.py        # Agno agent setup
├── main.py                   # FastAPI application
├── start.py                  # Pipeline starter
├── requirements.txt          # Python dependencies
└── .env                     # Environment variables
```

## 🔧 Core Components

### Database Setup
- **PostgreSQL** with **pgvector** extension for vector storage
- Two main tables: `knowledge_contents` (metadata) and `documents` (vector embeddings)
- Automatic schema creation and indexing

### PDF Processing
- Loads PDFs from `knowledge/` directory
- Splits documents into chunks with configurable size and overlap
- Extracts metadata and generates embeddings

### Embedding Generation
- Uses **Ollama** with configurable models
- Supports multiple embedding models
- Batch processing for efficiency

### RAG Agent
- **Agno framework** for intelligent question-answering
- Vector similarity search
- Context-aware responses

## 🛠️ API Endpoints

### Query Documents
```http
POST /query
Content-Type: application/json

{
  "question": "What are the main research findings?",
  "k": 5
}
```

### List Documents
```http
GET /documents
```

### Database Stats
```http
GET /stats
```

## 📊 Features

- **Multi-document support**: Process multiple PDFs simultaneously
- **Configurable chunking**: Adjust chunk size and overlap
- **Vector similarity search**: Find relevant content using embeddings
- **Interactive chat**: Real-time Q&A with your documents
- **RESTful API**: Easy integration with other applications
- **Docker-ready**: Containerized database setup

## 🎯 Usage Examples

### Process PDFs and Create Embeddings
```python
from utils.embedding_utils import main_pipeline_embedding_process_pdfs_to_pgvector

# Process all PDFs in knowledge directory
chunks = main_pipeline_embedding_process_pdfs_to_pgvector()
```

### Query Your Documents
```python
from utils.embedding_utils import search_similar_chunks

# Search for relevant content
results = search_similar_chunks("What is machine learning?", k=5)
```

### Use the Agno Agent
```python
from utils.agent_utils import create_agno_agent

agent, knowledge = create_agno_agent()
response = await agent.aprint_response("Explain the key concepts in the documents.")
```

## 🔍 Monitoring

Check database status:
```python
from utils.pg_database_utils import list_documents_in_db, get_database_stats

list_documents_in_db()
get_database_stats()
```

## 🐛 Troubleshooting

### Common Issues

1. **Database Connection Failed**
   - Verify PostgreSQL is running: `docker ps`
   - Check credentials in `.env` file

2. **Ollama Not Responding**
   - Ensure Ollama is running: `ollama serve`
   - Verify model names: `ollama list`

3. **No PDFs Processed**
   - Check `knowledge/` directory contains PDF files
   - Verify file permissions

4. **Vector Search Not Working**
   - Confirm pgvector extension is enabled
   - Check embedding dimensions match (default: 768)

### Debug Mode

Enable debug output by setting:
```python
agent = Agent(show_tool_calls=True, debug=True)
```

## 📈 Performance Tips

- Use `uv` for faster dependency management
- Adjust `CHUNK_SIZE` based on document complexity
- Use GPU-accelerated Ollama for faster embeddings
- Monitor database performance with indexes

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a pull request

## 📄 License

This project is licensed under the MIT License.

---

**Ready to start?** Place your PDFs in the `knowledge/` directory and run `python start.py`!