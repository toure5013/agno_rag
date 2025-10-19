import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_API_BASE = os.getenv("OLLAMA_EMBEDDING_MODEL_NAME", "nomic-embed-text")
OLLAMA_EMBEDDING_MODEL_NAME = os.getenv("OLLAMA_EMBEDDING_MODEL_NAME", "nomic-embed-text")
OLLAMA_BASE_URL= os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


OLLAMA_EMBEDDING_CONFIG = {
    'base_url':  OLLAMA_BASE_URL,
    'model': OLLAMA_EMBEDDING_MODEL_NAME
}

OLLAMA_CONFIG = {
    'base_url': OLLAMA_BASE_URL + "/v1",
    'model': OLLAMA_MODEL
}

