import os
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

# Document processing
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables from .env file
load_dotenv()

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

# Determine script directory
base_dir = Path(__file__).parent

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"

def load_pdf(pdf_path: str) -> List[Dict]:
    """Load and split a single PDF into chunks with Agno-compatible format."""
    try:
        # Load PDF
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(pages)

        # Convert to Agno-compatible dictionaries
        doc_chunks = []
        doc_id = Path(pdf_path).stem

        for i, chunk in enumerate(chunks):
            doc_chunks.append(
                {
                    "name": doc_id,  # Agno: name field
                    "description": f"Chunk {i} from {doc_id}",  # Agno: description field
                    "type": "document",  # Agno: type field
                    "content": chunk.page_content,  # Agno: content field
                    "metadata": {
                        "source": pdf_path,
                        "page": chunk.metadata.get("page", 0),
                        "doc_id": doc_id,
                        "chunk_index": i,
                    },  # Agno: metadata field
                    "external_id": f"{doc_id}_{i}",  # Agno: external_id field
                    "size": len(chunk.page_content),  # Agno: size field
                    # Additional fields for Agno compatibility
                    "linked_to": None,
                    "status": "completed",
                    "status_message": None,
                    "access_count": 0,
                }
            )

        print(f"✅ Loaded {pdf_path}: {len(doc_chunks)} chunks (Agno-compatible)")
        return doc_chunks

    except Exception as e:
        print(f"❌ Failed to load {pdf_path}: {e}")
        import traceback
        traceback.print_exc()
        return []

def load_pdfs(pdf_paths: List[str] = None) -> List[Dict]:
    """Load multiple PDFs and return all chunks in Agno-compatible format."""
    all_chunks = []
    
    if pdf_paths is None:
        pdf_paths = get_pdf_files_from_knowledge_dir()
        
    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            print(f"⚠️  File not found: {pdf_path}")
        else:
            chunks = load_pdf(pdf_path)
            all_chunks.extend(chunks)
        
    print(f"📄 Total chunks loaded from PDFs: {len(all_chunks)}")
    
    return all_chunks

def get_pdf_files_from_knowledge_dir() -> List[str]:
    """Get list of PDF files from the knowledge directory."""
    pdf_files = list(KNOWLEDGE_DIR.glob("*.pdf"))
    pdf_files = [str(p) for p in pdf_files]
    print("📄 List of PDFs to process:", pdf_files)
    return pdf_files


# Fonction utilitaire pour vérifier le format des chunks
def validate_agno_chunk_format(chunk: Dict) -> bool:
    """Validate that a chunk has all required Agno fields."""
    required_fields = {
        'name', 'description', 'type', 'content', 'metadata',
        'external_id', 'size', 'status', 'access_count'
    }
    
    missing_fields = required_fields - set(chunk.keys())
    if missing_fields:
        print(f"⚠️  Missing Agno fields: {missing_fields}")
        return False
    
    return True


def print_chunk_sample(chunks: List[Dict], sample_size: int = 2):
    """Print a sample of chunks for verification."""
    if not chunks:
        print("📝 No chunks to display")
        return
        
    print(f"\n📝 Sample of first {min(sample_size, len(chunks))} chunks:")
    for i, chunk in enumerate(chunks[:sample_size]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Name: {chunk.get('name')}")
        print(f"Type: {chunk.get('type')}")
        print(f"External ID: {chunk.get('external_id')}")
        print(f"Content preview: {chunk.get('content', '')[:100]}...")
        print(f"Metadata keys: {list(chunk.get('metadata', {}).keys())}")