import json
from dotenv import load_dotenv
import time
import psycopg2
from psycopg2.extras import execute_values
from typing import List, Dict
import uuid
import hashlib

# Document processing
from langchain_ollama import OllamaEmbeddings

# Load utility functions
from .pdf_utils import load_pdfs
from .ollama_utils import OLLAMA_EMBEDDING_CONFIG
from .pg_database_utils import DB_CONFIG, POSTGRES_KNOWLEDGE_TABLE_NAME, POSTGRES_VECTOR_TABLE_NAME, setup_database

# Load environment variables from .env file
load_dotenv()

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings using Ollama."""
    print(f"🔄 Generating embeddings for {len(texts)} documents...")
    try:
        embeddings_model = OllamaEmbeddings(
            model=OLLAMA_EMBEDDING_CONFIG["model"], 
            base_url=OLLAMA_EMBEDDING_CONFIG["base_url"]
        )

        embeddings = embeddings_model.embed_documents(texts)
        print(f"✅ Embeddings generated for {len(texts)} texts")
        return embeddings

    except Exception as e:
        print(f"❌ Failed to generate embeddings: {e}")
        raise

def generate_content_hash(content: str) -> str:
    """Generate SHA256 hash for content deduplication."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def insert_chunks_to_db(chunks: List[Dict]):
    """Insert document chunks into both knowledge_contents and vector tables."""
    if not chunks:
        print("⚠️  No chunks to insert")
        return

    try:
        print(f"💾 Inserting {len(chunks)} chunks into PostgreSQL with dual-table schema...")
        
        # Get embeddings for all chunks
        texts = [chunk['content'] for chunk in chunks]
        embeddings = get_embeddings(texts)

        # Prepare data for both tables
        knowledge_data = []
        vector_data = []
        current_time = int(time.time())
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Remove NUL characters from content
            clean_content = chunk['content'].replace('\x00', ' ')
            
            # Generate IDs
            content_id = str(uuid.uuid4())
            vector_id = generate_content_hash(clean_content)  # Use content hash as vector ID
            
            # Generate content hash
            content_hash = generate_content_hash(clean_content)
            
            # Prepare knowledge_contents data
            knowledge_data.append((
                content_id,  # id
                chunk.get('name', 'unknown'),  # name
                chunk.get('description', ''),  # description
                json.dumps(chunk.get('metadata', {})),  # metadata
                chunk.get('type', 'document'),  # type
                chunk.get('size', 0),  # size
                chunk.get('linked_to'),  # linked_to
                chunk.get('access_count', 0),  # access_count
                'completed',  # status
                chunk.get('status_message'),  # status_message
                current_time,  # created_at
                current_time,  # updated_at
                chunk.get('external_id', '')  # external_id
            ))

            # Prepare vector data
            vector_data.append((
                vector_id,  # id (content hash)
                chunk.get('name', 'unknown'),  # name
                json.dumps(chunk.get('metadata', {})),  # meta_data
                json.dumps({"source": "local_pdf"}),  # filters
                clean_content,  # content
                embedding,  # embedding
                None,  # usage
                f"NOW()",  # created_at (will use PostgreSQL NOW())
                f"NOW()",  # updated_at (will use PostgreSQL NOW())
                content_hash,  # content_hash
                content_id  # content_id (links to knowledge_contents)
            ))

        # Insert into database
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        # Insert into knowledge_contents
        knowledge_sql = f"""
        INSERT INTO {POSTGRES_KNOWLEDGE_TABLE_NAME} (
            id, name, description, metadata, type, size, linked_to, 
            access_count, status, status_message, created_at, updated_at, external_id
        ) VALUES %s
        """
        execute_values(cur, knowledge_sql, knowledge_data, page_size=100)

        # Insert into vector table
        vector_sql = f"""
        INSERT INTO {POSTGRES_VECTOR_TABLE_NAME} (
            id, name, meta_data, filters, content, embedding, usage,
            created_at, updated_at, content_hash, content_id
        ) VALUES %s
        """
        execute_values(cur, vector_sql, vector_data, page_size=100)

        conn.commit()
        cur.close()
        conn.close()

        print(f"✅ Successfully inserted:")
        print(f"   - {len(knowledge_data)} entries into {POSTGRES_KNOWLEDGE_TABLE_NAME}")
        print(f"   - {len(vector_data)} entries into {POSTGRES_VECTOR_TABLE_NAME}")

    except Exception as e:
        print(f"❌ Failed to insert chunks: {e}")
        raise

def search_similar_chunks(query: str, k: int = 5) -> List[Dict]:
    """Search for similar chunks using the vector table."""
    try:
        # Get query embedding
        embeddings_model = OllamaEmbeddings(
            model=OLLAMA_EMBEDDING_CONFIG["model"], 
            base_url=OLLAMA_EMBEDDING_CONFIG["base_url"]
        )

        query_embedding = embeddings_model.embed_query(query)
        print(f"🔍 Searching for: '{query}'")

        # Search in vector table
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        search_sql = f"""
        SELECT v.name, v.content, v.meta_data, v.content_id,
               1 - (v.embedding <=> %s::vector) as similarity
        FROM {POSTGRES_VECTOR_TABLE_NAME} v
        ORDER BY v.embedding <=> %s::vector
        LIMIT %s
        """

        cur.execute(search_sql, (query_embedding, query_embedding, k))
        results = cur.fetchall()

        cur.close()
        conn.close()

        # Format results
        search_results = []
        for row in results:
            search_results.append({
                "name": row[0],
                "content": row[1],
                "metadata": row[2],
                "content_id": row[3],
                "similarity": float(row[4]),
            })

        print(f"✅ Found {len(search_results)} similar chunks")
        return search_results

    except Exception as e:
        print(f"❌ Search failed: {e}")
        import traceback
        traceback.print_exc()
        return []

def validate_chunks_compatibility(chunks: List[Dict]) -> bool:
    """Validate that chunks have all required fields for both tables."""
    required_fields = {
        'name', 'description', 'type', 'content', 'metadata', 
        'external_id', 'size', 'status', 'access_count'
    }
    
    for i, chunk in enumerate(chunks):
        missing_fields = required_fields - set(chunk.keys())
        if missing_fields:
            print(f"❌ Chunk {i} missing required fields: {missing_fields}")
            return False
    
    print("✅ All chunks are compatible with dual-table schema")
    return True

def print_insertion_summary(chunks: List[Dict]):
    """Print summary of chunks to be inserted."""
    if not chunks:
        print("📊 No chunks to summarize")
        return
    
    print(f"\n📊 Insertion Summary:")
    print(f"   Total chunks: {len(chunks)}")
    
    # Group by document name
    doc_counts = {}
    for chunk in chunks:
        doc_name = chunk.get('name', 'unknown')
        doc_counts[doc_name] = doc_counts.get(doc_name, 0) + 1
    
    for doc_name, count in doc_counts.items():
        print(f"   - {doc_name}: {count} chunks")
    
    # Show sample of fields
    sample = chunks[0]
    print(f"\n📝 Sample chunk fields:")
    for key, value in sample.items():
        if key != 'content':  # Don't print full content
            value_preview = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
            print(f"   - {key}: {value_preview}")

'''------------------- MAIN PIPELINE -------------------'''
def main_pipeline_embedding_process_pdfs_to_pgvector():
    """Main function to process PDFs and store in dual-table schema."""
    print("🚀 Starting PDF to PGVector pipeline with dual-table schema...")

    # Step 1: Setup database with both tables
    setup_database()

    # Step 2: Load and chunk PDFs
    chunks = load_pdfs()

    if not chunks:
        print("❌ No chunks to process!")
        return

    # Step 2.5: Validate chunks compatibility
    if not validate_chunks_compatibility(chunks):
        print("❌ Chunks are not compatible. Please check pdf_utils.load_pdfs()")
        return

    # Step 2.6: Print insertion summary
    print_insertion_summary(chunks)

    # Step 3: Store in both tables
    insert_chunks_to_db(chunks)

    # Step 4: Show database statistics
    from .pg_database_utils import get_database_stats
    get_database_stats()

    print("✅ Dual-table pipeline completed successfully!")
    return chunks

if __name__ == "__main__":
    main_pipeline_embedding_process_pdfs_to_pgvector()