import os
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values
from typing import List, Dict

# Load environment variables from .env file
load_dotenv()

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))
POSTGRES_USER = os.getenv("POSTGRES_USER", "admin_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "my_strong_password")
POSTGRES_DB = os.getenv("POSTGRES_DB", "rag_db")
POSTGRES_KNOWLEDGE_TABLE_NAME = os.getenv("POSTGRES_KNOWLEDGE_TABLE_NAME", "knowledge_contents")
POSTGRES_VECTOR_TABLE_NAME = os.getenv("POSTGRES_VECTOR_TABLE_NAME", "knowledge_vectors")

DB_CONFIG = {
    "host": POSTGRES_HOST,
    "port": POSTGRES_PORT,
    "user": POSTGRES_USER,
    "password": POSTGRES_PASSWORD,
    "database": POSTGRES_DB,
}

def setup_database():
    """Setup PostgreSQL database with both knowledge_contents and vector tables."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        # Enable pgvector extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Create knowledge_contents table (metadata)
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {POSTGRES_KNOWLEDGE_TABLE_NAME} (
                id TEXT PRIMARY KEY,
                name TEXT,
                description TEXT,
                metadata JSONB,
                type TEXT,
                size INTEGER,
                linked_to TEXT,
                access_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'completed',
                status_message TEXT,
                created_at BIGINT,
                updated_at BIGINT,
                external_id TEXT
            );
            """
        )

        # Create vector table (content + embeddings)
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {POSTGRES_VECTOR_TABLE_NAME} (
                id TEXT PRIMARY KEY,
                name TEXT,
                meta_data JSONB,
                filters JSONB,
                content TEXT,
                embedding VECTOR(768),
                usage TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                content_hash TEXT,
                content_id TEXT
            );
            """
        )

        # Create indexes for better performance
        # Indexes for knowledge_contents
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{POSTGRES_KNOWLEDGE_TABLE_NAME}_name ON {POSTGRES_KNOWLEDGE_TABLE_NAME} (name);")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{POSTGRES_KNOWLEDGE_TABLE_NAME}_type ON {POSTGRES_KNOWLEDGE_TABLE_NAME} (type);")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{POSTGRES_KNOWLEDGE_TABLE_NAME}_status ON {POSTGRES_KNOWLEDGE_TABLE_NAME} (status);")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{POSTGRES_KNOWLEDGE_TABLE_NAME}_created_at ON {POSTGRES_KNOWLEDGE_TABLE_NAME} (created_at);")
        
        # Indexes for vector table
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{POSTGRES_VECTOR_TABLE_NAME}_name ON {POSTGRES_VECTOR_TABLE_NAME} (name);")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{POSTGRES_VECTOR_TABLE_NAME}_content_id ON {POSTGRES_VECTOR_TABLE_NAME} (content_id);")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{POSTGRES_VECTOR_TABLE_NAME}_embedding ON {POSTGRES_VECTOR_TABLE_NAME} USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);")

        conn.commit()
        cur.close()
        conn.close()
        print(f"✅ Database setup complete with tables: {POSTGRES_KNOWLEDGE_TABLE_NAME} and {POSTGRES_VECTOR_TABLE_NAME}")

    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        raise


def build_connection_string() -> str:
    """Build PostgreSQL connection string."""
    return (
        f"postgresql+psycopg://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
        f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )


def list_documents_in_db():
    """List all documents in both tables."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        # Count from knowledge_contents
        cur.execute(f"SELECT COUNT(*) FROM {POSTGRES_KNOWLEDGE_TABLE_NAME}")
        knowledge_count = cur.fetchone()[0]

        # Count from vector table
        cur.execute(f"SELECT COUNT(*) FROM {POSTGRES_VECTOR_TABLE_NAME}")
        vector_count = cur.fetchone()[0]

        print(f"\n📚 Documents in database:")
        print(f"   {POSTGRES_KNOWLEDGE_TABLE_NAME}: {knowledge_count} metadata entries")
        print(f"   {POSTGRES_VECTOR_TABLE_NAME}: {vector_count} vector entries")

        # Show documents by name
        cur.execute(f"SELECT name, type, status, COUNT(*) FROM {POSTGRES_KNOWLEDGE_TABLE_NAME} GROUP BY name, type, status")
        results = cur.fetchall()

        for name, doc_type, status, count in results:
            print(f"   - {name} ({doc_type}, {status}): {count} chunks")

        cur.close()
        conn.close()

    except Exception as e:
        print(f"❌ Failed to list documents: {e}")


def get_database_stats():
    """Get statistics about both tables."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        print(f"\n📊 Database Statistics:")

        # Knowledge contents stats
        cur.execute(f"SELECT COUNT(*), COUNT(DISTINCT name) FROM {POSTGRES_KNOWLEDGE_TABLE_NAME}")
        knowledge_count, unique_docs = cur.fetchone()
        print(f"   {POSTGRES_KNOWLEDGE_TABLE_NAME}:")
        print(f"     - Total entries: {knowledge_count}")
        print(f"     - Unique documents: {unique_docs}")

        # Vector table stats
        cur.execute(f"SELECT COUNT(*), COUNT(DISTINCT name) FROM {POSTGRES_VECTOR_TABLE_NAME}")
        vector_count, vector_unique_docs = cur.fetchone()
        print(f"   {POSTGRES_VECTOR_TABLE_NAME}:")
        print(f"     - Total entries: {vector_count}")
        print(f"     - Unique documents: {vector_unique_docs}")

        # Status distribution
        cur.execute(f"SELECT status, COUNT(*) FROM {POSTGRES_KNOWLEDGE_TABLE_NAME} GROUP BY status")
        status_counts = cur.fetchall()
        print(f"   Status distribution:")
        for status, count in status_counts:
            print(f"     - {status}: {count}")

        cur.close()
        conn.close()

    except Exception as e:
        print(f"❌ Failed to get database stats: {e}")