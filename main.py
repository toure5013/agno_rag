import asyncio
from agno.agent import Agent, RunOutput, RunOutputEvent, RunEvent
from agno.models.ollama import Ollama
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.pgvector import PgVector
from agno.knowledge.embedder.ollama import OllamaEmbedder
from agno.utils.pprint import pprint_run_response
from agno.db.postgres.postgres import PostgresDb

from agno.os import AgentOS



from utils.pg_database_utils import (
    POSTGRES_VECTOR_TABLE_NAME,
    build_connection_string,
    POSTGRES_KNOWLEDGE_TABLE_NAME,
)
from utils.ollama_utils import OLLAMA_EMBEDDING_CONFIG, OLLAMA_CONFIG


local_llama = Ollama(id=OLLAMA_CONFIG["model"])

db_url = build_connection_string()

db = PostgresDb(
    db_schema="public",
    db_url=db_url,
    knowledge_table=POSTGRES_KNOWLEDGE_TABLE_NAME,
)

knowledge = Knowledge(
    name="Deepseek research papers",
    description="A database of research papers related to DeepSeek project",
    contents_db=db,
    vector_db=PgVector(
        schema="public",
        table_name=POSTGRES_VECTOR_TABLE_NAME,
        db_url=db_url,
        embedder=OllamaEmbedder(id=OLLAMA_EMBEDDING_CONFIG["model"], dimensions=768),
    ),
)


agent = Agent(
    name="DeepSeek Research Paper Agent",
    description="An agent that answers questions about DeepSeek research papers.",
    model=local_llama,
    db=db,
    knowledge=knowledge,
    markdown=True,
    add_history_to_context=True,
)


agent_os = AgentOS(agents=[agent])
# Get the FastAPI app for the AgentOS
app = agent_os.get_app()