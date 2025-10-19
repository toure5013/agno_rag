import asyncio
from agno.agent import Agent, RunOutput, RunOutputEvent, RunEvent
from agno.models.ollama import Ollama
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.pgvector import PgVector
from agno.knowledge.embedder.ollama import OllamaEmbedder
from agno.utils.pprint import pprint_run_response
from agno.db.postgres.postgres import PostgresDb

from typing import Iterator
import os
from dotenv import load_dotenv

from utils.pg_database_utils import POSTGRES_VECTOR_TABLE_NAME, build_connection_string
from utils.ollama_utils import OLLAMA_EMBEDDING_CONFIG, OLLAMA_CONFIG


local_llama = Ollama(id=OLLAMA_CONFIG["model"])

db_url = build_connection_string()

db = PostgresDb(
    db_url=db_url,
    knowledge_table="knowledge_contents",
)

knowledge = Knowledge(
    contents_db=db,
    vector_db=PgVector(
        table_name="recipes",
        db_url=db_url,
          embedder=OllamaEmbedder(id=OLLAMA_EMBEDDING_CONFIG["model"],  dimensions=768),
    )
)

agent = Agent(
    model=local_llama,
    db=db,
    knowledge=knowledge,
    markdown=True,
)
if __name__ == "__main__":
    asyncio.run(
        knowledge.add_content_async(
            name="Recipes",
            url="https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf",
            metadata={"user_tag": "Recipes from website"}
        )
        
    )
    # Create and use the agent
    asyncio.run(
        agent.aprint_response(
            "How do I make chicken and galangal in coconut milk soup?",
            markdown=True,
        )
    )