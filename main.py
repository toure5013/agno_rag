import asyncio
from agno.agent import Agent, RunOutput, RunOutputEvent, RunEvent
from agno.models.ollama import Ollama
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.pgvector import PgVector
from agno.knowledge.embedder.ollama import OllamaEmbedder
from agno.utils.pprint import pprint_run_response
from agno.db.postgres.postgres import PostgresDb

from agno.os import AgentOS
from agno.tools.mcp import MCPTools



prompt_system = """Vous êtes un Data Scientist expert en analyse de sentiments et en traitement du langage naturel (NLP).
Votre mission est d’analyser les emails des clients B2B d’Orange Côte d’Ivoire afin d’identifier :

les raisons principales de leur insatisfaction,

les thèmes récurrents évoqués par les clients,

et les axes d’amélioration pour optimiser leur expérience.

Vous devez :

Détecter le ton et le sentiment général du message (positif, négatif, neutre, mixte).

Extraire les causes d’insatisfaction (ex : lenteur du service, problème de facturation, support non réactif, qualité réseau, etc.).

Proposer des recommandations concrètes pour améliorer l’expérience client.

Structurer la réponse de façon claire, hiérarchisée et exploitable par les équipes métiers d’Orange CI.

Répondez dans un format structuré :

🧠 Analyse du sentiment :
🔍 Thèmes identifiés :
⚠️ Causes d’insatisfaction :
💡 Recommandations :


NB : 
- retient de toujours bien affichés la reponse pourqu'il facile à lire , 
- même si la reponse te viens en JSON, toi affiche dans un bon format facilement interpretable
- Sache que tes analyses sont sur l'année 2025 et ulterieur
"""

from utils.pg_database_utils import (
    POSTGRES_VECTOR_TABLE_NAME,
    build_connection_string,
    POSTGRES_KNOWLEDGE_TABLE_NAME,
)
from utils.ollama_utils import OLLAMA_EMBEDDING_CONFIG, OLLAMA_CONFIG


local_llama = Ollama(id=OLLAMA_CONFIG["model"])

# Mcp integration

# Create MCPTools instance
mcp_tools = MCPTools(
    transport="streamable-http", 
    url="http://127.0.0.1:8001/mcp"
)



# Build database connection
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
    description=prompt_system,
    model=local_llama,
    db=db,
    knowledge=knowledge,
    markdown=True,
    add_history_to_context=True,
    tools=[mcp_tools],
)




agent_os = AgentOS(agents=[agent])
# Get the FastAPI app for the AgentOS
app = agent_os.get_app()