from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain_core.tools import tool
from agent_graph.load_tools_config import LoadToolsConfig
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

# Carregar as variáveis de ambiente do arquivo .env
load_dotenv()

# Informações de conexão
db_host = os.getenv('POSTGRES_DB_HOST')
db_port = os.getenv('POSTGRES_DB_PORT')
db_name = os.getenv('POSTGRES_DB_NAME')
db_user = os.getenv('POSTGRES_DB_USER')
db_password = os.getenv('POSTGRES_DB_PASSWORD')

# String de conexão
DATABASE_URL = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

# Criando o engine
engine = create_engine(DATABASE_URL)

# Testando a conexão ao inicializar
try:
    with engine.connect() as connection:
        print("Conexão ao banco de dados PostgreSQL estabelecida com sucesso.")
except Exception as error:
    print(f"Erro ao conectar ao banco de dados PostgreSQL: {error}")

@tool
def query_postgres_sqldb(query: str) -> str:
    """Query the PostgreSQL Database. Input should be a search query."""
    try:
        with engine.connect() as connection:
            result = connection.execute(query)
            return str(result.fetchall())
    except Exception as error:
        return f"Erro ao executar a consulta: {error}"