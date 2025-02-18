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

TOOLS_CFG = LoadToolsConfig()


class Table(BaseModel):
    """
    Representa uma tabela no banco de dados SQL.

    Attributos:
        name (str): O nome da tabela no banco de dados SQL.
    """

    name: str = Field(description="O nome da tabela no banco de dados SQL.")


def get_tables(categories: List[Table]) -> List[str]:
    """Mapeia nomes de categorias para nomes de tabelas SQL correspondentes.

    Argumentos:
        categories (List[Table]): Uma lista de objetos `Table` representando diferentes categorias.

    Retornos:
        List[str]: Uma lista de nomes de tabelas SQL correspondentes às categorias fornecidas.
    """
    tables = []
    for category in categories:
        if category.name == "Music":
            tables.extend(
                [
                    "Album",
                    "Artist",
                    "Genre",
                    "MediaType",
                    "Playlist",
                    "PlaylistTrack",
                    "Track",
                ]
            )
        elif category.name == "Business":
            tables.extend(
                ["Customer", "Employee", "Invoice", "InvoiceLine"])
    return tables


class ChinookSQLAgent:
    """
    Um agente SQL especializado que interage com o banco de dados Chinook SQL usando um LLM (Large Language Model).

    O agente lida com consultas SQL mapeando as perguntas do usuário para tabelas SQL relevantes com base em categorias como "Música"
    e "Negócios". Ele usa uma cadeia de extração para determinar tabelas relevantes com base na pergunta e depois
    executa consultas no banco de dados usando as tabelas apropriadas.

    Attributos:
        sql_agent_llm (ChatOpenAI): O modelo de linguagem usado para interpretar e interagir com o banco de dados.
        db (SQLDatabase): O objeto de banco de dados SQL, representando o banco de dados Chinook.
        full_chain (Runnable): Uma cadeia de operações que mapeia as perguntas do usuário para tabelas SQL e executa consultas.

    Metodos:
        __init__: Inicializa o agente configurando o LLM, conectando-se ao banco de dados SQL e criando cadeias de consulta.

    Argumentos:
        sqldb_directory (str): O diretório onde o arquivo de banco de dados Chinook SQLite está localizado.
        llm (str): O nome do modelo LLM a ser usado (por exemplo, "gpt-3.5-turbo").
        llm_temperature (float): A configuração de temperatura para o LLM, controlando a aleatoriedade das respostas.
    """

    def __init__(self, sqldb_directory: str, llm: str, llm_temerature: float) -> None:
        """Inicializa o ChinookSQLAgent com o LLM e a conexão com o banco de dados.

        Argumentos:
            sqldb_directory (str): O caminho do diretório para o arquivo de banco de dados SQLite.
            llm (str): O identificador do modelo LLM (por exemplo, "gpt-3.5-turbo").
            llm_temerature (float): O valor da temperatura para o LLM, determinando a aleatoriedade da saída do modelo.
        """
        self.sql_agent_llm = ChatOpenAI(
            model=llm, temperature=llm_temerature)

        self.db = SQLDatabase.from_uri(f"sqlite:///{sqldb_directory}")
        print(self.db.get_usable_table_names())
        category_chain_system = """Retornae os nomes das tabelas SQL relevantes para a pergunta do usuário. \
        TAs tabelas são:

        Music
        Business"""
        category_chain = create_extraction_chain_pydantic(
            Table, self.sql_agent_llm, system_message=category_chain_system)
        table_chain = category_chain | get_tables  # noqa
        query_chain = create_sql_query_chain(self.sql_agent_llm, self.db)
        # Convert "question" key to the "input" key expected by current table_chain.
        table_chain = {"input": itemgetter("question")} | table_chain
        # Set table_names_to_use using table_chain.
        self.full_chain = RunnablePassthrough.assign(
            table_names_to_use=table_chain) | query_chain


@tool
def query_chinook_sqldb(query: str) -> str:
    """Query the Chinook SQL Database. Input should be a search query."""
    # Create an instance of ChinookSQLAgent
    agent = ChinookSQLAgent(
        sqldb_directory=TOOLS_CFG.chinook_sqldb_directory,
        llm=TOOLS_CFG.chinook_sqlagent_llm,
        llm_temerature=TOOLS_CFG.chinook_sqlagent_llm_temperature
    )

    query = agent.full_chain.invoke({"question": query})

    return agent.db.run(query)
