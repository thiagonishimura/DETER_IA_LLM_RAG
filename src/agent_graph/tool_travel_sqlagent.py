from langchain_core.tools import tool
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain_openai import ChatOpenAI
from agent_graph.load_tools_config import LoadToolsConfig

TOOLS_CFG = LoadToolsConfig()


class TravelSQLAgentTool:
    """
    Uma ferramenta para interagir com um banco de dados SQL relacionado a viagens usando um LLM (Language Model) para gerar e executar consultas SQL.

    Esta ferramenta permite aos usuários fazer perguntas relacionadas a viagens, que são transformadas em consultas SQL por um modelo de linguagem.
    As consultas SQL são executadas no banco de dados SQLite fornecido e os resultados são processados ​​pelo modelo de linguagem para
    gerar uma resposta final para o usuário.

    Atributos:
        sql_agent_llm (ChatOpenAI): uma instância de um modelo de linguagem ChatOpenAI usado para gerar e processar consultas SQL.
        system_role (str): Um modelo de prompt do sistema que orienta o modelo de linguagem na resposta às perguntas do usuário com base nos resultados da consulta SQL.
        db (SQLDatabase): Uma instância do banco de dados SQL usada para executar consultas.
        cadeia (RunnablePassthrough): Uma cadeia de operações que cria consultas SQL, as executa e gera uma resposta.

    Métodos:
        __init__: inicializa o TravelSQLAgentTool configurando o modelo de linguagem, o banco de dados SQL e o pipeline de resposta a consultas.
    """

    def __init__(self, llm: str, sqldb_directory: str, llm_temerature: float) -> None:
        """
        Inicializa o TravelSQLAgentTool com as configurações necessárias.

        Argumentos:
            llm (str): O nome do modelo de linguagem a ser usado para gerar e interpretar consultas SQL.
            sqldb_directory (str): O caminho do diretório onde o banco de dados SQLite está armazenado.
            llm_temerature (float): A configuração de temperatura para o modelo de linguagem, controlando a aleatoriedade da resposta.
        """
        self.sql_agent_llm = ChatOpenAI(
            model=llm, temperature=llm_temerature)
        self.system_role = """Given the following user question, corresponding SQL query, and SQL result, answer the user question.\n
            Question: {question}\n
            SQL Query: {query}\n
            SQL Result: {result}\n
            Answer:
            """
        self.db = SQLDatabase.from_uri(
            f"sqlite:///{sqldb_directory}")
        print(self.db.get_usable_table_names())

        execute_query = QuerySQLDataBaseTool(db=self.db)
        write_query = create_sql_query_chain(
            self.sql_agent_llm, self.db)
        answer_prompt = PromptTemplate.from_template(
            self.system_role)

        answer = answer_prompt | self.sql_agent_llm | StrOutputParser()
        self.chain = (
            RunnablePassthrough.assign(query=write_query).assign(
                result=itemgetter("query") | execute_query
            )
            | answer
        )


@tool
def query_travel_sqldb(query: str) -> str:
    """Consulte o banco de dados SQL da Swiss Airline e acesse todas as informações da empresa. A entrada deve ser uma consulta de pesquisa."""
    agent = TravelSQLAgentTool(
        llm=TOOLS_CFG.travel_sqlagent_llm,
        sqldb_directory=TOOLS_CFG.travel_sqldb_directory,
        llm_temerature=TOOLS_CFG.travel_sqlagent_llm_temperature
    )
    response = agent.chain.invoke({"question": query})
    return response
