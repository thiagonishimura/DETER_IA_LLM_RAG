from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool
from agent_graph.load_tools_config import LoadToolsConfig

TOOLS_CFG = LoadToolsConfig()


class SwissAirlinePolicyRAGTool:
    """
    Uma ferramenta para recuperar documentos relevantes da política da Swiss Airline usando um 
    Abordagem de geração aumentada de recuperação (RAG) com incorporações de vetores.

    Esta ferramenta usa um modelo de incorporação OpenAI pré-treinado para transformar consultas em 
    representações vetoriais. Esses vetores são então usados ​​para consultar um arquivo baseado em Chroma. 
    banco de dados vetorial (persistido no disco) para recuperar os top-k mais relevantes 
    documentos ou entradas de uma coleção específica, como apólices da Swiss Airline.

    Atributos:
        embedding_model (str): O nome do modelo de incorporação OpenAI usado para 
            gerando representações vetoriais das consultas.
        vectordb_dir (str): O diretório onde está o banco de dados vetorial Chroma 
            persistiu no disco.
        k (int): O número dos k vizinhos mais próximos (documentos mais relevantes) 
            para recuperar do banco de dados vetorial.
        vectordb (Chroma): A instância do banco de dados vetorial Chroma conectada ao 
            coleção especificada e modelo de incorporação.

    Metodos:
        __init__: Inicializa a ferramenta configurando o modelo de incorporação, 
            banco de dados vetorial e parâmetros de recuperação.
    """

    def __init__(self, embedding_model: str, vectordb_dir: str, k: int, collection_name: str) -> None:
        """
        Inicializa o SwissAirlinePolicyRAGTool com a configuração necessária.

        Argumentos:
            embedding_model (str): O nome do modelo de incorporação (por exemplo, "text-embedding-ada-002")
                usado para converter consultas em representações vetoriais.
            vectordb_dir (str): O caminho do diretório onde o banco de dados vetorial Chroma está armazenado 
                e persistiu no disco.
            k (int): O número de documentos vizinhos mais próximos a serem recuperados com base na similaridade da consulta.
            nome_coleção (str): O nome da coleção dentro do banco de dados vetorial que contém 
                os documentos da política da Swiss Airline.
        """
        self.embedding_model = embedding_model
        self.vectordb_dir = vectordb_dir
        self.k = k
        self.vectordb = Chroma(
            collection_name=collection_name,
            persist_directory=self.vectordb_dir,
            embedding_function=OpenAIEmbeddings(model=self.embedding_model)
        )
        print("Number of vectors in vectordb:",
              self.vectordb._collection.count(), "\n\n")


@tool
def lookup_swiss_airline_policy(query: str) -> str:
    """Consulte as políticas da empresa para verificar se determinadas opções são permitidas."""
    rag_tool = SwissAirlinePolicyRAGTool(
        embedding_model=TOOLS_CFG.policy_rag_embedding_model,
        vectordb_dir=TOOLS_CFG.policy_rag_vectordb_directory,
        k=TOOLS_CFG.policy_rag_k,
        collection_name=TOOLS_CFG.policy_rag_collection_name)
    docs = rag_tool.vectordb.similarity_search(query, k=rag_tool.k)
    return "\n\n".join([doc.page_content for doc in docs])
