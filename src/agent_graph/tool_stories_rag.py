from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool
from agent_graph.load_tools_config import LoadToolsConfig

TOOLS_CFG = LoadToolsConfig()


class StoriesRAGTool:
    """
    Uma ferramenta para recuperar histórias relevantes usando uma abordagem Retrieval-Augmented Generation (RAG) com incorporações de vetores.

    Esta ferramenta aproveita um modelo de incorporação OpenAI pré-treinado para transformar as consultas do usuário em incorporações de vetores.
    Em seguida, ele usa esses embeddings para consultar um banco de dados de vetores baseado em Chroma para recuperar os k mais relevantes
    histórias de uma coleção específica armazenada no banco de dados.

    Atributos:
        embedding_model (str): O nome do modelo de incorporação OpenAI usado para gerar representações vetoriais de consultas.
        vectordb_dir (str): O diretório onde o banco de dados vetorial Chroma é mantido no disco.
        k (int): O número das k histórias vizinhas mais próximas a serem recuperadas do banco de dados vetorial.
        vectordb (Chroma): A instância do banco de dados vetorial Chroma conectada à coleção especificada e ao modelo de incorporação.

    Métodos:
        __init__: inicializa a ferramenta com o modelo de incorporação, banco de dados vetorial e parâmetros de recuperação especificados.
    """

    def __init__(self, embedding_model: str, vectordb_dir: str, k: int, collection_name: str) -> None:
        """
        Inicializa o StoriesRAGTool com as configurações necessárias.

        Argumentos:
            embedding_model (str): O nome do modelo de incorporação (por exemplo, "text-embedding-ada-002")
                usado para converter consultas em representações vetoriais.
            vectordb_dir (str): O caminho do diretório onde o banco de dados vetorial Chroma é armazenado e persistido no disco.
            k (int): O número de histórias vizinhas mais próximas a serem recuperadas com base na similaridade da consulta.
            nome_coleção (str): O nome da coleção dentro do banco de dados vetorial que contém as histórias relevantes.
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
def lookup_stories(query: str) -> str:
    """Pesquise entre as histórias de ficção e encontre a resposta para a pergunta. A entrada deve ser a consulta."""
    rag_tool = StoriesRAGTool(
        embedding_model=TOOLS_CFG.stories_rag_embedding_model,
        vectordb_dir=TOOLS_CFG.stories_rag_vectordb_directory,
        k=TOOLS_CFG.stories_rag_k,
        collection_name=TOOLS_CFG.stories_rag_collection_name)
    docs = rag_tool.vectordb.similarity_search(query, k=rag_tool.k)
    return "\n\n".join([doc.page_content for doc in docs])
