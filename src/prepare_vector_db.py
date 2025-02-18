import os
import yaml
from pyprojroot import here
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv


class PrepareVectorDB:
    """
    Uma classe para preparar e gerenciar um banco de dados vetorial (VectorDB) usando documentos de um diretório especificado.
    A classe executa as seguintes tarefas:
    - Carrega e divide documentos (PDFs).
    - Divide o texto em pedaços com base no tamanho e na sobreposição do pedaço especificado.
    - Incorpora os pedaços do documento usando um modelo de incorporação especificado.
    - Armazena os vetores incorporados em um diretório VectorDB persistente.

    Atributos:
        doc_dir (str): Caminho para o diretório que contém os documentos (PDFs) a serem processados.
        chunk_size (int): O tamanho máximo de cada pedaço (em caracteres) no qual o texto do documento será dividido.
        chunk_overlap (int): O número de caracteres sobrepostos entre pedaços consecutivos.
        embedding_model (str): O nome do modelo de incorporação a ser usado para gerar representações vetoriais de texto.
        vectordb_dir (str): Diretório onde será armazenado o banco de dados vetorial resultante.
        nome_coleção (str): O nome da coleção a ser usada no banco de dados vetorial.

    Métodos:
        path_maker(nome_do_arquivo: str, doc_dir: str) -> str:
            Cria um caminho de arquivo completo juntando o diretório e o nome de arquivo fornecidos.

        executar() -> Nenhum:
            Executa o processo de leitura de documentos, divisão de texto, incorporação em vetores e 
            salvando o banco de dados vetorial resultante. Se o diretório do banco de dados vetorial já existir, ele ignora
            o processo de criação.
    """

    def __init__(self,
                 doc_dir: str,
                 chunk_size: int,
                 chunk_overlap: int,
                 embedding_model: str,
                 vectordb_dir: str,
                 collection_name: str
                 ) -> None:

        self.doc_dir = doc_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.vectordb_dir = vectordb_dir
        self.collection_name = collection_name

    def path_maker(self, file_name: str, doc_dir):
        """
        Cria um caminho de arquivo completo juntando o diretório e o nome de arquivo fornecidos.

        Argumentos:
            file_name (str): Nome do arquivo.
            doc_dir (str): Caminho do diretório.

        Retorna:
            str: Caminho completo do arquivo.
        """
        return os.path.join(here(doc_dir), file_name)

    def run(self):
        """
        Executa a lógica principal para criar e armazenar embeddings de documentos em um VectorDB.

        Se o diretório do banco de dados vetorial não existir:
        - Carrega documentos PDF do `doc_dir`, divide-os em pedaços,
        - Incorpora os pedaços do documento usando o modelo de incorporação especificado,
        - Armazena os embeddings em um diretório VectorDB persistente.

        Se o diretório já existir, ele ignora o processo de criação da incorporação.

        Imprime o status de criação e o número de vetores no banco de dados de vetores.

        Retorna:
            Nenhum
        """
        if not os.path.exists(here(self.vectordb_dir)):
            # Se não existir, crie o diretório e crie os embeddings
            os.makedirs(here(self.vectordb_dir))
            print(f"Directory '{self.vectordb_dir}' was created.")

            file_list = os.listdir(here(self.doc_dir))
            docs = [PyPDFLoader(self.path_maker(
                fn, self.doc_dir)).load_and_split() for fn in file_list]
            docs_list = [item for sublist in docs for item in sublist]

            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )
            doc_splits = text_splitter.split_documents(docs_list)
            # Adicionar ao vetorDB
            vectordb = Chroma.from_documents(
                documents=doc_splits,
                collection_name=self.collection_name,
                embedding=OpenAIEmbeddings(model=self.embedding_model),
                persist_directory=str(here(self.vectordb_dir))
            )
            print("VectorDB is created and saved.")
            print("Number of vectors in vectordb:",
                  vectordb._collection.count(), "\n\n")
        else:
            print(f"Directory '{self.vectordb_dir}' already exists.")


if __name__ == "__main__":
    load_dotenv()
    os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

    with open(here("configs/tools_config.yml")) as cfg:
        app_config = yaml.load(cfg, Loader=yaml.FullLoader)

    # Remova o comentário das seguintes configurações para executar o documento de swiss airline policy
    chunk_size = app_config["swiss_airline_policy_rag"]["chunk_size"]
    chunk_overlap = app_config["swiss_airline_policy_rag"]["chunk_overlap"]
    embedding_model = app_config["swiss_airline_policy_rag"]["embedding_model"]
    vectordb_dir = app_config["swiss_airline_policy_rag"]["vectordb"]
    collection_name = app_config["swiss_airline_policy_rag"]["collection_name"]
    doc_dir = app_config["swiss_airline_policy_rag"]["unstructured_docs"]

    prepare_db_instance = PrepareVectorDB(
        doc_dir=doc_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
        vectordb_dir=vectordb_dir,
        collection_name=collection_name)

    prepare_db_instance.run()

    # Remova o comentário das seguintes configurações para executar o documento de stories
    chunk_size = app_config["stories_rag"]["chunk_size"]
    chunk_overlap = app_config["stories_rag"]["chunk_overlap"]
    embedding_model = app_config["stories_rag"]["embedding_model"]
    vectordb_dir = app_config["stories_rag"]["vectordb"]
    collection_name = app_config["stories_rag"]["collection_name"]
    doc_dir = app_config["stories_rag"]["unstructured_docs"]

    prepare_db_instance = PrepareVectorDB(
        doc_dir=doc_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
        vectordb_dir=vectordb_dir,
        collection_name=collection_name)

    prepare_db_instance.run()
