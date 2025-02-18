from typing import List, Tuple
from chatbot.load_config import LoadProjectConfig
from agent_graph.load_tools_config import LoadToolsConfig
from agent_graph.build_full_graph import build_graph
from utils.app_utils import create_directory
from chatbot.memory import Memory

URL = "https://github.com/Farzad-R/LLM-Zero-to-Hundred/tree/master/RAG-GPT"
hyperlink = f"[RAG-GPT user guideline]({URL})"

PROJECT_CFG = LoadProjectConfig()
TOOLS_CFG = LoadToolsConfig()

graph = build_graph()
config = {"configurable": {"thread_id": TOOLS_CFG.thread_id}}

create_directory("memory")


class ChatBot:
    """
    Uma classe para lidar com interações de chatbot utilizando um gráfico de agente predefinido. Os processos do chatbot
    mensagens do usuário, gera respostas apropriadas e salva o histórico de bate-papo em um diretório de memória especificado.

    Atributos:
        config (dict): Um dicionário de configuração que armazena configurações específicas, como `thread_id`.

    Métodos:
        responder(chatbot: Lista, mensagem: str) -> Tupla:
            Processa a mensagem do usuário por meio do gráfico do agente, gera uma resposta, anexa-a ao histórico do chat,
            e grava o histórico do bate-papo em um arquivo.
    """
    @staticmethod
    def respond(chatbot: List, message: str) -> Tuple:
        """
        Processa uma mensagem do usuário usando o gráfico do agente, gera uma resposta e a anexa ao histórico de chat.
        O histórico do bate-papo também é salvo em um arquivo de memória para referência futura.

        Argumentos:
            chatbot (lista): uma lista que representa o histórico de conversas do chatbot. Cada entrada é uma tupla da mensagem do usuário e da resposta do bot.
            mensagem (str): A mensagem do usuário a ser processada.

        Retorna:
            Tupla: Retorna uma string vazia (representando o novo espaço reservado de entrada do usuário) e o histórico de conversa atualizado.
        """
        # A configuração é o **segundo argumento posicional** para stream() ou invocar()!
        events = graph.stream(
            {"messages": [("user", message)]}, config, stream_mode="values"
        )
        for event in events:
            event["messages"][-1].pretty_print()

        chatbot.append(
            (message, event["messages"][-1].content))

        Memory.write_chat_history_to_file(
            gradio_chatbot=chatbot, folder_path=PROJECT_CFG.memory_dir, thread_id=TOOLS_CFG.thread_id)
        return "", chatbot
