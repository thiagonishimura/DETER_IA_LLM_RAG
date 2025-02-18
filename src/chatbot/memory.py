import os
import pandas as pd
from typing import List
from datetime import datetime, date


class Memory:
    """
    Uma classe para lidar com o armazenamento do histórico de conversas do chatbot gravando logs de chat em um arquivo CSV.

    Métodos:
        write_chat_history_to_file(gradio_chatbot: Lista, thread_id: str, folder_path: str) -> Nenhum:
            Grava a interação mais recente do chatbot (consulta do usuário e resposta do bot) em um arquivo CSV. 
            O registro do bate-papo é salvo com a data atual como nome do arquivo e a interação é 
            carimbo de data/hora.
    """
    @staticmethod
    def write_chat_history_to_file(gradio_chatbot: List,  thread_id: str, folder_path: str) -> None:
        """
        Grava a interação mais recente do chatbot (consulta e resposta do usuário) em um arquivo CSV. O registro inclui
        o ID do thread e o carimbo de data/hora da interação. O arquivo de cada dia é salvo com a data atual como nome do arquivo.

        Argumentos:
            gradio_chatbot (Lista): Uma lista contendo tuplas de consultas de usuários e respostas de chatbot. 
                                   A interação mais recente é anexada ao log.
            thread_id (str): O identificador exclusivo da sessão de chat (ou thread).
            folder_path (str): O caminho do diretório onde os arquivos CSV de log de bate-papo devem ser armazenados.

        Retorna:
            Nenhum

        Estrutura do arquivo:
            - O registro do bate-papo de cada dia é salvo como um arquivo CSV separado na pasta especificada.
            - O arquivo CSV é nomeado usando a data atual no formato 'AAAA-MM-DD'.
            - Cada linha no arquivo CSV contém as seguintes colunas: 'thread_id', 'timestamp', 'user_query', 'response'.
        """
        tmp_list = list(gradio_chatbot[-1])  # Converta a tupla em uma lista

        today_str = date.today().strftime('%Y-%m-%d')
        tmp_list.insert(0, thread_id)  # Adiciona um novo valor à lista

        current_time_str = datetime.now().strftime('%H:%M:%S')
        tmp_list.insert(1, current_time_str)  # Adiciona um novo valor à lista

        # Caminho do arquivo CSV de hoje
        file_path = os.path.join(folder_path, f'{today_str}.csv')

        # Crie um DataFrame da lista
        new_df = pd.DataFrame([tmp_list], columns=[
                              "thread_id", "timestamp", "user_query", "response"])

        # Verifique se o arquivo de hoje existe
        if os.path.exists(file_path):
            # Se existir, anexe os novos dados ao arquivo CSV
            new_df.to_csv(file_path, mode='a', header=False, index=False)
        else:
            # Caso não exista, crie o arquivo CSV com os novos dados
            new_df.to_csv(file_path, mode='w', header=True, index=False)
