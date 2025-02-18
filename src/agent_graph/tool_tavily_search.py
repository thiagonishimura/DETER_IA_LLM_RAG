from langchain_community.tools.tavily_search import TavilySearchResults


def load_tavily_search_tool(tavily_search_max_results: int):
    """
    Esta função inicializa uma ferramenta de pesquisa Tavily, que realiza pesquisas e retorna resultados
    com base nas consultas do usuário. O parâmetro `max_results` controla quantos resultados de pesquisa são
    recuperado para cada consulta.

    Argumentos:
        tavily_search_max_results (int): O número máximo de resultados de pesquisa a serem retornados para cada consulta.

    Retorna:
        TavilySearchResults: uma instância configurada da ferramenta de pesquisa Tavily com os `max_results` especificados.
    """
    return TavilySearchResults(max_results=tavily_search_max_results)
