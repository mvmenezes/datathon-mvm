"""Agente ReAct para análise e previsão de valores de ações.

Referência: Yao et al. (2023) — ReAct: Synergizing Reasoning and Acting
            in Language Models. https://arxiv.org/abs/2210.03629
"""
import logging
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from src.agent.tools import download_stock_data, feature_engineering_tool, get_stock_history, calculate_technical_indicators, get_stock_news, list_trained_models, predict_stock_price, train_stock_model
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()
model_name: str = "gpt-4o-mini"
temperature: float = 0.0
# ---------------------------------------------------------------------------
# CONSTRUÇÃO DO AGENTE
# ---------------------------------------------------------------------------

tools = [get_stock_history, calculate_technical_indicators, get_stock_news, predict_stock_price, train_stock_model, download_stock_data, feature_engineering_tool, list_trained_models]
tool_names = [tool.name for tool in tools]
input = ""
agent_scratchpad = ""

# O template ReAct exige exatamente estas variáveis para create_react_agent:
# {tools}, {tool_names}, {input}, {agent_scratchpad}
SYSTEM_PROMPT = (
    f"""Você é um analista financeiro especializado em ações da bolsa de valores.
Use as ferramentas disponíveis para fazer as seguintes tarefas 
- Sempre que o usuário perguntar sobre uma ação, use a ferramenta para listar os modelos treinados para verificar se já existe um modelo treinado para essa ação. Se não existir, use a ferramenta de download para baixar os dados históricos e depois use a ferramenta de preparação das features de treinamento para preparar os dados para o treinamento do modelo. Depois disso, use a ferramenta de treinamento para treinar um modelo LSTM para essa ação.
- Sempre forneça a previsão da ação para os próximos dias, nunca forneça a previsão sem usar a ferramenta de previsão.
- Sempre use  as outras ferramentas para verificar se a previsão faz sentido ou não, com base em notícias e indicadores técnicos.
- Adicione uma probabilidade numérica de acerto para a previsão.
- Não utilize mais de 300 palavras para responder, seja direto e objetivo.
- Perguntas que não sejam possíveis de serem respondidas com as ferramentas disponíveis devem ser respondidas com "Desculpe, não tenho informações suficientes para responder a essa pergunta."

Ferramentas disponíveis:
{tools}

Nomes das ferramentas: {tool_names}

Use SEMPRE o formato:
Thought: pensar sobre o que fazer
Action: nome_da_ferramenta
Action Input: input para a ferramenta
Observation: resultado da ferramenta
... (repita quantas vezes necessário)
Thought: Agora tenho dados suficientes para responder
Previsão: Previsão do preço da ação para os próximos dias.
Notícias: Breve informativo sobre as notícias mais recentes relacionadas à ação.
Indicadores Técnicos: Breve resumo dos indicadores técnicos mais recentes para a ação.
Probabilidade de Acerto: Probabilidade numérica de acerto da previsão (ex: 70%).


Pergunta: {input}
{agent_scratchpad}"""
)







def create_stock_agent():
    """Cria agente ReAct para análise de ações.
    Returns:
        AgentExecutor configurado com 3 tools financeiras.
    """
    

    llm   = ChatOpenAI(model=model_name, temperature=temperature)
    model_with_tools = llm.bind_tools(tools)
    agent = create_agent(model_with_tools, tools=tools, system_prompt=SYSTEM_PROMPT)


    logger.info("Agente criado com %d tools", len(tools))
    return agent


def run_agent(user_input):
    """Executa o agente com a entrada do usuário.
    Args:
        user_input: String com a pergunta do usuário (ex: "Qual a previsão para PETR4.SA?").
    Returns:
        Resposta final do agente após usar as ferramentas.
    """
    agent = create_stock_agent()
    response = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
    return response["messages"][-1].content

# ---------------------------------------------------------------------------
# EXEMPLO DE USO
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    agent = create_stock_agent()
    response = agent.invoke({"messages": [{"role": "user", "content": "Qual a previsão para USIM5.SA?"}]})


    print("\n" + "=" * 50)
    print("RESPOSTA FINAL:")
    print(response["messages"][-1].content)