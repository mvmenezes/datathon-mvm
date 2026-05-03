"""Agente ReAct para análise e previsão de valores de ações.

Referência: Yao et al. (2023) — ReAct: Synergizing Reasoning and Acting
            in Language Models. https://arxiv.org/abs/2210.03629
"""
import logging
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from src.security.guardrails import InputGuardrail, OutputGuardrail
from src.agent.tools import download_stock_data, feature_engineering_tool, get_stock_history, get_stock_news, list_trained_models, predict_stock_price, train_stock_model
from dotenv import load_dotenv
from datetime import datetime
logger = logging.getLogger(__name__)
load_dotenv()
model_name: str = "gpt-4o-mini"
temperature: float = 0.0
# ---------------------------------------------------------------------------
# CONSTRUÇÃO DO AGENTE
# ---------------------------------------------------------------------------

tools = [get_stock_history, get_stock_news, predict_stock_price, train_stock_model, download_stock_data, feature_engineering_tool, list_trained_models]
tool_names = [tool.name for tool in tools]
input = ""
agent_scratchpad = ""

# O template ReAct exige exatamente estas variáveis para create_react_agent:
# {tools}, {tool_names}, {input}, {agent_scratchpad}
hoje = datetime.today().strftime("%Y-%m-%d")
SYSTEM_PROMPT = (
    f"""Você é um analista financeiro especializado em ações da bolsa de valores. Saiba que hoje é {hoje}. Use as ferramentas disponíveis para responder perguntas sobre ações, como previsões de preços, notícias recentes e indicadores técnicos. Siga a metodologia ReAct para pensar passo a passo, agir usando as ferramentas e observar os resultados antes de chegar a uma conclusão.
Use as ferramentas disponíveis para fazer as seguintes tarefas 
- Não utilize mais de 300 palavras para responder, seja direto e objetivo.
- Não invente respostas, Perguntas que não sejam possíveis de serem respondidas com as ferramentas disponíveis devem ser respondidas com "Desculpe, não tenho informações suficientes para responder a essa pergunta."

Ferramentas disponíveis:
{tools}

Nomes das ferramentas: {tool_names}




Pergunta: {input}
{agent_scratchpad}"""
)







def create_stock_agent():
    """Cria agente ReAct para análise de ações.
    Returns:
        AgentExecutor configurado com 3 tools financeiras.
    """
    

    llm   = ChatOpenAI(model=model_name, temperature=temperature)

    logger.info("RAG pipeline carregado")
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
    validated, reason = InputGuardrail().validate(user_input)
    if not validated:
        logger.warning("Input inválido: %s", reason)
        return "Input inválido.",[]
    agent = create_stock_agent()
    
    response = agent.invoke({"messages": [{"role": "user", "content": user_input}]})

    print(response["messages"][-1].content)
    answer_sanitized = OutputGuardrail().sanitize(response["messages"][-1].content)
    return answer_sanitized

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