# system_card.py
"""
Módulo para gerar o System Card do projeto datathon-grupo-mvm.
O System Card documenta o sistema de IA, incluindo propósito, dados, modelos, performance, limitações e considerações éticas.
"""

import json
from pathlib import Path

# Definição do System Card como dicionário
system_card = {
    "name": "datathon-grupo-mvm",
    "version": "1.0.0",
    "description": "Sistema de predição de preços de ações utilizando modelos LSTM e um agente RAG para análise e explicabilidade.",
    "intended_use": {
        "primary": "Predizer preços futuros de ações para as empresas PETR4.SA (Petrobras), USIM5.SA (Usiminas) e VALE3.SA (Vale).",
        "users": "Analistas financeiros, investidores e pesquisadores interessados em previsões de mercado de ações.",
        "use_cases": [
            "Análise de tendências de preços",
            "Suporte à decisão de investimento",
            "Avaliação de risco baseada em dados históricos"
        ]
    },
    "data": {
        "sources": [
            "Dados históricos de preços de ações obtidos via yfinance (Yahoo Finance)",
            "Dados processados e armazenados em data/processed/"
        ],
        "preprocessing": "Feature engineering aplicado, incluindo indicadores técnicos e normalização.",
        "privacy": "Dados públicos de mercado financeiro. Cumpre regulamentações de privacidade (LGPD)."
    },
    "models": {
        "lstm_simple": "Modelo LSTM básico para predição de séries temporais.",
        "lstm_complex": "Modelo LSTM avançado com mais camadas para melhor captura de padrões.",
        "agent_rag": "Agente baseado em LangChain com RAG (Retrieval-Augmented Generation) para respostas contextuais e explicáveis."
    },
    "performance": {
        "metrics": "Avaliado com MSE, MAE e R². Resultados logados no MLflow.",
        "evaluation": "Usa ferramentas como Ragas para avaliação de RAG e Evidently para detecção de drift.",
        "benchmarks": "Comparado com baselines simples (média móvel)."
    },
    "limitations": {
        "data": "Baseado em dados históricos; não garante performance futura devido à volatilidade do mercado.",
        "scope": "Limitado a ações brasileiras específicas; não generaliza para outros ativos.",
        "bias": "Possível viés de dados históricos; não considera eventos externos não modelados."
    },
    "ethical_considerations": {
        "fairness": "Dados públicos e acessíveis; sem discriminação baseada em grupos protegidos.",
        "transparency": "Explicabilidade via SHAP para modelos e rastreabilidade de fontes para RAG.",
        "accountability": "Logs de auditoria via MLflow e monitoramento contínuo.",
        "privacy": "Dados não pessoais; conformidade com LGPD."
    },
    "monitoring": {
        "tools": "Prometheus para métricas, Evidently para drift detection.",
        "alerts": "Monitoramento de performance e detecção de anomalias."
    },
    "security": {
        "measures": "Autenticação via JWT, rate limiting, PII detection com Presidio.",
        "compliance": "OWASP guidelines seguidas."
    },
    "deployment": {
        "platform": "FastAPI para serving, Docker para containerização.",
        "orchestration": "Prefect para workflows, DVC para versionamento de dados."
    }
}

def generate_markdown() -> str:
    """
    Gera o conteúdo do System Card em formato Markdown.
    """
    md = f"# System Card: {system_card['name']}\n\n"
    md += f"**Versão:** {system_card['version']}\n\n"
    md += f"## Descrição\n{system_card['description']}\n\n"
    
    md += "## Uso Pretendido\n"
    intended = system_card['intended_use']
    md += f"**Primário:** {intended['primary']}\n\n"
    md += f"**Usuários:** {intended['users']}\n\n"
    md += "**Casos de Uso:**\n"
    for use in intended['use_cases']:
        md += f"- {use}\n"
    md += "\n"
    
    md += "## Dados\n"
    data = system_card['data']
    md += "**Fontes:**\n"
    for source in data['sources']:
        md += f"- {source}\n"
    md += f"\n**Pré-processamento:** {data['preprocessing']}\n\n"
    md += f"**Privacidade:** {data['privacy']}\n\n"
    
    md += "## Modelos\n"
    models = system_card['models']
    for key, desc in models.items():
        md += f"- **{key}:** {desc}\n"
    md += "\n"
    
    md += "## Performance\n"
    perf = system_card['performance']
    md += f"**Métricas:** {perf['metrics']}\n\n"
    md += f"**Avaliação:** {perf['evaluation']}\n\n"
    md += f"**Benchmarks:** {perf['benchmarks']}\n\n"
    
    md += "## Limitações\n"
    lim = system_card['limitations']
    for key, desc in lim.items():
        md += f"- **{key}:** {desc}\n"
    md += "\n"
    
    md += "## Considerações Éticas\n"
    eth = system_card['ethical_considerations']
    for key, desc in eth.items():
        md += f"- **{key}:** {desc}\n"
    md += "\n"
    
    md += "## Monitoramento\n"
    mon = system_card['monitoring']
    md += f"**Ferramentas:** {mon['tools']}\n\n"
    md += f"**Alertas:** {mon['alerts']}\n\n"
    
    md += "## Segurança\n"
    sec = system_card['security']
    md += f"**Medidas:** {sec['measures']}\n\n"
    md += f"**Conformidade:** {sec['compliance']}\n\n"
    
    md += "## Implantação\n"
    dep = system_card['deployment']
    md += f"**Plataforma:** {dep['platform']}\n\n"
    md += f"**Orquestração:** {dep['orchestration']}\n\n"
    
    return md

def save_to_file(filepath: str = "docs/SYSTEM_CARD.md"):
    """
    Salva o System Card em formato Markdown no arquivo especificado.
    """
    md_content = generate_markdown()
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"System Card salvo em {filepath}")

if __name__ == "__main__":
    # Imprime o Markdown no console
    print(generate_markdown())
    
    # Opcional: salva no arquivo
    save_to_file()