# System Card: datathon-grupo-mvm

**Versão:** 1.0.0

## Descrição
Sistema de predição de preços de ações utilizando modelos LSTM e um agente RAG para análise e explicabilidade.

## Uso Pretendido
**Primário:** Predizer preços futuros de ações para empresas dependentes de dólar PETR4.SA (Petrobras), USIM5.SA (Usiminas) e VALE3.SA (Vale).

**Usuários:** Analistas financeiros, investidores e pesquisadores interessados em previsões de mercado de ações.

**Casos de Uso:**
- Análise de tendências de preços
- Suporte à decisão de investimento
- Avaliação de risco baseada em dados históricos

## Dados
**Fontes:**
- Dados históricos de preços de ações obtidos via yfinance (Yahoo Finance)
- Dados processados e armazenados em data/processed/

**Pré-processamento:** Feature engineering aplicado, incluindo indicadores técnicos e normalização.

**Privacidade:** Dados públicos de mercado financeiro. Cumpre regulamentações de privacidade (LGPD).

## Modelos
- **lstm_simple:** Modelo LSTM básico para predição de séries temporais.
- **lstm_complex:** Modelo LSTM avançado com mais camadas para melhor captura de padrões.
- **agent_rag:** Agente baseado em LangChain com RAG (Retrieval-Augmented Generation) para respostas contextuais e explicáveis.

## Performance
**Métricas:** Avaliado com MSE, MAE e R². Resultados logados no MLflow.

**Avaliação:** Usa ferramentas como Ragas para avaliação de RAG e Evidently para detecção de drift.

**Benchmarks:** Comparado com baselines simples (média móvel).

## Limitações
- **data:** Baseado em dados históricos; não garante performance futura devido à volatilidade do mercado.
- **scope:** Limitado a ações brasileiras específicas; não generaliza para outros ativos.
- **bias:** Possível viés de dados históricos; não considera eventos externos não modelados.

## Considerações Éticas
- **fairness:** Dados públicos e acessíveis; sem discriminação baseada em grupos protegidos.
- **transparency:** Explicabilidade via SHAP para modelos e rastreabilidade de fontes para RAG.
- **accountability:** Logs de auditoria via MLflow e monitoramento contínuo.
- **privacy:** Dados não pessoais; conformidade com LGPD.

## Monitoramento
**Ferramentas:** Prometheus para métricas, Evidently para drift detection.

**Alertas:** Monitoramento de performance e detecção de anomalias.

## Segurança
**Medidas:** Autenticação via JWT, rate limiting, PII detection com Presidio.

**Conformidade:** OWASP guidelines seguidas.

## Implantação
**Plataforma:** FastAPI para serving, Docker para containerização.

**Orquestração:** Prefect para workflows, DVC para versionamento de dados.

