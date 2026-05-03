# Plano de Conformidade — Lei Geral de Proteção de Dados (LGPD)

> **Lei nº 13.709/2018 — Lei Geral de Proteção de Dados Pessoais**
> Versão: 1.0
> Escopo: Sistema de Agente Inteligente com LLM — Datathon Fase 05

---

## 1. Objetivo

Este documento estabelece as diretrizes de conformidade com a LGPD adotadas pelo sistema desenvolvido no Datathon Fase 05. O plano descreve as responsabilidades do sistema em relação ao tratamento de dados pessoais, as medidas técnicas implementadas para proteção do titular e as recomendações de uso seguro direcionadas aos usuários finais.

---

## 2. Recomendação Oficial ao Usuário

> ⚠️ **AVISO IMPORTANTE**
>
> **Não recomendamos o envio de informações pessoais, sensíveis ou confidenciais** nas interações com este sistema.
>
> Exemplos de dados que **não devem ser inseridos**:
>
> - CPF, RG, passaporte ou qualquer documento de identificação
> - Nome completo associado a outros dados identificadores
> - Endereço residencial ou comercial
> - Dados financeiros (número de conta, cartão, saldo, extratos)
> - Dados de saúde, origem racial, religião ou biometria
> - Senhas, tokens de acesso ou credenciais de qualquer natureza
> - Informações confidenciais de terceiros ou de organizações
>
> O sistema foi projetado para responder perguntas e auxiliar em tarefas analíticas. **A qualidade das respostas não depende do fornecimento de dados pessoais reais.**

---

## 3. Medidas Técnicas de Proteção Implementadas

Mesmo diante de envio inadvertido de dados pessoais pelo usuário, o sistema aplica camadas de proteção automáticas conforme descrito abaixo.

### 3.1 Guardrail de Saída — Sanitização de Respostas

Todas as respostas geradas pelo modelo passam por um pipeline de sanitização antes de serem entregues ao usuário. O sistema detecta e remove automaticamente dados pessoais do texto de saída do LLM.

**Entidades detectadas e tratadas:**

| Entidade | Ação | Exemplo |
|---|---|---|
| CPF / CNPJ | Substituição por `<CPF_REMOVIDO>` | `123.456.789-00` → `<CPF_REMOVIDO>` |
| Nome de pessoa | Substituição por `<PESSOA>` | `João Silva` → `<PESSOA>` |
| Endereço de e-mail | Substituição por `<EMAIL_REMOVIDO>` | `joao@email.com` → `<EMAIL_REMOVIDO>` |
| Número de telefone | Substituição por `<TELEFONE_REMOVIDO>` | `(11) 99999-9999` → `<TELEFONE_REMOVIDO>` |
| Dados financeiros detectáveis | Mascaramento parcial | `**** **** **** 1234` |

**Tecnologia utilizada:** Microsoft Presidio (motor de análise e anonimização de PII).

### 3.2 Guardrail de Entrada — Detecção de Padrões Sensíveis

O sistema aplica verificações no input do usuário para identificar e alertar sobre o envio de dados pessoais antes do processamento.

**Comportamento:**

- Detecção de CPF, e-mail, telefone e outros identificadores no input
- Alerta ao usuário sobre o envio inadvertido
- Registro de ocorrência em log de auditoria (sem armazenar o dado em si)

### 3.3 Não Retenção de Dados Pessoais

- O sistema **não armazena** o conteúdo das conversas em base de dados persistente com associação a identidade do usuário
- Logs operacionais registram apenas metadados técnicos (timestamp, latência, status da requisição) — **nunca o conteúdo da mensagem**
- O vector store (base RAG) é alimentado exclusivamente com dados corporativos previamente anonimizados — não com inputs dos usuários

---

## 4. Base Legal para Tratamento Residual de Dados

Na hipótese de dados pessoais chegarem ao sistema, o tratamento é fundamentado nas seguintes bases legais previstas no Art. 7º da LGPD:

| Situação | Base Legal | Fundamentação |
|---|---|---|
| Logs técnicos de operação | Legítimo interesse (Art. 7º, IX) | Garantia de segurança e auditabilidade do sistema |
| Detecção de PII no input | Legítimo interesse (Art. 7º, IX) | Proteção do próprio titular |
| Dados de treinamento do modelo | Execução de contrato / Legítimo interesse | Dados anonimizados previamente ao uso |

> **Nota:** O sistema não coleta consentimento explícito porque a arquitetura foi desenhada para **não necessitar** do tratamento de dados pessoais. A base legal acima cobre apenas situações residuais e involuntárias.

---

## 5. Direitos do Titular

Conforme os Arts. 17 a 22 da LGPD, os titulares possuem os seguintes direitos, que o sistema respeita da forma descrita:

| Direito | Como é atendido |
|---|---|
| **Acesso** (Art. 18, II) | O sistema não armazena dados pessoais associados a usuários — não há perfil a ser consultado |
| **Correção** (Art. 18, III) | Não aplicável — dados não são retidos |
| **Eliminação** (Art. 18, VI) | Não aplicável — dados não são retidos |
| **Portabilidade** (Art. 18, V) | Não aplicável — dados não são retidos |
| **Informação** (Art. 18, VII) | Este documento e o System Card descrevem integralmente o tratamento realizado |
| **Revogação de consentimento** (Art. 18, IX) | O sistema não opera por base de consentimento — não aplicável |

---

## 6. Responsabilidades

| Papel | Responsabilidade |
|---|---|
| **Equipe de desenvolvimento** | Manter os guardrails de input/output ativos e atualizados; revisar o plano a cada ciclo de melhoria do sistema |
| **Operador do sistema** | Garantir que o ambiente de produção não armazene logs com conteúdo de mensagens; controlar o acesso ao ambiente |
| **Usuário final** | Seguir a recomendação de não enviar dados pessoais; reportar comportamentos inesperados do sistema |

---

## 7. Gestão de Incidentes

Em caso de incidente de segurança com potencial exposição de dados pessoais, o procedimento adotado é:

1. **Identificação** — detecção via monitoramento automatizado ou reporte manual
2. **Contenção** — isolamento do componente afetado (ver arquitetura de deploy isolado)
3. **Avaliação** — verificação do escopo: quais dados, quantos titulares, qual o risco
4. **Correção e documentação** — aplicação de patch, registro do incidente e lições aprendidas

---

## 8. Referências

- Brasil. *Lei nº 13.709, de 14 de agosto de 2018* — Lei Geral de Proteção de Dados Pessoais (LGPD).
- ANPD. *Guia Orientativo para Definições dos Agentes de Tratamento de Dados Pessoais*, 2021.
- Microsoft. *Presidio — Data Protection and Anonymization API*. https://microsoft.github.io/presidio/
- OWASP. *Top 10 for Large Language Model Applications*, 2025. https://owasp.org/www-project-top-10-for-large-language-model-applications/

---

*Documento mantido pela equipe do Datathon Fase 05. Revisão recomendada a cada nova versão do sistema.*