# 🛡️ Relatório de Testes de Segurança – Cenários Adversariais (OWASP)

## 📌 Visão Geral

Este documento apresenta a execução de **cenários adversariais testados e documentados**, com base nas diretrizes do OWASP Top 10, aplicados a uma API de predição de ações desenvolvida em FastAPI.

O objetivo é validar a robustez do sistema contra ataques comuns, garantindo segurança, confiabilidade e conformidade com boas práticas.

---

## 🧠 Arquitetura do Sistema

A aplicação é composta por:

- API REST construída com FastAPI
- Autenticação baseada em JWT
- Validação de entrada com Pydantic
- Proteção contra dados sensíveis (PII)
- Guardrails de input e output
- Modelo de predição de ações (ML/LLM)

---

## 🎯 Objetivos dos Testes

- Validar controle de acesso
- Prevenir ataques de injeção
- Garantir proteção de dados sensíveis
- Verificar validação de entradas
- Evitar comportamento incorreto do modelo (alucinação)

---

## 🧪 Metodologia

Os testes foram realizados simulando ataques reais através de:

- Requisições HTTP (cURL)
- Inputs maliciosos
- Casos extremos
- Validação de comportamento esperado

Cada cenário inclui:
- Descrição do ataque
- Procedimento de teste
- Resultado esperado
- Evidência do comportamento

---

# 🔟 Cenários Adversariais Testados

---

## 1. 🔒 Broken Access Control

**Descrição:**  
Tentativa de acesso a endpoint protegido sem autenticação.

**Teste:**
```bash
curl -X GET http://127.0.0.1:8000/predicao


{
  "detail": "Not authenticated"
}

