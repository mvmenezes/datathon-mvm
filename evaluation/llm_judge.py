# evaluation/llm_judge.py

import json
import os
import logging
from google import generativeai as genai
from dotenv import load_dotenv

from src.agent.rag_pipeline import run_pipeline_llm_judge

load_dotenv()
logger = logging.getLogger(__name__)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


JUDGE_PROMPT = """Você é um avaliador especialista em sistemas de IA para o mercado financeiro.

Avalie a resposta abaixo segundo os critérios indicados.
Para cada critério, atribua uma nota de 0 a 10 e justifique em 1-2 frases.

---
PERGUNTA DO USUÁRIO:
{question}

CONTEXTOS RECUPERADOS:
{contexts}

RESPOSTA DO SISTEMA:
{answer}

RESPOSTA ESPERADA (ground truth):
{ground_truth}
---

Avalie os seguintes critérios:

1. FIDELIDADE FACTUAL: A resposta se baseia apenas nos contextos fornecidos, sem alucinar informações?

2. CLAREZA E COMPLETUDE: A resposta é clara, bem estruturada e cobre os aspectos essenciais da pergunta?

3. ADEQUAÇÃO AO NEGÓCIO: A resposta seria útil para um profissional do mercado financeiro tomar uma decisão? Usa terminologia correta do domínio?

Responda APENAS em JSON, no formato:
{{
  "fidelidade_factual": {{"nota": 0-10, "justificativa": "..."}},
  "clareza_completude": {{"nota": 0-10, "justificativa": "..."}},
  "adequacao_negocio": {{"nota": 0-10, "justificativa": "..."}},
  "nota_geral": 0-10}}
"""

def llm_judge(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str,
) -> dict:
    prompt = JUDGE_PROMPT.format(
        question=question,
        answer=answer,
        contexts="\n\n".join(contexts),
        ground_truth=ground_truth,
    )
    model = genai.GenerativeModel(
            model_name="gemini-2.5-flash-lite",
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,       
                response_mime_type="application/json",  # força JSON direto
            ),
        )
    response = model.generate_content(prompt)

    try:
        return json.loads(response.text)
    except json.JSONDecodeError as e:
        logger.error("Falha ao parsear JSON do Gemini: %s\nResposta: %s", e, response.text)
        raise


def evaluate_golden_set(golden_set_path: str, rag_fn) -> dict:
    """Roda o judge em todo o golden set e agrega os resultados."""
    with open(golden_set_path, encoding="utf-8") as f:
        golden_set = json.load(f)

    all_scores = []

    for item in golden_set:
        answer, contexts = rag_fn(item["query"])
        scores = llm_judge(
            question=item["query"],
            answer=answer,
            contexts=contexts,
            ground_truth=item["expected_answer"],
        )
        all_scores.append(scores)

    # Agrega médias por critério
    criterios = ["fidelidade_factual", "clareza_completude", "adequacao_negocio"]
    summary = {}
    for criterio in criterios:
        notas = [s[criterio]["nota"] for s in all_scores]
        summary[criterio] = round(sum(notas) / len(notas), 2)

    summary["nota_geral_media"] = round(
        sum(s["nota_geral"] for s in all_scores) / len(all_scores), 2
    )

    return summary


def run_llm_judge():
    scores = evaluate_golden_set("data/golden_set_curto.json", run_pipeline_llm_judge)
    os.makedirs("evaluation/results", exist_ok=True)
    
    print(scores)
    # lê o que já existe
    if os.path.exists("evaluation/results/latest.json"):
        with open("evaluation/results/latest.json", "r", encoding="utf-8") as f:
            current = json.load(f)
    else:
        current = {}
    score_tratado =    { "llm_judge": {
        "fundamentacao":          scores["fidelidade_factual"],
        "clareza_recomendacao":   scores["clareza_completude"],
        "adequacao_risco":        scores["adequacao_negocio"],
        "nota_geral_media":        scores["nota_geral_media"],
    }}
    # merge (nível de categoria)
    for category, metrics in score_tratado.items():
        if category not in current:
            current[category] = {}

        current[category].update(metrics)

    # escreve de volta
    with open("evaluation/results/latest.json", "w", encoding="utf-8") as f:
        json.dump(current, f, indent=4)
    return scores
if __name__ == "__main__":
    run_llm_judge()