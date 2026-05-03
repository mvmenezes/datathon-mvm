"""Avaliação do pipeline RAG com RAGAS — 4 métricas obrigatórias."""
import json
import logging
import asyncio
from openai import AsyncOpenAI
from ragas.embeddings.base import embedding_factory
import mlflow
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)
from src.agent.rag_pipeline import query_rag_with_context, build_rag_chain, get_or_create_vector_store

logger = logging.getLogger(__name__)


async def run_ragas_evaluation(
    golden_set_path: str,
    retriever,
    rag_chain,
) -> dict[str, float]:
    """Avalia o pipeline RAG contra o golden set."""

    openai_client = AsyncOpenAI()

    llm = llm_factory(
        "gpt-4o-mini",
        client=openai_client,
    )

   
    embeddings = embedding_factory(
        "openai",
        model="text-embedding-3-small",
        client=openai_client,
    )

    # 1. Carrega o golden set
    with open(golden_set_path) as f:
        golden_set = json.load(f)

    # 2. Roda cada pergunta pelo RAG e coleta resultados
    scores = {
        "faithfulness": [],
        "answer_relevancy": [],
        "context_precision": [],
        "context_recall": []
    }
    for i, item in enumerate(golden_set): 

        answer, contexts = query_rag_with_context(
            retriever=retriever,
            rag_chain=rag_chain,
            question=item["query"],
        )

        scorer = Faithfulness(llm=llm)
        scores["faithfulness"].append(await scorer.ascore(
                    user_input=item["query"],
                    response=answer,
                    retrieved_contexts=contexts))
        
        scorer = AnswerRelevancy(llm=llm,embeddings=embeddings)
        scores["answer_relevancy"].append(await scorer.ascore(
                    user_input=item["query"],
                    response=answer))
        
        scorer = ContextPrecision(llm=llm)        
        scores["context_precision"].append(await scorer.ascore(
                    user_input=item["query"],
                    retrieved_contexts=contexts,
                    reference=item["expected_answer"]))
        
        scorer = ContextRecall(llm=llm)
        scores["context_recall"].append(await scorer.ascore(
                    user_input=item["query"],
                    reference=item["expected_answer"],
                    retrieved_contexts=contexts))
        

    logger.info("Avaliado: %s", item["query"][:60])


    metric_faithfulness = sum(scores["faithfulness"])/len(scores["faithfulness"]) 
    metric_answer_relevancy = sum(scores["answer_relevancy"])/len(scores["answer_relevancy"])
    metric_context_precision = sum(scores["context_precision"])/len(scores["context_precision"]) 
    metric_context_recall = sum(scores["context_recall"])/len(scores["context_recall"]) 

    logger.info("RAGAS scores: %s", scores)

    metrics = {
        "ragas/faithfulness":       float(metric_faithfulness),
        "ragas/answer_relevancy":   float(metric_answer_relevancy),
        "ragas/context_precision":  float(metric_context_precision),
        "ragas/context_recall":     float(metric_context_recall),
    }
    print(metrics)
    # 5. Loga no MLflow
    with mlflow.start_run(run_name="ragas_evaluation"):
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(golden_set_path)
        mlflow.set_tag("evaluation_type", "ragas")
        mlflow.set_tag("golden_set_size", len(golden_set))

    logger.info("RAGAS concluído: %s", metrics)
    return metrics


if __name__ == "__main__":
    vector_store = get_or_create_vector_store("PETR4.SA")
    retriever, chain = build_rag_chain(vector_store=vector_store)

    asyncio.run(run_ragas_evaluation(
        "data/golden_set_curto.json",
        retriever,
        chain,
    ))