    


# 2. Cria um registry isolado (boa prática em jobs batch)
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

def push_metrics_to_gateway(scores: dict, job_name: str, pushgateway_url: str = "localhost:9091"):
    registry = CollectorRegistry()

    metrics = []

    #Seta os valores com labels
    labels = {"model": "gpt-4o-mini", "dataset_version": "v1.0"}
    for chave, valor in scores.items():
        metrics.append(Gauge(chave, chave, ["model", "dataset_version"], registry=registry))
        metrics[-1].labels(**labels).set(scores[chave])


    # 5. Push pro gateway
    push_to_gateway(
        pushgateway_url,
        job=job_name,
        registry=registry,
    )

    print(f"Métricas enviadas: {dict(scores)}")
    return scores