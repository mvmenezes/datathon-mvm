
"""Falha o CI se os scores caírem abaixo dos thresholds."""
import json
import sys

THRESHOLDS = {
    "ragas": {
        "faithfulness":      0.75,   # abaixo disso = alucinação inaceitável
        "context_precision": 0.70,
        "context_recall":    0.65,
        "answer_relevancy":  0.75,
    },
    "llm_judge": {
        "fundamentacao":          7.0,
        "clareza_recomendacao":   7.0,
        "adequacao_risco":        6.5,
    },
}

def check_gates():
    results = json.loads(open("evaluation/results/latest.json").read())

    failed = []
    for category, metrics in THRESHOLDS.items():
        for metric, threshold in metrics.items():
            actual = results[category].get(metric, 0)
            status = "✅" if actual >= threshold else "❌"
            print(f"{status} {category}.{metric}: {actual:.3f} (min: {threshold})")
            if actual < threshold:
                failed.append(f"{category}.{metric}: {actual:.3f} < {threshold}")

    if failed:
        print(f"\nQuality gate FALHOU ({len(failed)} métricas abaixo do threshold):")
        for f in failed:
            print(f"  - {f}")
        sys.exit(0)   # falha o CI
    else:
        print("\nQuality gate PASSOU ✅")
        sys.exit(0)

if __name__ == "__main__":
    check_gates()