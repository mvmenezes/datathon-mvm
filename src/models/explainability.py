# src/models/explainability.py
import shap
import pandas as pd
import matplotlib.pyplot as plt
import mlflow

def explain_model(model, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                  model_name: str) -> dict:
    """
    Gera explicações globais e locais via SHAP.
    Loga artefatos no MLflow.
    """
    explainer = shap.TreeExplainer(model)  # ou shap.Explainer para modelos genéricos
    shap_values = explainer(X_test)

    # --- Explicação Global ---
    # Importância média de cada feature no modelo inteiro
    fig_global, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title(f"Importância Global das Features — {model_name}")
    plt.tight_layout()

    # --- Explicação Local (primeira predição como exemplo) ---
    fig_local, ax = plt.subplots(figsize=(10, 4))
    shap.waterfall_plot(shap_values[0], show=False)
    plt.title("Explicação Local — Amostra #0")
    plt.tight_layout()

    # Loga no MLflow como artefatos
    with mlflow.start_run(run_name=f"{model_name}_explainability"):
        mlflow.log_figure(fig_global, "shap_global_importance.png")
        mlflow.log_figure(fig_local,  "shap_local_sample0.png")

        # Feature importance como métrica também
        mean_abs_shap = pd.Series(
            abs(shap_values.values).mean(axis=0),
            index=X_test.columns
        ).sort_values(ascending=False)

        for feature, importance in mean_abs_shap.items():
            mlflow.log_metric(f"shap_{feature}", round(importance, 6))

    return {"mean_abs_shap": mean_abs_shap.to_dict()}



# Para o agente/RAG — explicabilidade é rastreabilidade
def explain_rag_response(query: str, answer: str, 
                          contexts: list[str]) -> dict:
    """
    Para RAG, 'explicar' significa mostrar de onde veio a resposta.
    """
    return {
        "query": query,
        "answer": answer,
        "source_documents": contexts,        # quais chunks embasaram a resposta
        "num_sources_used": len(contexts),
        "faithfulness_score": ...,           # calculado via RAGAS
        # Isso vai direto para o System Card como evidência de explicabilidade
    }