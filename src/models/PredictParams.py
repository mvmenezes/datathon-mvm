"""
Módulo de definição de parâmetros para predição de preços de ações.

Utiliza Pydantic para validação automática de tipos e documentação de entrada.
"""

from pydantic import BaseModel


class PredictParams(BaseModel):
    """
    Esquema de validação para parâmetros de predição.
    
    Utiliza Pydantic BaseModel para validação automática de tipos e geração
    de documentação OpenAPI nos endpoints FastAPI.
    
    Attributes:
        stock (str): Símbolo da ação na B3 (ex: 'VALE3.SA', 'PETR4.SA')
            - Deve corresponder a um modelo previamente treinado
            - Se não houver modelo treinado, levantará ModelNotTrainedException
        days (int): Número de dias de histórico para usar na predição (ex: 30, 60, 90)
            - Define quanto de histórico recente será baixado
            - Maior valor = mais dados para contexto
            - Menor valor = apenas tendência recente
        model_type (str): Tipo de modelo a utilizar ('simple' ou 'complex')
            - Deve corresponder ao tipo usado no treinamento
            - Se não corresponder, o modelo não será encontrado
    
    Example:
        ```python
        params = PredictParamas(
            stock="VALE3.SA",
            days=30,
            model_type="simple"
        )
        ```
    """
    stock: str
    days: int
    model_type: str