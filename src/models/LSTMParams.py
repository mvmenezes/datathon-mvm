"""
Módulo de definição de parâmetros para treinamento de modelos LSTM.

Utiliza Pydantic para validação automática de tipos e documentação de entrada.
"""

from pydantic import BaseModel


class LSTMParams(BaseModel):
    """
    Esquema de validação para parâmetros de treinamento do modelo LSTM.
    
    Utiliza Pydantic BaseModel para validação automática de tipos e geração
    de documentação OpenAPI nos endpoints FastAPI.
    
    Attributes:
        stock (str): Símbolo da ação na B3 (ex: 'VALE3.SA', 'PETR4.SA')
        epochs (int): Número de iterações de treinamento (ex: 50, 100, 200)
        window (int): Tamanho da janela temporal de histórico (ex: 10, 20, 30)
            - Define quantos dias anteriores são usados como features
            - Maior window = mais contexto histórico
            - Menor window = menos dependência de dados antigos
        per_training (float): Fração de dados para treinamento (0.0 < per_training < 1.0)
            - Exemplo: 0.8 = 80% treino, 20% validação
            - Usado para divisão temporal treino/teste
        learning_rate (float): Taxa de aprendizado do otimizador Adam (ex: 0.001, 0.0001)
            - Valores maiores: convergência rápida mas pode divergir
            - Valores menores: convergência lenta mas mais estável
        hidden_size (int): Dimensionalidade das camadas ocultas LSTM (ex: 32, 64, 128)
            - Maior hidden_size = maior capacidade do modelo (mais parâmetros)
            - Menor hidden_size = modelo mais rápido e simples
        num_layers (int): Número de camadas LSTM empilhadas (ex: 1, 2, 3)
            - Camadas adicionais permitem aprender padrões mais complexos
            - Aumenta custo computacional e risco de overfitting
        model_type (str): Tipo de arquitetura ('simple' ou 'complex')
            - 'simple': LSTM + Linear (regressão direta)
            - 'complex': LSTM + MLP com camadas ocultas (mais flexível)
    
    Example:
        ```python
        params = LSTMParams(
            stock="VALE3.SA",
            epochs=100,
            window=20,
            per_training=0.8,
            learning_rate=0.001,
            hidden_size=64,
            num_layers=2,
            model_type="simple"
        )
        ```
    """
    stock: str
    epochs: int
    window: int
    per_training: float
    learning_rate: float
    hidden_size: int
    num_layers: int
    model_type: str