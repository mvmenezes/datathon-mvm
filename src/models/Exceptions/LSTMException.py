"""
Módulo de exceções personalizadas do sistema LSTM.

Define exceções customizadas para diferentes cenários de erro
específicos do domínio de predição de ações.
"""


class ModelNotTrainedException(Exception):
    """
    Exceção levantada quando se tenta usar um modelo que não foi treinado.
    
    Este erro ocorre quando:
    - O arquivo de pesos do modelo não existe no disco
    - A ação solicitada não teve modelo treinado anteriormente
    - O tipo de modelo ('simple' ou 'complex') não correspond
    
    Exemplo de uso:
        ```python
        try:
            model, checkpoints = _load_model('VALE3.SA', 'simple')
        except ModelNotTrainedException as e:
            print(f"Erro: {e}")
            print("Execute o endpoint /train_model primeiro")
        ```
    
    Mensagem típica:
        "Modelo não treinado para essa ação. Por favor, treine o modelo antes de fazer predições."
    """
    pass