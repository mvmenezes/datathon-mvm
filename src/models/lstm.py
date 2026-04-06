"""
Módulo com arquiteturas de redes neurais LSTM para predição de séries temporais.

Contém duas variações de modelos:
- LSTMSimple: Abordagem direta (LSTM + Linear regression)
- LSTMComplex: Abordagem com MLP adicional para maior flexibilidade

Utiliza factory pattern para instanciação dinâmica dos modelos.
"""

import torch.nn as nn


class LSTMSimple(nn.Module):
    """
    Modelo LSTM simples para predição de séries temporais.
    
    Arquitetura: LSTM -> Linear
    
    Características:
    - Mais rápido de treinar
    - Menos parâmetros
    - Melhor generalização em datasets pequenos
    - Performance: adequada para padrões simples
    
    Args:
        input_size (int): Número de features de entrada (default: 6)
            - Corresponde ao número de colunas nos dados de entrada
        hidden_size (int): Dimensionalidade do estado oculto LSTM (default: 64)
            - Maior valor captura padrões mais complexos
        num_layers (int): Número de camadas LSTM empilhadas (default: 2)
            - Permite aprender representações em diferentes níveis de abstração
    
    Forward pass:
        input shape: (batch_size, sequence_length, input_size)
        output shape: (batch_size, 1) - um valor predito por amostra
    """
    
    def __init__(self, input_size=6, hidden_size=64, num_layers=2):
        """
        Inicializar camadas do modelo.
        
        Args:
            input_size (int): Dimensionalidade das features de entrada
            hidden_size (int): Dimensionalidade do estado oculto LSTM
            num_layers (int): Número de camadas LSTM
        """
        super().__init__()
        
        # Camada LSTM para processar sequências temporais
        # batch_first=True: entrada é (batch, seq_len, features)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Camada linear para mapeamento final: hidden_size -> 1 valor predito
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Forward pass do modelo.
        
        Args:
            x (torch.Tensor): Entrada de shape (batch_size, sequence_length, input_size)
        
        Returns:
            torch.Tensor: Predição de shape (batch_size, 1)
        """
        # Forward pass LSTM
        # out: (batch_size, sequence_length, hidden_size)
        # _: estados finais (não utilizados aqui)
        out, _ = self.lstm(x)
        
        # Selecionar apenas o último timestep (t=-1) para fazer predição
        # shape: (batch_size, hidden_size)
        out = out[:, -1, :]
        
        # Aplicar camada linear para obter predição final
        # shape: (batch_size, 1)
        out = self.fc(out)
        
        return out


class LSTMComplex(nn.Module):
    """
    Modelo LSTM complexo com rede neural adicional (MLP).
    
    Arquitetura: LSTM -> Linear(hidden_size, 32) -> ReLU -> Linear(32, 16) -> ReLU -> Linear(16, 1)
    
    Características:
    - Maior capacidade de aprendizado
    - Mais parâmetros para ajustar
    - Melhor em datasets maiores
    - Risco maior de overfitting
    - Performance: captura padrões mais complexos
    
    Args:
        input_size (int): Número de features de entrada (default: 6)
        hidden_size (int): Dimensionalidade do estado oculto LSTM (default: 64)
        num_layers (int): Número de camadas LSTM empilhadas (default: 2)
    
    Forward pass:
        input shape: (batch_size, sequence_length, input_size)
        output shape: (batch_size, 1)
    """
    
    def __init__(self, input_size=6, hidden_size=64, num_layers=2):
        """
        Inicializar camadas do modelo.
        
        Args:
            input_size (int): Dimensionalidade das features de entrada
            hidden_size (int): Dimensionalidade do estado oculto LSTM
            num_layers (int): Número de camadas LSTM
        """
        super().__init__()
        
        # Camada LSTM para processar sequências temporais
        # batch_first=True: entrada é (batch, seq_len, features)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # MLP (Multi-Layer Perceptron) com ativações ReLU para não-linearidade
        # Arquitetura: hidden_size -> 32 -> 16 -> 1
        # ReLU introduz não-linearidade entre as camadas
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),  # Camada 1: hidden_size -> 32
            nn.ReLU(),                    # Ativação ReLU
            nn.Linear(32, 16),            # Camada 2: 32 -> 16
            nn.ReLU(),                    # Ativação ReLU
            nn.Linear(16, 1)              # Camada 3: 16 -> 1 (predição final)
        )

    def forward(self, x):
        """
        Forward pass do modelo.
        
        Args:
            x (torch.Tensor): Entrada de shape (batch_size, sequence_length, input_size)
        
        Returns:
            torch.Tensor: Predição de shape (batch_size, 1)
        """
        # Forward pass LSTM
        # out: (batch_size, sequence_length, hidden_size)
        # _: estados finais (não utilizados aqui)
        out, _ = self.lstm(x)
        
        # Selecionar apenas o último timestep (t=-1) para fazer predição
        # shape: (batch_size, hidden_size)
        out = out[:, -1, :]
        
        # Aplicar MLP sequencial
        # shape final: (batch_size, 1)
        out = self.fc(out)
        
        return out


class ModelFactory:
    """
    Factory pattern para instanciação de modelos LSTM.
    
    Permite criar modelos dinamicamente por tipo sem precisar importar
    as classes específicas diretamente. Facilita extensão e testes.
    
    Modelos disponíveis:
        - 'simple': LSTMSimple (mais rápido, menos parâmetros)
        - 'complex': LSTMComplex (mais expressivo, mais parâmetros)
    
    Example:
        ```python
        # Criar modelo simples
        model_simple = ModelFactory.create(
            'simple',
            hidden_size=64,
            input_size=6,
            num_layers=2
        )
        
        # Criar modelo complexo
        model_complex = ModelFactory.create(
            'complex',
            hidden_size=64,
            input_size=6,
            num_layers=2
        )
        ```
    """
    
    # Dicionário mapeando nomes para classes de modelos
    _models = {
        "simple": LSTMSimple,
        "complex": LSTMComplex
    }

    @staticmethod
    def create(tipo: str, hidden_size, input_size, num_layers):
        """
        Criar instância de modelo LSTM.
        
        Args:
            tipo (str): Nome do tipo de modelo ('simple' ou 'complex')
                - Case-insensitive (converte para lowercase)
            hidden_size (int): Dimensionalidade do estado oculto
            input_size (int): Número de features de entrada
            num_layers (int): Número de camadas LSTM
        
        Returns:
            nn.Module: Instância do modelo solicitado
        
        Raises:
            ValueError: Se tipo não estiver registrado nos modelos disponíveis
        
        Example:
            ```python
            model = ModelFactory.create(
                tipo='simple',
                hidden_size=64,
                input_size=6,
                num_layers=2
            )
            ```
        """
        # Converter tipo para lowercase para aceitar 'Simple', 'SIMPLE', 'simple'
        tipo_lower = tipo.lower()
        
        # Validar se modelo está registrado
        if tipo_lower not in ModelFactory._models:
            raise ValueError(
                f"Modelo inválido: '{tipo}'. "
                f"Escolha entre {list(ModelFactory._models.keys())}"
            )
        
        # Obter classe do modelo e instanciar com parâmetros
        model_class = ModelFactory._models[tipo_lower]
        return model_class(
            hidden_size=hidden_size,
            input_size=input_size,
            num_layers=num_layers
        )