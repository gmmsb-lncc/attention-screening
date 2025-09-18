"""
Modelo MLP para classificação de embeddings moleculares.

Esta implementação mantém exatamente a mesma arquitetura e comportamento
do classifier.py original, mas de forma modularizada.
"""

import torch
import torch.nn as nn
from typing import Optional


class MLPEmbeddingClassifier(nn.Module):
    """
    MLP para classificação binária de embeddings moleculares.
    
    Arquitetura idêntica ao classifier.py original:
    - Camada 1: input_dim -> hidden_dim (ex: 3328 -> 1024)
    - Camada 2: hidden_dim -> hidden_dim//2 (ex: 1024 -> 512) 
    - Camada 3: hidden_dim//2 -> 1 (output binário)
    
    Cada camada hidden tem: Linear -> BatchNorm -> ReLU -> Dropout
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 1024, dropout: float = 0.3):
        """
        Args:
            input_dim: Dimensão dos embeddings de entrada
            hidden_dim: Dimensão da primeira camada oculta (default: 1024)
            dropout: Taxa de dropout (default: 0.3)
        """
        super(MLPEmbeddingClassifier, self).__init__()
        
        # Primeira camada: reduz de input_dim para hidden_dim (ex: 3328 -> 1024)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        
        # Segunda camada: reduz para uma dimensão intermediária (ex: 1024 -> 512)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        
        # Camada final para classificação binária (reduz para 1)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass do modelo.
        
        Args:
            x: Tensor de embeddings [batch_size, input_dim]
            
        Returns:
            Tensor de probabilidades [batch_size, 1] após sigmoid
        """
        x = self.fc1(x)
        # Verifica se o tamanho do batch é maior que 1 antes de aplicar BatchNorm
        if x.size(0) > 1:
            x = self.bn1(x)
        x = self.act1(x)
        x = self.drop1(x)
        
        x = self.fc2(x)
        # Verifica se o tamanho do batch é maior que 1 antes de aplicar BatchNorm
        if x.size(0) > 1:
            x = self.bn2(x)
        x = self.act2(x)
        x = self.drop2(x)
        
        x = self.fc3(x)
        return torch.sigmoid(x)


def create_mlp_model(input_dim: int, 
                     hidden_dim: int = 1024, 
                     dropout: float = 0.3,
                     device: Optional[torch.device] = None) -> MLPEmbeddingClassifier:
    """
    Factory function para criar e configurar o modelo MLP.
    
    Args:
        input_dim: Dimensão dos embeddings de entrada
        hidden_dim: Dimensão da primeira camada oculta
        dropout: Taxa de dropout
        device: Device para colocar o modelo (CPU/CUDA)
        
    Returns:
        Modelo MLP configurado e movido para o device apropriado
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MLPEmbeddingClassifier(input_dim, hidden_dim, dropout)
    model = model.to(device)
    
    return model


if __name__ == "__main__":
    # Teste básico do modelo
    print("🧪 Testando MLPEmbeddingClassifier...")
    
    # Configurações de teste
    batch_size = 4
    input_dim = 512
    hidden_dim = 256
    
    # Criar modelo
    model = create_mlp_model(input_dim, hidden_dim)
    
    # Criar dados de teste
    x = torch.randn(batch_size, input_dim)
    
    # Mover dados para o mesmo device do modelo
    x = x.to(next(model.parameters()).device)
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"✅ Modelo criado com sucesso")
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output range: [{output.min():.4f}, {output.max():.4f}]")
    print(f"   Device: {next(model.parameters()).device}")
    
    # Teste com batch size = 1 (edge case para BatchNorm)
    x_single = torch.randn(1, input_dim)
    x_single = x_single.to(next(model.parameters()).device)  # Mover para device correto
    with torch.no_grad():
        output_single = model(x_single)
    print(f"✅ Teste com batch_size=1: {output_single.shape}")
    
    print("🎯 MLPEmbeddingClassifier funcionando perfeitamente!")
