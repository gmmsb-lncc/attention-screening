#!/usr/bin/env python3
"""
Pipeline MLP - Interface simplificada para classificação
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from torch.utils.data import TensorDataset

# Imports essenciais
from config.mlp_config import MLPConfig, create_default_config
from models.mlp import MLPEmbeddingClassifier
from core.trainer import ModelTrainer, TrainingConfig
from core.data_manager import SimpleDataManager
from utils.device_manager import SimpleDeviceManager
from utils.metrics import MetricsCalculator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MLPPipeline:
    """Pipeline MLP - interface simplificada e direta para classificação."""
    
    def __init__(self):
        """Inicialização simples e direta."""
        # Componentes essenciais
        self.device_manager = SimpleDeviceManager()
        self.device = self.device_manager.get_device()
        self.data_manager = SimpleDataManager()
        self.metrics_calculator = MetricsCalculator(device=self.device)
        
        # Configurações (serão criadas conforme necessário)
        self.model_config: Optional[MLPConfig] = None
        self.training_config: Optional[TrainingConfig] = None
        
        # Dados e modelo
        self.dataset: Optional[TensorDataset] = None
        self.model: Optional[MLPEmbeddingClassifier] = None
        self.trainer: Optional[ModelTrainer] = None
        
        logger.info(f"Pipeline inicializado - Device: {self.device}")
    
    def load_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Carrega dados para o pipeline.
        
        Args:
            X: Features (N, D)
            y: Labels (N,)
        """
        logger.info(f"Carregando dados: X={X.shape}, y={y.shape}")
        
        # Criar dataset
        self.dataset = self.data_manager.create_dataset_from_arrays(X, y)
        
        # Auto-configurar baseado nos dados
        self._auto_configure(X.shape[1], len(np.unique(y)))
        
        logger.info("Dados carregados e configuração criada")
    
    def _auto_configure(self, n_features: int, n_classes: int) -> None:
        """Auto-configura o pipeline baseado nos dados."""
        # Armazenar dimensões
        self.n_features = n_features
        self.n_classes = n_classes
        
        # Criar configuração do modelo (com auto-detecção)
        self.model_config = create_default_config()  # input_size=None para auto-detecção
        self.model_config.use_batch_norm = False  # Desabilitar BatchNorm para evitar problemas com batch size pequeno
        
        # Configuração de treinamento otimizada para os dados
        self.training_config = TrainingConfig()
        self.training_config.max_epochs = 10  # Reduzido para teste rápido
        self.training_config.patience = 5
        
        logger.info(f"Auto-configurado para {n_features} features, {n_classes} classes")
    
    def create_model(self) -> MLPEmbeddingClassifier:
        """Cria o modelo MLP."""
        if self.model_config is None:
            raise ValueError("Configure o modelo primeiro com load_data() ou configure()")
        
        # Passar o input_size explicitamente
        self.model = MLPEmbeddingClassifier(
            self.model_config, 
            input_size=self.n_features
        )
        self.model.to(self.device)
        
        logger.info(f"Modelo criado: {self.model.__class__.__name__}")
        return self.model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Treina o modelo.
        
        Args:
            X_train: Features de treino
            y_train: Labels de treino  
            X_val: Features de validação (opcional)
            y_val: Labels de validação (opcional)
            
        Returns:
            Resultados do treinamento
        """
        logger.info("Iniciando treinamento...")
        
        # Criar datasets
        train_dataset = self.data_manager.create_dataset_from_arrays(X_train, y_train)
        val_dataset = None
        if X_val is not None and y_val is not None:
            val_dataset = self.data_manager.create_dataset_from_arrays(X_val, y_val)
        
        # Criar modelo se não existe
        if self.model is None:
            self.create_model()
        
        # Criar trainer
        self.trainer = ModelTrainer(
            model=self.model,
            config=self.training_config,
            device=self.device
        )
        
        # Criar optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-3,  # Learning rate padrão
            weight_decay=1e-4
        )
        
        # Setup do treinamento (configurar optimizer, etc.)
        self.trainer.setup_training(optimizer)
        
        # Treinar
        history = self.trainer.train(train_dataset, val_dataset)
        results = history.get_summary() if hasattr(history, 'get_summary') else {'training': 'completed'}
        
        logger.info(f"Treinamento concluído - Total de épocas: {history.total_epochs if hasattr(history, 'total_epochs') else 'N/A'}")
        return results
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Avalia o modelo em dados de teste.
        
        Args:
            X_test: Features de teste
            y_test: Labels de teste
            
        Returns:
            Métricas de avaliação
        """
        if self.model is None:
            raise ValueError("Treine o modelo primeiro")
        
        logger.info("Avaliando modelo...")
        
        # Criar dataset de teste
        test_dataset = self.data_manager.create_dataset_from_arrays(X_test, y_test)
        test_loader = self.data_manager.create_dataloader(test_dataset, 
                                                         batch_size=64, 
                                                         shuffle=False)
        
        # Avaliar
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                preds = torch.sigmoid(outputs).cpu().numpy()
                
                all_preds.extend(preds.flatten())
                all_labels.extend(batch_y.cpu().numpy())
        
        # Calcular métricas
        y_true = np.array(all_labels).flatten()
        y_prob = np.array(all_preds).flatten()  
        y_pred = (y_prob > 0.5).astype(int)  # Converter probabilidades em predições binárias
        
        metrics = self.metrics_calculator.calculate_metrics(
            y_true=y_true,
            y_prob=y_prob, 
            y_pred=y_pred
        )
        
        logger.info(f"Avaliação concluída - Accuracy: {metrics.accuracy:.4f}")
        return {"accuracy": metrics.accuracy, "roc_auc": metrics.roc_auc, "f1": metrics.f1}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Faz predições em novos dados.
        
        Args:
            X: Features para predição
            
        Returns:
            Probabilidades de classe positiva
        """
        if self.model is None:
            raise ValueError("Treine o modelo primeiro")
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.sigmoid(outputs).cpu().numpy()
        
        return probabilities.flatten()
    
    def quick_train_evaluate(self, X: np.ndarray, y: np.ndarray, 
                           test_size: float = 0.2) -> Dict[str, Any]:
        """
        Pipeline completo: carrega dados, treina e avalia.
        
        Args:
            X: Features completas
            y: Labels completas
            test_size: Proporção para teste
            
        Returns:
            Resultados completos
        """
        logger.info("=== PIPELINE COMPLETO ===")
        
        # 1. Dividir dados
        X_train, X_test, y_train, y_test = self.data_manager.train_test_split(
            X, y, test_size=test_size
        )
        
        # 2. Carregar e configurar
        self.load_data(X_train, y_train)
        
        # 3. Treinar
        train_results = self.train(X_train, y_train)
        
        # 4. Avaliar
        test_results = self.evaluate(X_test, y_test)
        
        # 5. Compilar resultados
        results = {
            "training": train_results,
            "test": test_results,
            "model_info": {
                "input_size": self.model.config.input_size,
                "parameters": self.model.count_parameters(),
                "architecture": self.model.config.get_architecture_summary()
            }
        }
        
        logger.info("=== PIPELINE CONCLUÍDO ===")
        return results


def main():
    """Exemplo de uso do pipeline."""
    # Dados sintéticos para exemplo
    np.random.seed(42)
    X = np.random.randn(1000, 512)  # 1000 amostras, 512 features
    y = np.random.randint(0, 2, 1000)  # Labels binários
    
    # Criar pipeline
    pipeline = MLPPipeline()
    
    # Executar pipeline completo
    results = pipeline.quick_train_evaluate(X, y)
    
    # Mostrar resultados
    print(f"\n🎯 RESULTADOS FINAIS:")
    print(f"Test Accuracy: {results['test']['accuracy']:.4f}")
    print(f"Model Parameters: {results['model_info']['parameters']:,}")
    print(f"Architecture: {results['model_info']['architecture']}")


if __name__ == "__main__":
    main()
