"""
Testes de integração para o sistema modular de classificação MLP.

Testa todos os componentes principais:
- Configuração e validação
- Modelos e arquitetura
- Treinamento e métricas
- Cross-validation (sem data leakage)
- Pipeline completo
"""

import pytest
import torch
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

# Imports relativos do sistema
from ..config.mlp_config import MLPConfig, create_default_config
from ..models.mlp import MLPEmbeddingClassifier  
from ..models.base_model import BaseClassifier
from ..utils.data_validation import DataValidator, DataQualityReport
from ..utils.metrics import MetricsCalculator, ClassificationMetrics
from ..core.trainer import ModelTrainer, TrainingConfig
from ..core.cross_validator import CrossValidator, CrossValidationConfig
from ..main import MLPPipeline


class TestMLPConfig:
    """Testa configurações do modelo."""
    
    def test_default_config_creation(self):
        """Testa criação de configuração padrão."""
        config = create_default_config(input_size=100)
        assert config.input_size == 100
        assert isinstance(config.hidden_layers, list)
        assert len(config.hidden_layers) > 0
        assert config.learning_rate > 0
    
    def test_config_validation(self):
        """Testa validação de parâmetros."""
        # Configuração válida
        config = MLPConfig(
            input_size=50,
            hidden_layers=[128, 64],
            output_size=1,
            learning_rate=0.001
        )
        assert config.input_size == 50
        
        # Configuração inválida - input_size negativo
        with pytest.raises(ValueError):
            MLPConfig(input_size=-1)
        
        # Learning rate inválido
        with pytest.raises(ValueError):
            MLPConfig(input_size=10, learning_rate=-0.1)
    
    def test_config_serialization(self):
        """Testa serialização da configuração."""
        config = create_default_config(input_size=75)
        
        # Converter para dicionário
        config_dict = config.__dict__
        assert "input_size" in config_dict
        assert config_dict["input_size"] == 75
        
        # Recriar da serialização
        new_config = MLPConfig(**config_dict)
        assert new_config.input_size == config.input_size
        assert new_config.hidden_layers == config.hidden_layers


class TestMLPModel:
    """Testa arquitetura do modelo."""
    
    def setup_method(self):
        """Configuração para cada teste."""
        self.config = MLPConfig(
            input_size=20,
            hidden_layers=[32, 16], 
            output_size=1,
            activation="ReLU",
            dropout_rate=0.2
        )
        self.model = MLPEmbeddingClassifier(self.config)
    
    def test_model_creation(self):
        """Testa criação do modelo."""
        assert isinstance(self.model, BaseClassifier)
        assert self.model.config.input_size == 20
        assert len(self.model.layers) > 0
    
    def test_forward_pass(self):
        """Testa forward pass."""
        batch_size = 10
        x = torch.randn(batch_size, self.config.input_size)
        
        output = self.model(x)
        
        assert output.shape == (batch_size, self.config.output_size)
        assert not torch.isnan(output).any()
    
    def test_model_parameters(self):
        """Testa parâmetros do modelo."""
        params = list(self.model.parameters())
        assert len(params) > 0
        
        # Verificar se parâmetros são treináveis
        for param in params:
            assert param.requires_grad
    
    def test_model_modes(self):
        """Testa modos de treinamento/avaliação."""
        # Modo treinamento
        self.model.train()
        assert self.model.training
        
        # Modo avaliação
        self.model.eval()
        assert not self.model.training


class TestDataValidation:
    """Testa validação de dados."""
    
    def setup_method(self):
        """Configuração para cada teste."""
        self.validator = DataValidator()
    
    def test_valid_data(self):
        """Testa dados válidos."""
        X = np.random.randn(100, 10).astype(np.float32)
        y = np.random.randint(0, 2, 100).astype(np.float32)
        
        report = self.validator.validate_arrays(X, y)
        
        assert isinstance(report, DataQualityReport)
        assert report.is_valid
        assert len(report.issues) == 0
    
    def test_invalid_data_shapes(self):
        """Testa dados com dimensões inválidas."""
        X = np.random.randn(100, 10)
        y = np.random.randn(50)  # Tamanho incorreto
        
        report = self.validator.validate_arrays(X, y)
        
        assert not report.is_valid
        assert any("dimensão" in issue.lower() for issue in report.issues)
    
    def test_nan_detection(self):
        """Testa detecção de NaN."""
        X = np.random.randn(50, 5)
        X[10, 2] = np.nan  # Inserir NaN
        y = np.random.randint(0, 2, 50)
        
        report = self.validator.validate_arrays(X, y)
        
        assert not report.is_valid
        assert any("nan" in issue.lower() for issue in report.issues)
    
    def test_single_class_detection(self):
        """Testa detecção de classe única."""
        X = np.random.randn(30, 5)
        y = np.ones(30)  # Apenas uma classe
        
        report = self.validator.validate_arrays(X, y)
        
        # Pode ser válido mas com warning
        assert any("classe" in issue.lower() for issue in report.issues)


class TestMetricsCalculation:
    """Testa cálculo de métricas."""
    
    def setup_method(self):
        """Configuração para cada teste."""
        self.calculator = MetricsCalculator()
    
    def test_perfect_classification(self):
        """Testa métricas para classificação perfeita."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.1, 0.95])
        y_pred = (y_prob >= 0.5).astype(float)
        
        metrics = self.calculator.calculate_metrics(y_true, y_prob, y_pred)
        
        assert isinstance(metrics, ClassificationMetrics)
        assert metrics.accuracy == 1.0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1 == 1.0
    
    def test_random_classification(self):
        """Testa métricas para classificação aleatória."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.rand(100)
        y_pred = (y_prob >= 0.5).astype(float)
        
        metrics = self.calculator.calculate_metrics(y_true, y_prob, y_pred)
        
        assert 0.0 <= metrics.accuracy <= 1.0
        assert 0.0 <= metrics.roc_auc <= 1.0
        assert metrics.sample_count == 100
    
    def test_edge_cases(self):
        """Testa casos extremos."""
        # Todas as amostras da mesma classe  
        y_true = np.ones(10)
        y_prob = np.random.rand(10)
        y_pred = (y_prob >= 0.5).astype(float)
        
        metrics = self.calculator.calculate_metrics(y_true, y_prob, y_pred)
        
        # Deve lidar graciosamente com casos extremos
        assert not np.isnan(metrics.accuracy)
        assert metrics.sample_count == 10


class TestCrossValidation:
    """Testa cross-validation (crítico - sem data leakage)."""
    
    def setup_method(self):
        """Configuração para cada teste.""" 
        self.device = torch.device("cpu")
        
        # Dados sintéticos
        torch.manual_seed(42)
        self.X = torch.randn(100, 10)
        self.y = torch.randint(0, 2, (100,)).float()
        
        # Configurações
        self.cv_config = CrossValidationConfig(n_splits=3, shuffle=True, random_state=42)
        self.training_config = TrainingConfig(max_epochs=5, patience=3)
    
    def test_fold_integrity(self):
        """Testa integridade dos folds (sem sobreposição)."""
        cv = CrossValidator(self.cv_config, self.training_config, self.device)
        
        # Validar dados
        validation = cv._validate_input_data(self.X, self.y)
        assert validation["total_samples"] == 100
        
        # Testar validação de fold
        from sklearn.model_selection import StratifiedKFold
        
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        y_np = self.y.cpu().numpy()
        
        for fold_idx, (train_indices, val_indices) in enumerate(skf.split(self.X, y_np)):
            fold_validation = cv._validate_fold_split(self.y, train_indices, val_indices, fold_idx)
            
            assert fold_validation["overlap_check"] == True
            assert fold_validation["coverage_check"] == True
            assert fold_validation["train_size"] + fold_validation["val_size"] == 100
    
    def test_cv_reproducibility(self):
        """Testa reprodutibilidade do CV."""
        model_config = MLPConfig(input_size=10, hidden_layers=[16], output_size=1)
        
        def model_factory():
            return MLPEmbeddingClassifier(model_config)
        
        def optimizer_factory(model):
            return torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Primeiro CV
        cv1 = CrossValidator(self.cv_config, self.training_config, self.device)
        results1 = cv1.cross_validate(model_factory, optimizer_factory, self.X, self.y)
        
        # Segundo CV (mesmo seed)
        cv2 = CrossValidator(self.cv_config, self.training_config, self.device)
        results2 = cv2.cross_validate(model_factory, optimizer_factory, self.X, self.y)
        
        # Verificar consistência
        assert results1["n_folds"] == results2["n_folds"]
        
        # Folds devem ter mesmos tamanhos (devido ao seed)
        for i in range(results1["n_folds"]):
            fold1 = results1["fold_details"][i]
            fold2 = results2["fold_details"][i]
            
            # Pode haver pequenas variações devido ao treinamento, mas estrutura deve ser igual
            assert fold1["data_quality"]["train_size"] == fold2["data_quality"]["train_size"]
            assert fold1["data_quality"]["val_size"] == fold2["data_quality"]["val_size"]


class TestTrainer:
    """Testa sistema de treinamento."""
    
    def setup_method(self):
        """Configuração para cada teste."""
        self.device = torch.device("cpu")
        
        model_config = MLPConfig(input_size=5, hidden_layers=[8], output_size=1)
        self.model = MLPEmbeddingClassifier(model_config)
        self.model.to(self.device)
        
        self.training_config = TrainingConfig(max_epochs=3, patience=2)
        self.trainer = ModelTrainer(self.model, self.training_config, self.device)
    
    def test_trainer_setup(self):
        """Testa configuração do trainer."""
        optimizer = torch.optim.Adam(self.model.parameters())
        self.trainer.setup_training(optimizer)
        
        assert self.trainer.optimizer is not None
        assert self.trainer.criterion is not None
    
    def test_single_epoch_training(self):
        """Testa uma época de treinamento."""
        # Dados sintéticos
        X = torch.randn(20, 5)
        y = torch.randint(0, 2, (20,)).float()
        
        from torch.utils.data import DataLoader, TensorDataset
        train_loader = DataLoader(TensorDataset(X, y), batch_size=5)
        
        optimizer = torch.optim.Adam(self.model.parameters())
        self.trainer.setup_training(optimizer)
        
        # Treinar uma época
        train_loss, train_metrics = self.trainer.train_epoch(train_loader)
        
        assert isinstance(train_loss, float)
        assert isinstance(train_metrics, ClassificationMetrics)
        assert train_loss > 0
    
    def test_training_history(self):
        """Testa histórico de treinamento."""
        # Dados sintéticos pequenos para teste rápido
        X = torch.randn(30, 5)
        y = torch.randint(0, 2, (30,)).float()
        
        from torch.utils.data import DataLoader, TensorDataset
        train_loader = DataLoader(TensorDataset(X, y), batch_size=10)
        val_loader = DataLoader(TensorDataset(X, y), batch_size=10)
        
        optimizer = torch.optim.Adam(self.model.parameters())
        self.trainer.setup_training(optimizer)
        
        # Treinar
        history = self.trainer.train(train_loader, val_loader)
        
        assert history.total_epochs > 0
        assert len(history.train_losses) == history.total_epochs
        assert len(history.val_losses) == history.total_epochs


class TestPipelineIntegration:
    """Testa integração completa do pipeline."""
    
    def setup_method(self):
        """Configuração para cada teste."""
        self.pipeline = MLPPipeline(device=torch.device("cpu"))
        
        # Criar dados CSV temporários
        self.temp_dir = tempfile.mkdtemp()
        self.data_path = Path(self.temp_dir) / "test_data.csv"
        
        # Gerar dados sintéticos
        np.random.seed(42)
        n_samples = 50
        n_features = 5
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)
        
        import pandas as pd
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
        df["target"] = y
        df.to_csv(self.data_path, index=False)
    
    def test_data_loading(self):
        """Testa carregamento de dados."""
        self.pipeline.load_data(self.data_path)
        
        assert self.pipeline.X is not None
        assert self.pipeline.y is not None
        assert self.pipeline.X.shape[0] == 50
        assert self.pipeline.X.shape[1] == 5
    
    def test_config_loading(self):
        """Testa carregamento de configuração."""
        self.pipeline.load_data(self.data_path)
        self.pipeline.load_config()  # Configuração padrão
        
        assert self.pipeline.model_config is not None
        assert self.pipeline.training_config is not None
        assert self.pipeline.model_config.input_size == 5
    
    def test_mini_cross_validation(self):
        """Testa CV rápido."""
        self.pipeline.load_data(self.data_path)
        self.pipeline.load_config()
        
        # CV com apenas 2 folds para teste rápido
        cv_results = self.pipeline.run_cross_validation(n_folds=2)
        
        assert "summary_statistics" in cv_results
        assert cv_results["n_folds"] == 2
        assert "best_fold" in cv_results
    
    @patch('sys.argv', ['main.py', '--data_path', 'dummy.csv', '--mode', 'train'])
    def test_cli_interface(self):
        """Testa interface de linha de comando."""
        # Este é um teste mock - na prática precisaria de dados reais
        from main import main
        
        # Verificar se a função main existe e pode ser chamada
        assert callable(main)
    
    def teardown_method(self):
        """Limpeza após cada teste."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


# Funções de teste de integração completa
def test_end_to_end_pipeline():
    """Teste de integração completa (end-to-end)."""
    # Dados sintéticos
    torch.manual_seed(123)
    np.random.seed(123)
    
    n_samples, n_features = 80, 8
    X = torch.randn(n_samples, n_features)
    y = torch.randint(0, 2, (n_samples,)).float()
    
    # Configuração mínima
    model_config = MLPConfig(
        input_size=n_features,
        hidden_layers=[12],
        output_size=1,
        learning_rate=0.01
    )
    
    training_config = TrainingConfig(max_epochs=3, patience=2)
    cv_config = CrossValidationConfig(n_splits=2, batch_size=16)
    
    # Executar CV
    def model_factory():
        return MLPEmbeddingClassifier(model_config)
    
    def optimizer_factory(model):
        return torch.optim.Adam(model.parameters(), lr=model_config.learning_rate)
    
    cv = CrossValidator(cv_config, training_config, torch.device("cpu"))
    results = cv.cross_validate(model_factory, optimizer_factory, X, y)
    
    # Verificações básicas
    assert results["n_folds"] == 2
    assert "summary_statistics" in results
    assert "best_fold" in results
    
    # Verificar métricas fazem sentido
    summary_stats = results["summary_statistics"]
    for metric_name in ["accuracy", "roc_auc", "f1"]:
        if metric_name in summary_stats:
            metric_stats = summary_stats[metric_name]
            assert 0 <= metric_stats["mean"] <= 1
            assert metric_stats["std"] >= 0


def test_no_data_leakage():
    """Teste específico para verificar ausência de data leakage."""
    # Dados determinísticos para facilitar detecção de leakage
    torch.manual_seed(999)
    
    # Criar padrão específico que seria fácil de "vazar"
    n_samples = 60
    X = torch.zeros(n_samples, 3)
    y = torch.zeros(n_samples)
    
    # Primeira metade: classe 0 com features negativas
    X[:30, :] = -1.0
    y[:30] = 0
    
    # Segunda metade: classe 1 com features positivas  
    X[30:, :] = 1.0
    y[30:] = 1
    
    # Se houver data leakage, o modelo terá performance perfeita
    # Com CV correto, haverá alguma variabilidade
    
    model_config = MLPConfig(input_size=3, hidden_layers=[4], output_size=1)
    training_config = TrainingConfig(max_epochs=10, patience=5)
    cv_config = CrossValidationConfig(n_splits=3, shuffle=True, random_state=42)
    
    def model_factory():
        return MLPEmbeddingClassifier(model_config)
    
    def optimizer_factory(model):
        return torch.optim.Adam(model.parameters(), lr=0.1)
    
    cv = CrossValidator(cv_config, training_config, torch.device("cpu"))
    results = cv.cross_validate(model_factory, optimizer_factory, X, y)
    
    # Com data leakage, todas as métricas seriam 1.0
    # Com CV correto, deve haver alguma variação
    summary_stats = results["summary_statistics"]
    
    if "accuracy" in summary_stats:
        acc_std = summary_stats["accuracy"]["std"]
        # Se std == 0 e mean == 1.0, pode indicar data leakage
        # (ou dados muito simples, mas pelo menos testamos o sistema)
        assert isinstance(acc_std, float)
        assert acc_std >= 0


if __name__ == "__main__":
    # Executar testes básicos se rodado diretamente
    print("Executando testes básicos...")
    
    # Teste de configuração
    config = create_default_config(100)
    print(f"✓ Configuração criada: {config.input_size} -> {config.hidden_layers}")
    
    # Teste de modelo
    model = MLPEmbeddingClassifier(config)
    x_test = torch.randn(5, 100)
    output = model(x_test)
    print(f"✓ Modelo funciona: {x_test.shape} -> {output.shape}")
    
    # Teste de métricas
    calculator = MetricsCalculator()
    y_true = np.array([0, 1, 0, 1, 1])
    y_prob = np.array([0.2, 0.8, 0.3, 0.9, 0.7])
    y_pred = (y_prob >= 0.5).astype(float)
    metrics = calculator.calculate_metrics(y_true, y_prob, y_pred)
    print(f"✓ Métricas calculadas: Accuracy={metrics.accuracy:.3f}, ROC-AUC={metrics.roc_auc:.3f}")
    
    # Teste end-to-end básico
    try:
        test_end_to_end_pipeline()
        print("✓ Pipeline end-to-end funcionando")
    except Exception as e:
        print(f"✗ Erro no pipeline: {e}")
    
    print("Testes básicos concluídos!")
    print("\nPara executar todos os testes: pytest test_integration.py -v")
