"""
Orquestrador principal para o pipeline de classificação MLP.

PIPELINE COMPLETO:
1. Carregamento e validação de dados
2. Configuração de modelos e treinamento
3. Cross-validation ou otimização de hiperparâmetros
4. Treinamento final e avaliação
5. Salvar resultados e modelos

USO:
    python main.py --data_path data.csv --config config.json --mode train
    python main.py --data_path data.csv --mode hyperopt --n_trials 100
    python main.py --data_path data.csv --mode cv --n_folds 5
"""

import argparse
import logging
import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import torch
import pandas as pd
import numpy as np
from datetime import datetime

# Imports locais com imports relativos
from .config.mlp_config import MLPConfig, create_default_config
from .models.mlp import MLPEmbeddingClassifier
from .core.trainer import ModelTrainer, TrainingConfig
from .core.cross_validator import CrossValidator, CrossValidationConfig, quick_cross_validate
from .core.hyperopt import HyperparameterOptimizer, OptimizationConfig, quick_hyperparameter_search
from .utils.data_validation import DataValidator
from .utils.metrics import MetricsCalculator
from .utils.data_manager import DataManager, ScalableDataset

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mlp_pipeline.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class MLPPipeline:
    """Pipeline completo para classificação MLP."""
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_validator = DataValidator()
        self.metrics_calculator = MetricsCalculator(device=self.device)
        
        # 🚀 NOVO: Gerenciador de dados escalável 
        self.data_manager = DataManager(device=self.device)
        
        # Dados - agora usando sistema escalável
        self.dataset: Optional[ScalableDataset] = None
        self.dataloader: Optional[torch.utils.data.DataLoader] = None
        self.feature_names: Optional[list] = None
        
        # Para compatibilidade (deprecated - usar dataset)
        self.X: Optional[torch.Tensor] = None
        self.y: Optional[torch.Tensor] = None
        
        # Configurações
        self.model_config: Optional[MLPConfig] = None
        self.training_config: Optional[TrainingConfig] = None
        
        # Resultados
        self.results: Dict[str, Any] = {}
        
        logger.info(f"MLPPipeline inicializado - Device: {self.device}")
    
    def load_data(self, data_path: Path, target_column: str = "target", 
                  feature_columns: Optional[list] = None, 
                  batch_size: Optional[int] = None) -> None:
        """
        Carrega e valida dados usando sistema escalável (evita OOM).
        
        Args:
            data_path: Caminho para arquivo CSV
            target_column: Nome da coluna target
            feature_columns: Lista de colunas de features (None = todas exceto target)
            batch_size: Tamanho do batch (None = automático)
        """
        logger.info(f"🔄 Carregando dados de: {data_path}")
        
        # Carregar CSV
        df = pd.read_csv(data_path)
        logger.info(f"📊 Dados carregados: {df.shape}")
        
        # Validar colunas
        if target_column not in df.columns:
            raise ValueError(f"Coluna target '{target_column}' não encontrada")
        
        # Selecionar features
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column]
        else:
            missing_cols = set(feature_columns) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Colunas não encontradas: {missing_cols}")
        
        self.feature_names = feature_columns
        
        # Extrair arrays
        X = df[feature_columns].values.astype(np.float32)
        y = df[target_column].values.astype(np.float32)
        
        # Validar dados
        validation_report = self.data_validator.validate_arrays(X, y)
        if not validation_report.is_valid:
            logger.warning("⚠️  Problemas encontrados nos dados:")
            for issue in validation_report.issues:
                logger.warning(f"    - {issue}")
        
        # 🚀 NOVO: Usar sistema escalável ao invés de carregar tudo na GPU
        self.dataset, self.dataloader = self.data_manager.load_from_arrays(
            X, y, batch_size=batch_size, shuffle=True
        )
        
        # Para compatibilidade com código existente (deprecated)
        # ⚠️  AVISO: Pode causar OOM em datasets grandes
        if len(X) <= 10000:  # Limite de segurança
            self.X = torch.from_numpy(X).to(self.device)
            self.y = torch.from_numpy(y).to(self.device)
            logger.info("✅ Dados também disponíveis em formato legacy (X, y)")
        else:
            logger.warning("⚠️  Dataset grande - apenas formato escalável disponível")
            self.X = None
            self.y = None
        
        logger.info(f"✅ Dados preparados: {self.dataset.n_samples} samples, {self.dataset.n_features} features")
        
        # Estatísticas básicas
        if self.y is not None:
            unique_labels, label_counts = torch.unique(self.y, return_counts=True)
            for label, count in zip(unique_labels, label_counts):
                logger.info(f"    Classe {label}: {count} samples ({count/len(self.y)*100:.1f}%)")
        else:
            # Para datasets grandes, fazer amostragem para estatísticas
            y_sample = torch.from_numpy(y[:1000])  # Amostra de 1000
            unique_labels, label_counts = torch.unique(y_sample, return_counts=True)
            logger.info("📈 Distribuição (amostra de 1000):")
            for label, count in zip(unique_labels, label_counts):
                logger.info(f"    Classe {label}: ~{count/10:.0f}% das amostras")
    
    def load_config(self, config_path: Optional[Path] = None) -> None:
        """
        Carrega configuração do modelo e treinamento.
        
        Args:
            config_path: Caminho para arquivo JSON (None = configuração padrão)
        """
        if config_path and config_path.exists():
            logger.info(f"Carregando configuração de: {config_path}")
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            # Configuração do modelo
            model_config = config_dict.get("model", {})
            if self.X is not None:
                model_config["input_size"] = self.X.shape[1]
            self.model_config = MLPConfig(**model_config)
            
            # Configuração de treinamento
            training_config = config_dict.get("training", {})
            self.training_config = TrainingConfig(**training_config)
            
        else:
            logger.info("Usando configuração padrão")
            # 🚀 NOVO: Usar dataset escalável para obter input_size
            input_size = self.dataset.n_features if self.dataset is not None else (
                self.X.shape[1] if self.X is not None else 100
            )
            self.model_config = create_default_config(input_size)
            self.training_config = TrainingConfig()
        
        logger.info(f"Configuração carregada - Input: {self.model_config.input_size}, "
                   f"Hidden: {self.model_config.hidden_layers}")
    
    def run_cross_validation(self, n_folds: int = 5) -> Dict[str, Any]:
        """Executa cross-validation com sistema escalável."""
        if self.dataset is None:
            raise ValueError("Dados não carregados. Use load_data() primeiro.")
        
        logger.info(f"🔄 Executando cross-validation com {n_folds} folds")
        
        # Para cross-validation, precisamos dos dados completos
        # Se dataset for pequeno, usar dados na GPU
        if self.X is not None and self.y is not None:
            logger.info("📊 Usando dados em GPU para cross-validation")
            cv_results = quick_cross_validate(
                model_config=self.model_config,
                X=self.X,
                y=self.y,
                n_splits=n_folds,
                device=self.device
            )
        else:
            # Dataset grande - usar amostragem para cross-validation
            logger.info("📊 Dataset grande - usando dados completos (CPU → GPU sob demanda)")
            X_full, y_full = self.dataset.get_full_data()
            cv_results = quick_cross_validate(
                model_config=self.model_config,
                X=X_full,
                y=y_full,
                n_splits=n_folds,
                device=self.device
            )
        
        self.results["cross_validation"] = cv_results
        
        # Log dos resultados principais
        summary = cv_results["summary_statistics"]
        for metric, stats in summary.items():
            logger.info(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        return cv_results
    
    def run_hyperparameter_optimization(self, n_trials: int = 100, 
                                      cv_folds: int = 3) -> Tuple[MLPConfig, TrainingConfig]:
        """Executa otimização de hiperparâmetros."""
        if self.X is None or self.y is None:
            raise ValueError("Dados não carregados. Use load_data() primeiro.")
        
        logger.info(f"Otimizando hiperparâmetros - {n_trials} trials, {cv_folds} folds")
        
        # Usar função de conveniência
        best_model_config, best_training_config, study = quick_hyperparameter_search(
            X=self.X,
            y=self.y,
            base_config=self.model_config,
            n_trials=n_trials,
            cv_folds=cv_folds,
            device=self.device
        )
        
        # Atualizar configurações com as otimizadas
        self.model_config = best_model_config
        self.training_config = best_training_config
        
        # Salvar resultados
        self.results["hyperparameter_optimization"] = {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "n_trials": len(study.trials),
            "study": study
        }
        
        logger.info(f"Otimização concluída - Melhor valor: {study.best_value:.4f}")
        
        return best_model_config, best_training_config
    
    def train_final_model(self, train_ratio: float = 0.8, use_robust_split: bool = True) -> Dict[str, Any]:
        """Treina modelo final usando sistema escalável de dados."""
        if self.dataset is None:
            raise ValueError("Dados não carregados. Use load_data() primeiro.")
        
        logger.info(f"🚀 Treinando modelo final - {train_ratio*100:.0f}% treino, "
                   f"{(1-train_ratio)*100:.0f}% teste")
        
        # 🚀 NOVO SISTEMA: Divisão escalável sem carregar tudo na GPU
        if use_robust_split:
            from .utils.train_test_split import TrainTestSplitter
            
            # Se temos dados pequenos na GPU, usar método antigo
            if self.X is not None and self.y is not None:
                logger.info("📊 Usando divisão tradicional (dados pequenos)")
                splitter = TrainTestSplitter(random_state=42)
                X_train, X_test, y_train, y_test = splitter.split(
                    self.X, self.y,
                    test_size=1-train_ratio,
                    stratify=True,
                    verbose=True
                )
                
                # Criar datasets tradicionais
                from torch.utils.data import DataLoader, TensorDataset
                train_dataset = TensorDataset(X_train, y_train)
                test_dataset = TensorDataset(X_test, y_test)
                
            else:
                # 🚀 DATASET GRANDE: Divisão por índices (escalável)
                logger.info("📊 Usando divisão escalável (dataset grande)")
                
                # Obter índices de divisão
                from sklearn.model_selection import train_test_split
                n_samples = self.dataset.n_samples
                indices = np.arange(n_samples)
                
                # Fazer split dos índices apenas
                train_indices, test_indices = train_test_split(
                    indices, 
                    test_size=1-train_ratio, 
                    random_state=42,
                    shuffle=True
                )
                
                logger.info(f"✅ Divisão por índices: {len(train_indices)} treino, {len(test_indices)} teste")
                
                # Criar subsets escaláveis
                train_dataset = self.dataset.get_subset(train_indices)
                test_dataset = self.dataset.get_subset(test_indices)
        
        else:
            logger.warning("⚠️  Divisão simples não recomendada para datasets grandes")
            # Para compatibilidade, mas pode causar OOM
            if self.X is not None:
                n_samples = len(self.X)
                n_train = int(n_samples * train_ratio)
                indices = torch.randperm(n_samples)
                train_indices = indices[:n_train]
                test_indices = indices[n_train:]
                
                from torch.utils.data import DataLoader, TensorDataset
                train_dataset = TensorDataset(self.X[train_indices], self.y[train_indices])
                test_dataset = TensorDataset(self.X[test_indices], self.y[test_indices])
            else:
                raise ValueError("Dados grandes requerem use_robust_split=True")
        
        # 🚀 CRIAR DATALOADERS ESCALÁVEIS
        train_loader = self.data_manager.create_dataloader(
            train_dataset, shuffle=True, batch_size=None  # batch_size automático
        )
        test_loader = self.data_manager.create_dataloader(
            test_dataset, shuffle=False, batch_size=None
        )
        
        # Criar e treinar modelo
        model = MLPEmbeddingClassifier(self.model_config)
        model.to(self.device)
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.model_config.learning_rate,
            weight_decay=self.model_config.weight_decay
        )
        
        # Trainer
        trainer = ModelTrainer(
            model=model,
            config=self.training_config,
            device=self.device
        )
        trainer.setup_training(optimizer)
        
        # 🚀 TREINAMENTO ESCALÁVEL
        logger.info("🔥 Iniciando treinamento escalável...")
        training_history = trainer.train(train_loader, test_loader)
        
        # 🚀 AVALIAÇÃO FINAL ESCALÁVEL
        final_metrics = self.metrics_calculator.evaluate_model(
            model, test_loader, 
            amp_enabled=self.training_config.amp_enabled
        )
        
        # Resultados
        final_results = {
            "model": model,
            "training_history": training_history,
            "test_metrics": final_metrics,
            "train_size": len(train_dataset) if hasattr(train_dataset, '__len__') else "unknown",
            "test_size": len(test_dataset) if hasattr(test_dataset, '__len__') else "unknown",
            "memory_efficient": True  # Flag indicando uso de sistema escalável
        }
        
        self.results["final_model"] = final_results
        
        logger.info(f"Modelo final treinado - Test ROC-AUC: {final_metrics.roc_auc:.4f}")
        
        return final_results
    
    def save_results(self, output_dir: Path) -> None:
        """Salva todos os resultados."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Salvar configurações
        config_data = {
            "model_config": self.model_config.__dict__ if self.model_config else None,
            "training_config": self.training_config.__dict__ if self.training_config else None,
            "feature_names": self.feature_names,
            "device": str(self.device)
        }
        
        with open(output_dir / "config.json", 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
        
        # Salvar métricas (excluindo objetos complexos)
        metrics_data = {}
        for key, value in self.results.items():
            if key == "final_model":
                # Salvar apenas métricas, não o modelo
                metrics_data[key] = {
                    "test_metrics": value["test_metrics"].to_dict(),
                    "train_size": value["train_size"],
                    "test_size": value["test_size"],
                    "training_summary": value["training_history"].get_summary()
                }
            elif key == "cross_validation":
                metrics_data[key] = {
                    "summary_statistics": value["summary_statistics"],
                    "best_fold": value["best_fold"],
                    "n_folds": value["n_folds"]
                }
            elif key == "hyperparameter_optimization":
                metrics_data[key] = {
                    "best_params": value["best_params"],
                    "best_value": value["best_value"],
                    "n_trials": value["n_trials"]
                }
        
        with open(output_dir / "results.json", 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
        
        # Salvar modelo final se disponível
        if "final_model" in self.results:
            model = self.results["final_model"]["model"]
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': self.model_config,
                'feature_names': self.feature_names
            }, output_dir / "final_model.pt")
        
        logger.info(f"Resultados salvos em: {output_dir}")


def main():
    """Função principal com interface de linha de comando."""
    parser = argparse.ArgumentParser(description="Pipeline de classificação MLP")
    
    parser.add_argument("--data_path", type=str, required=True,
                       help="Caminho para arquivo CSV com dados")
    parser.add_argument("--config_path", type=str,
                       help="Caminho para arquivo de configuração JSON")
    parser.add_argument("--target_column", type=str, default="target",
                       help="Nome da coluna target")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Diretório para salvar resultados")
    
    # Modos de execução
    parser.add_argument("--mode", type=str, choices=["train", "cv", "hyperopt", "full"],
                       default="train", help="Modo de execução")
    
    # Parâmetros específicos
    parser.add_argument("--n_folds", type=int, default=5,
                       help="Número de folds para cross-validation")
    parser.add_argument("--n_trials", type=int, default=100,
                       help="Número de trials para otimização")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                       help="Proporção de dados para treinamento")
    
    # Configurações técnicas
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "auto"],
                       default="auto", help="Device para treinamento")
    parser.add_argument("--seed", type=int, default=42,
                       help="Seed para reprodutibilidade")
    
    args = parser.parse_args()
    
    # Configurar device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    # Configurar seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Inicializar pipeline
    pipeline = MLPPipeline(device=device)
    
    try:
        # Carregar dados
        pipeline.load_data(Path(args.data_path), args.target_column)
        
        # Carregar configuração
        config_path = Path(args.config_path) if args.config_path else None
        pipeline.load_config(config_path)
        
        # Executar modo selecionado
        if args.mode == "train":
            logger.info("Modo: Treinamento simples")
            pipeline.train_final_model(args.train_ratio)
            
        elif args.mode == "cv":
            logger.info("Modo: Cross-validation")
            pipeline.run_cross_validation(args.n_folds)
            
        elif args.mode == "hyperopt":
            logger.info("Modo: Otimização de hiperparâmetros")
            pipeline.run_hyperparameter_optimization(args.n_trials)
            pipeline.train_final_model(args.train_ratio)
            
        elif args.mode == "full":
            logger.info("Modo: Pipeline completo")
            pipeline.run_hyperparameter_optimization(args.n_trials)
            pipeline.run_cross_validation(args.n_folds)
            pipeline.train_final_model(args.train_ratio)
        
        # Salvar resultados
        output_dir = Path(args.output_dir) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        pipeline.save_results(output_dir)
        
        logger.info("Pipeline concluído com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro no pipeline: {e}")
        raise


if __name__ == "__main__":
    main()
