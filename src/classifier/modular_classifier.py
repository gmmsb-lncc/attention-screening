#!/usr/bin/env python3
"""
Classificador MLP Modularizado - Interface CLI idêntica ao classifier.py original.

Este script mantém EXATAMENTE a mesma interface de linha de comando e funcionalidade
do classifier.py original, mas usando a implementação modularizada.

Exemplos de uso:
python modular_classifier.py embeddings.npy labels.npy --mode manual --lr 0.001 --batch_size 64 --epochs 50
python modular_classifier.py embeddings.npy labels.npy --mode optuna --trials 10 --cv_folds 5
"""

import os
import logging
import argparse
import random
import numpy as np
import torch
import optuna

# Configuração de logging EXATAMENTE como no original
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Fixação de sementes EXATAMENTE como no original
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Import do pipeline modularizado
from modular_pipeline import MLPEmbeddingPipeline


def main():
    """Função principal que implementa EXATAMENTE a mesma lógica do original."""
    
    # Parser de argumentos IDÊNTICO ao original
    parser = argparse.ArgumentParser(description="Treinamento de MLP com embeddings e labels.")
    parser.add_argument("embeddings_path", type=str, help="Caminho para o arquivo de embeddings (.npy)")
    parser.add_argument("labels_path", type=str, help="Caminho para o arquivo de labels (.npy)")
    parser.add_argument("--mode", type=str, choices=["optuna", "manual"], default="optuna",
                        help="Modo de execução: 'optuna' para otimização de hiperparâmetros ou 'manual' para usar hiperparâmetros definidos.")
    parser.add_argument("--lr", type=float, default=0.001, help="Taxa de aprendizado (usado no modo manual).")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (usado no modo manual).")
    parser.add_argument("--epochs", type=int, default=50, help="Número de épocas (usado no modo manual).")
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Paciencia para early stopping (usado no modo manual).")
    parser.add_argument("--trials", type=int, default=10, help="Número de trials para o Optuna (usado no modo optuna).")
    parser.add_argument("--cv_folds", type=int, default=5, help="Número de folds para cross validation no Optuna (usado no modo optuna).")
    parser.add_argument("--model_output", type=str, default="mlp_model.pth", help="Caminho para salvar o modelo treinado.")
    parser.add_argument("--metrics_output", type=str, default="training_metrics.json", help="Caminho para salvar as métricas de treinamento.")
    
    args = parser.parse_args()

    # Modo Optuna EXATAMENTE como no original
    if args.mode == "optuna":
        study = optuna.create_study(direction="minimize")
        
        def objective(trial):
            lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64, 128, 256, 512])
            epochs = trial.suggest_int('epochs', 10, 10)  # EXATAMENTE como no original
            
            pipeline = MLPEmbeddingPipeline(
                embeddings_path=args.embeddings_path,
                labels_path=args.labels_path,
                batch_size=batch_size,
                lr=lr,
                epochs=epochs,
                early_stopping_patience=args.early_stopping_patience,
                model_output=args.model_output,
                metrics_output=args.metrics_output
            )
            avg_val_loss = pipeline.cross_validate(k=args.cv_folds)
            return avg_val_loss
        
        study.optimize(objective, n_trials=args.trials)
        logger.info("Melhores hiperparâmetros: %s", study.best_params)
        
        best_params = study.best_params
        # Treinamento final com os melhores hiperparâmetros
        pipeline = MLPEmbeddingPipeline(
            embeddings_path=args.embeddings_path,
            labels_path=args.labels_path,
            batch_size=best_params["batch_size"],
            lr=best_params["lr"],
            epochs=best_params["epochs"],
            early_stopping_patience=args.early_stopping_patience,
            model_output=args.model_output,
            metrics_output=args.metrics_output
        )
        final_val_loss = pipeline.train(hyperparameters=best_params)
        logger.info("Loss de validação final: %.4f", final_val_loss)
        
    else:  # Modo manual EXATAMENTE como no original
        hyperparameters = {
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "early_stopping_patience": args.early_stopping_patience
        }
        
        pipeline = MLPEmbeddingPipeline(
            embeddings_path=args.embeddings_path,
            labels_path=args.labels_path,
            batch_size=args.batch_size,
            lr=args.lr,
            epochs=args.epochs,
            early_stopping_patience=args.early_stopping_patience,
            model_output=args.model_output,
            metrics_output=args.metrics_output
        )
        
        final_val_loss = pipeline.train(hyperparameters=hyperparameters)
        logger.info("Loss de validação final: %.4f", final_val_loss)


if __name__ == "__main__":
    main()
