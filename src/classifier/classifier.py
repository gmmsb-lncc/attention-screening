#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exemplos de uso:

python classifier.py \
    ../buildEmbedding/concatenated_embeddings/concatenated_embeddings.npy \
    ../buildEmbedding/concatenated_embeddings/interaction_labels.npy \
    --mode manual \
    --lr 0.0005 \
    --batch_size 256 \
    --epochs 80 \
    --hidden_dim 768 \
    --dtype float16 \
    --amp \
    --compile \
    --num_workers 8 \
         # ---------------- avaliação final no conjunto de teste ---------------- #
        if self.test_loader is None:
            logger.info("Modo cross-validation — não há conjunto de teste.")
            if best_model_state:
                torch.save(best_model_state, self.model_output)
            self._shutdown_old_loaders()
            # CORREÇÃO: Retorno consistente para CV
            return best_metric

        model.load_state_dict(best_model_state)  # pesos "ótimos"
        test_metrics = self.evaluate(model, self.test_loader)
        logger.info("✔️  Test Loss %.4f | Test AUC %.4f",
                    test_metrics["Loss"], test_metrics["ROC_AUC"])etric auc

python classifier.py \
    ../buildEmbedding/concatenated_embeddings/concatenated_embeddings.npy \
    ../buildEmbedding/concatenated_embeddings/interaction_labels.npy \
    --mode optuna \
    --trials 20 \
    --cv_folds 5 \
    --dtype float32 \
    --amp \
    --compile
"""

from __future__ import annotations
import os, json, random, logging, argparse, warnings, sys
from typing import Any
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Subset
import torch.multiprocessing as mp

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType

from sklearn.metrics import (
    roc_auc_score, confusion_matrix, fbeta_score, matthews_corrcoef,
    average_precision_score, brier_score_loss
)
from sklearn.model_selection import train_test_split, StratifiedKFold
import optuna

# ---------------------------------------------------------------------------
# 0. Multiprocessing: reduzir uso de descritores de arquivo
# ---------------------------------------------------------------------------
mp.set_sharing_strategy("file_system")  # usa arquivos temporários em vez de pipes

# ---------------------------------------------------------------------------
# 1. Configuração do Experimento
# ---------------------------------------------------------------------------
@dataclass
class MLPConfig:
    """Configuração centralizada para o MLP."""
    # Arquitetura
    hidden_dim: int = 1024
    dropout: float = 0.3
    activation: str = "relu"
    use_batch_norm: bool = True
    n_layers: int = 3
    
    # Treinamento
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 50
    early_stopping_patience: int = 5
    early_metric: str = "loss"  # "loss" ou "auc"
    
    # Otimização
    dtype: str = "float32"  # "float32", "float16", "bfloat16"
    amp: bool = False
    compile_model: bool = False
    num_workers: int = 0
    
    # Validação
    cv_folds: int = 5
    test_size: float = 0.2
    
    # Arquivos
    model_output: str = "mlp_model.pth"
    metrics_output: str = "training_metrics.json"

# ---------------------------------------------------------------------------
# 2. Reprodutibilidade & pequenas otimizações CUDA
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# ---------------------------------------------------------------------------
# 2. Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 3. MLP Model - VERSÃO MELHORADA
# ---------------------------------------------------------------------------
class MLPEmbeddingClassifier(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        dropout: float = 0.3, 
        activation: str = "relu",
        use_batch_norm: bool = True,
        n_layers: int = 3
    ):
        super().__init__()
        
        # Ativação configurável
        activation_map = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "swish": nn.SiLU(),
            "tanh": nn.Tanh()
        }
        self.activation = activation_map.get(activation, nn.ReLU())
        
        layers = []
        current_dim = input_dim
        
        # Primeira camada
        layers.append(nn.Linear(current_dim, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(self.activation)
        layers.append(nn.Dropout(dropout))
        current_dim = hidden_dim
        
        # Camadas intermediárias
        for i in range(n_layers - 2):
            next_dim = hidden_dim // (2 ** (i + 1))
            next_dim = max(next_dim, 64)  # Dimensão mínima
            
            layers.append(nn.Linear(current_dim, next_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(next_dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout))
            current_dim = next_dim
        
        # Camada final
        layers.append(nn.Linear(current_dim, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # logits
        return self.net(x)

# ---------------------------------------------------------------------------
# 4. Pipeline
# ---------------------------------------------------------------------------
class MLPEmbeddingPipeline:
    SUPPORTED_DTYPES = {"float32": torch.float32,
                        "float16": torch.float16,
                        "bfloat16": torch.bfloat16}

    def __init__(
        self,
        embeddings_path: str,
        labels_path: str,
        *,
        batch_size: int = 64,
        lr: float = 1e-3,
        epochs: int = 50,
        hidden_dim: int = 1024,
        early_stopping_patience: int = 5,
        early_metric: str = "loss",            # "loss" | "auc"
        dtype: str = "float32",                # "float32" | "float16" | "bfloat16"
        amp: bool = False,
        compile_model: bool = False,
        num_workers: int = 0,
        model_output: str = "mlp_model.pth",
        metrics_output: str = "training_metrics.json",
    ):
        assert dtype in self.SUPPORTED_DTYPES, f"dtype inválido: {dtype}"
        self.embeddings_path = embeddings_path
        self.labels_path = labels_path
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.hidden_dim = hidden_dim
        self.early_stopping_patience = early_stopping_patience
        self.early_metric = early_metric.lower()
        self.dtype_str = dtype
        self.dtype_torch = self.SUPPORTED_DTYPES[dtype]
        self.amp = amp and torch.cuda.is_available()
        self.compile_model = compile_model and hasattr(torch, "compile")
        self.num_workers = max(0, num_workers)
        self.model_output = model_output
        self.metrics_output = metrics_output

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Validação inicial dos dados
        self.data_validation = self._validate_data()
        
        self.input_dim = self._get_embedding_dim()

        # Spark session (criado sob demanda para reutilizar entre folds)
        self.spark: SparkSession | None = None

        # Armazena DataLoaders para encerrá-los explicitamente
        self._active_loaders: list[DataLoader] = []

    # ------------------------------------------------------------------- #
    def _validate_data(self) -> dict[str, Any]:
        """Valida qualidade dos dados e retorna estatísticas."""
        logger.info("🔍 Validando qualidade dos dados...")
        
        # Carrega dados para análise
        embeddings_np = np.load(self.embeddings_path, mmap_mode="r", allow_pickle=False)
        labels_np = np.load(self.labels_path, mmap_mode="r", allow_pickle=False).astype(np.float32).flatten()
        
        issues = []
        stats = {}
        
        # Validações básicas
        if len(embeddings_np) != len(labels_np):
            issues.append(f"Mismatch de tamanhos: embeddings {len(embeddings_np)}, labels {len(labels_np)}")
        
        # Análise de balanceamento de classes
        unique, counts = np.unique(labels_np, return_counts=True)
        class_distribution = dict(zip(unique, counts))
        imbalance_ratio = max(counts) / min(counts) if len(counts) > 1 else 1.0
        
        if imbalance_ratio > 10:
            issues.append(f"Desbalanceamento severo de classes: {imbalance_ratio:.1f}:1")
        
        stats['class_distribution'] = class_distribution
        stats['imbalance_ratio'] = float(imbalance_ratio)
        stats['total_samples'] = len(labels_np)
        
        # Detectar valores NaN/infinitos
        nan_embeddings = np.isnan(embeddings_np).sum()
        inf_embeddings = np.isinf(embeddings_np).sum()
        if nan_embeddings > 0:
            issues.append(f"Embeddings com NaN: {nan_embeddings}")
        if inf_embeddings > 0:
            issues.append(f"Embeddings com Inf: {inf_embeddings}")
        
        # Detectar duplicatas (amostragem se dataset muito grande)
        sample_size = min(10000, len(embeddings_np))
        if len(embeddings_np) > sample_size:
            logger.info("📊 Analisando duplicatas em amostra de %d exemplos...", sample_size)
            idx_sample = np.random.choice(len(embeddings_np), sample_size, replace=False)
            sample_embeddings = embeddings_np[idx_sample]
        else:
            sample_embeddings = embeddings_np
        
        unique_embeddings = np.unique(sample_embeddings, axis=0)
        duplicate_rate = 1.0 - (len(unique_embeddings) / len(sample_embeddings))
        if duplicate_rate > 0.05:  # >5% duplicatas
            issues.append(f"Alta taxa de duplicatas: {duplicate_rate*100:.1f}%")
        
        stats['duplicate_rate'] = float(duplicate_rate)
        
        # Log dos resultados
        if issues:
            logger.warning("⚠️  Problemas encontrados nos dados:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info("✅ Dados passaram na validação básica")
        
        logger.info(f"📊 Distribuição de classes: {class_distribution}")
        logger.info(f"📊 Taxa de duplicatas: {duplicate_rate*100:.1f}%")
        
        return {'issues': issues, 'stats': stats}

    # ------------------------------------------------------------------- #
    def _get_embedding_dim(self) -> int:
        emb = np.load(self.embeddings_path, mmap_mode="r", allow_pickle=False)
        return emb.shape[1]

    # ------------------------------------------------------------------- #
    def _shutdown_old_loaders(self) -> None:
        """Fecha workers de DataLoaders anteriores (evita vazamento de FDs)."""
        for loader in self._active_loaders:
            if hasattr(loader, "_shutdown_workers"):
                loader._shutdown_workers()
        self._active_loaders.clear()

    # ------------------------------------------------------------------- #
    def load_data(self, train_idx: Any = None, val_idx: Any = None, test_idx: Any = None) -> None:
        """Carrega embeddings/labels, cria DataLoaders eficientes."""
        # Fecha eventuais loaders antigos
        self._shutdown_old_loaders()

        embeddings_np = np.load(self.embeddings_path, mmap_mode="r", allow_pickle=False)

        # dtype conversão (memória ↓)
        if self.dtype_str == "float16":
            embeddings_np = embeddings_np.astype(np.float16, copy=False)
        elif self.dtype_str == "float32":  # as-is
            pass
        else:  # bfloat16 não existe em NumPy → ficará float32 e converte no tensor
            embeddings_np = embeddings_np.astype(np.float32, copy=False)

        labels_np = np.load(self.labels_path, mmap_mode="r", allow_pickle=False)
        try:
            labels_np = labels_np.astype(np.float32).flatten()
        except Exception as e:
            raise ValueError("Labels devem ser 0/1 ou numéricos conversíveis.") from e

        # Tensores (cópia única evita aviso de array read-only)
        X_tensor = torch.from_numpy(embeddings_np.copy()) \
            .to(torch.float16 if self.dtype_str == "float16" else torch.float32)
        y_tensor = torch.as_tensor(labels_np, dtype=torch.float32).unsqueeze(1)
        dataset = TensorDataset(X_tensor, y_tensor)

        if train_idx is None or val_idx is None or test_idx is None:
            idx_all = np.arange(len(dataset))
            train_idx, tmp_idx = train_test_split(
                idx_all, test_size=0.2, stratify=labels_np, random_state=42
            )
            val_idx, test_idx = train_test_split(
                tmp_idx, test_size=0.5, stratify=labels_np[tmp_idx], random_state=42
            )

        # Auditoria de sobreposição
        assert not set(train_idx) & set(val_idx), "Treino e validação se sobrepõem!"
        assert not set(train_idx) & set(test_idx), "Treino e teste se sobrepõem!"
        assert not set(val_idx) & set(test_idx), "Validação e teste se sobrepõem!"

        pin = self.device.type == "cuda"
        common_kwargs = dict(
            batch_size=self.batch_size,
            pin_memory=pin,
            num_workers=self.num_workers,
            persistent_workers=False,  # ← impede vazamento de FDs
        )
        self.train_loader = DataLoader(Subset(dataset, train_idx), shuffle=True, **common_kwargs)
        self.val_loader = DataLoader(Subset(dataset, val_idx), shuffle=False, **common_kwargs)
        self.test_loader = (
            DataLoader(Subset(dataset, test_idx), shuffle=False, **common_kwargs)
            if test_idx is not None
            else None
        )

        # Armazena para posterior shutdown
        self._active_loaders.extend([self.train_loader, self.val_loader, self.test_loader])

        logger.info("Split sizes — Train: %d  Val: %d  Test: %d",
                    len(train_idx), len(val_idx), len(test_idx))

    # ------------------------------------------------------------------- #
    @staticmethod
    def _convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: MLPEmbeddingPipeline._convert_to_native(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [MLPEmbeddingPipeline._convert_to_native(v) for v in obj]
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    # ------------------------------------------------------------------- #
    def _autocast_dtype(self) -> torch.dtype | None:
        """Retorna dtype apropriado para autocast."""
        if not self.amp:
            return None
        if self.dtype_str == "float16":
            return torch.float16
        return torch.bfloat16  # default para bfloat16

    # ------------------------------------------------------------------- #
    def evaluate(self, model: nn.Module, loader: DataLoader) -> dict[str, float]:
        """Calcula métricas em `loader` sem grad."""
        model.eval()
        criterion = nn.BCEWithLogitsLoss()
        total_loss, correct, total = 0.0, 0, 0
        probs, preds, labels = [], [], []

        torch_dtype = self._autocast_dtype()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.amp,
                                                      dtype=torch_dtype):
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=True).to(self.dtype_torch)
                yb = yb.to(self.device, non_blocking=True)

                logits = model(xb)                 # ← sem sigmoid
                loss   = criterion(logits, yb)
                total_loss += loss.item()

                prob_b  = torch.sigmoid(logits)    # probabilidade ∈ [0,1]
                pred_b  = (prob_b >= 0.5).float()  # classe 0/1

                correct += (pred_b == yb).sum().item()
                total   += yb.size(0)

                # ---- conversões seguras p/ NumPy (float32) ----
                probs.extend(prob_b.float().cpu().numpy().ravel())
                preds.extend(pred_b.float().cpu().numpy().ravel())
                labels.extend(yb.float().cpu().numpy().ravel())

        labels = np.asarray(labels, dtype=np.float32)
        preds  = np.asarray(preds,  dtype=np.float32)
        probs  = np.asarray(probs,  dtype=np.float32)

        acc       = correct / total
        precision = np.sum((preds == 1) & (labels == 1)) / max(1, np.sum(preds == 1))
        recall    = np.sum((preds == 1) & (labels == 1)) / max(1, np.sum(labels == 1))
        f1        = 2 * precision * recall / max(1e-6, precision + recall)
        roc_auc   = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5

        if len(np.unique(labels)) > 1:
            tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        else:  # todos da mesma classe
            tn = len(labels) if labels[0] == 0 else 0
            tp = len(labels) if labels[0] == 1 else 0
            fp = fn = 0

        specificity = tn / (tn + fp) if (tn + fp) else 0
        fbeta_0_5   = fbeta_score(labels, preds, beta=0.5, zero_division=0)
        fbeta_2     = fbeta_score(labels, preds, beta=2,   zero_division=0)
        mcc         = matthews_corrcoef(labels, preds) if len(np.unique(labels)) > 1 else 0
        avg_prec    = average_precision_score(labels, probs)
        brier       = brier_score_loss(labels, probs)

        return {
            "Loss": total_loss / len(loader),
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC_AUC": roc_auc,
            "True_Negatives": int(tn),
            "False_Positives": int(fp),
            "False_Negatives": int(fn),
            "True_Positives": int(tp),
            "Specificity": specificity,
            "Fbeta_0.5": fbeta_0_5,
            "Fbeta_2": fbeta_2,
            "MCC": mcc,
            "Average_Precision": avg_prec,
            "Brier_Score": brier,
        }

    # ------------------------------------------------------------------- #
    def train(
        self,
        train_idx=None,
        val_idx=None,
        test_idx=None,
        hyperparameters: dict | None = None,
    ) -> float:
        """Treina, valida e testa (se test_loader presente)."""
        self.load_data(train_idx, val_idx, test_idx)

        model = MLPEmbeddingClassifier(self.input_dim, self.hidden_dim).to(self.device)
        if self.compile_model:
            logger.info("🔧 Compilando modelo (torch.compile)…")
            model = torch.compile(model, mode="reduce-overhead")

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=2, factor=0.5, verbose=False
        )

        scaler = torch.cuda.amp.GradScaler(enabled=self.amp and self.dtype_str == "float16")
        autocast_dtype = self._autocast_dtype()

        # Early-stopping
        best_model_state = None
        if self.early_metric == "loss":
            best_metric = float("inf")
            better = lambda cur, best: cur < best
        else:  # auc
            best_metric = float("-inf")
            better = lambda cur, best: cur > best

        epochs_no_improve = 0
        history: list[dict[str, float]] = []

        for epoch in range(1, self.epochs + 1):
            model.train()
            running_loss = 0.0
            for xb, yb in self.train_loader:
                xb = xb.to(self.device, non_blocking=True).to(self.dtype_torch)
                yb = yb.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=self.amp, dtype=autocast_dtype):
                    out = model(xb)
                    loss = criterion(out, yb)

                if self.amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()

            # Avaliação
            train_metrics = self.evaluate(model, self.train_loader)
            val_metrics = self.evaluate(model, self.val_loader)
            scheduler.step(val_metrics["Loss"])

            record = {"Epoch": epoch,
                      **{f"Train {k}": v for k, v in train_metrics.items()},
                      **{f"Validation {k}": v for k, v in val_metrics.items()}}
            history.append(record)

            metric_to_check = val_metrics["Loss"] if self.early_metric == "loss" else val_metrics["ROC_AUC"]
            if better(metric_to_check, best_metric):
                best_metric = metric_to_check
                best_model_state = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            logger.info(
                "Epoch %03d | Train Loss %.4f | Val Loss %.4f | Val AUC %.4f",
                epoch, train_metrics["Loss"], val_metrics["Loss"], val_metrics["ROC_AUC"]
            )

            if epochs_no_improve >= self.early_stopping_patience:
                logger.info("⏹️  Early stopping at epoch %d.", epoch)
                break

        # ---------------- avaliação final no conjunto de teste ---------------- #
        if self.test_loader is None:
            logger.info("Modo cross-validation — não há conjunto de teste.")
            if best_model_state:
                torch.save(best_model_state, self.model_output)
            self._shutdown_old_loaders()
            return best_metric if self.early_metric == "loss" else -best_metric

        model.load_state_dict(best_model_state)  # pesos “ótimos”
        test_metrics = self.evaluate(model, self.test_loader)
        logger.info("✔️  Test Loss %.4f | Test AUC %.4f",
                    test_metrics["Loss"], test_metrics["ROC_AUC"])

        # ---------------- serialização ---------------- #
        history_native = self._convert_to_native(history)
        report = {
            "epoch_metrics": history_native,
            "test_metrics": self._convert_to_native({f"Test {k}": v for k, v in test_metrics.items()}),
        }
        if hyperparameters:
            report["best_hyperparameters"] = hyperparameters
        with open(self.metrics_output, "w") as fp:
            json.dump(report, fp, indent=2)
        logger.info("📄 Métricas salvas em '%s'.", self.metrics_output)

        # Pequeno DataFrame Spark (para quem usa)
        try:
            if self.spark is None:
                self.spark = SparkSession.builder.appName("MLP Training Metrics").getOrCreate()

            epoch_schema = StructType([
                StructField(k, DoubleType() if isinstance(v, float) else IntegerType(), True)
                for k, v in history_native[0].items()
            ])
            self.spark.createDataFrame(history_native, schema=epoch_schema).show(truncate=False)

            test_schema = StructType([
                StructField(k, DoubleType(), True) for k in report["test_metrics"]
            ])
            self.spark.createDataFrame([report["test_metrics"]], schema=test_schema).show(truncate=False)
        except Exception as e:
            warnings.warn(f"Schemas Spark não exibidos: {e}")

        torch.save(best_model_state, self.model_output)
        logger.info("💾 Modelo salvo em '%s'.", self.model_output)

        # Liberação de recursos
        self._shutdown_old_loaders()
        if self.spark is not None:
            self.spark.stop()
            self.spark = None

        # CORREÇÃO: Retorno consistente de métricas
        if self.early_metric == "loss":
            return test_metrics["Loss"]
        else:
            return test_metrics["ROC_AUC"]

    # ------------------------------------------------------------------- #
    def cross_validate(self, k: int = 5) -> float:
        """Validação cruzada estratificada k-fold CORRIGIDA."""
        labels = np.load(self.labels_path, mmap_mode="r", allow_pickle=False).astype(np.float32).flatten()
        indices = np.arange(len(labels))
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        results = []
        
        logger.info("🔄 Iniciando %d-fold cross-validation...", k)
        for fold, (train_val_idx, test_fold_idx) in enumerate(skf.split(indices, labels), 1):
            # Split interno: train_val_idx -> train + validation
            train_idx, val_idx = train_test_split(
                train_val_idx, 
                test_size=0.2, 
                stratify=labels[train_val_idx], 
                random_state=42 + fold  # Diferente para cada fold
            )
            
            logger.info("🔄 Fold %d/%d - Train: %d, Val: %d, Test: %d", 
                       fold, k, len(train_idx), len(val_idx), len(test_fold_idx))
            
            # CORREÇÃO: Usar test_fold_idx como conjunto de teste do fold
            metric = self.train(train_idx=train_idx, val_idx=val_idx, test_idx=test_fold_idx)
            results.append(metric)
            logger.info("✅ Fold %d result: %.4f", fold, metric)
        
        avg = float(np.mean(results))
        std = float(np.std(results))
        logger.info("⛳ CV Results: %.4f ± %.4f (mean ± std)", avg, std)
        return avg

# ---------------------------------------------------------------------------
# 5. CLI helpers
# ---------------------------------------------------------------------------
def positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError(f"{value} deve ser ≥ 1")
    return ivalue

def parse_args():
    parser = argparse.ArgumentParser(description="Treino de MLP sobre embeddings")
    parser.add_argument("embeddings_path")
    parser.add_argument("labels_path")

    parser.add_argument("--mode", choices=["optuna", "manual"], default="optuna")
    # Hiper-params manuais
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=positive_int, default=64)
    parser.add_argument("--epochs", type=positive_int, default=50)
    parser.add_argument("--hidden_dim", type=positive_int, default=1024)
    parser.add_argument("--early_stopping_patience", type=positive_int, default=5)
    parser.add_argument("--early_metric", choices=["loss", "auc"], default="loss")

    # Optuna
    parser.add_argument("--trials", type=positive_int, default=10)
    parser.add_argument("--cv_folds", type=positive_int, default=5)

    # Desempenho
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--compile", action="store_true", dest="compile_model")
    parser.add_argument("--num_workers", type=int, default=min(8, os.cpu_count()))

    # Arquivos
    parser.add_argument("--model_output", default="mlp_model.pth")
    parser.add_argument("--metrics_output", default="training_metrics.json")
    return parser.parse_args()

# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()

    # ←––– 1. kwargs “fixos”: NÃO incluem batch_size, lr, epochs, hidden_dim
    base_kwargs = dict(
        embeddings_path=args.embeddings_path,
        labels_path=args.labels_path,
        early_stopping_patience=args.early_stopping_patience,
        early_metric=args.early_metric,
        dtype=args.dtype,
        amp=args.amp,
        compile_model=args.compile_model,
        num_workers=args.num_workers,
        model_output=args.model_output,
        metrics_output=args.metrics_output,
    )

    if args.mode == "optuna":
        study = optuna.create_study(
            direction="minimize" if args.early_metric == "loss" else "maximize"
        )

        search_hidden = [256, 512, 768, 1024]

        def objective(trial: optuna.Trial):
            batch = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
            lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
            epochs = trial.suggest_int("epochs", 10, 100, log=True)
            hidden = trial.suggest_categorical("hidden_dim", search_hidden)

            pipe = MLPEmbeddingPipeline(
                **base_kwargs,
                batch_size=batch,
                lr=lr,
                epochs=epochs,
                hidden_dim=hidden,
            )
            return pipe.cross_validate(k=args.cv_folds)

        study.optimize(objective, n_trials=args.trials)
        logger.info("🏆 Melhores hiperparâmetros: %s", study.best_params)

        best = study.best_params
        pipeline = MLPEmbeddingPipeline(
            **base_kwargs,
            batch_size=best["batch_size"],
            lr=best["lr"],
            epochs=best["epochs"],
            hidden_dim=best["hidden_dim"],
        )
        pipeline.train(hyperparameters=best)

    else:  # ––– modo manual
        hyper = {
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "hidden_dim": args.hidden_dim,
            "early_stopping_patience": args.early_stopping_patience,
            "dtype": args.dtype,
        }
        pipeline = MLPEmbeddingPipeline(
            **base_kwargs,
            batch_size=args.batch_size,
            lr=args.lr,
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
        )
        pipeline.train(hyperparameters=hyper)
    logger.info("✅ Treinamento concluído.")