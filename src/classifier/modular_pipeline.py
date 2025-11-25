"""
Pipeline principal MLP - Versão modularizada do classifier.py original.

Esta implementação mantém EXATAMENTE a mesma funcionalidade e comportamento
do classifier.py original, mas de forma modularizada e organizada.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType
from sklearn.model_selection import StratifiedKFold
from typing import Optional, Dict, Any, List, Tuple
import logging

# Import SplitIndices for external stratification
try:
    from src.build.pipeline.split_indices import SplitIndices
except ImportError:
    SplitIndices = None  # Fallback if not available

# Imports dos módulos modularizados
try:
    # Primeiro tenta import relativo (correto)
    from .models.mlp_classifier import MLPEmbeddingClassifier, create_mlp_model
    from .core.evaluator import ModelEvaluator, DataTypeConverter
    from .core.data_loader import DataManager
except ImportError:
    try:
        # Tenta import absoluto
        from classifier.models.mlp_classifier import MLPEmbeddingClassifier, create_mlp_model
        from classifier.core.evaluator import ModelEvaluator, DataTypeConverter
        from classifier.core.data_loader import DataManager
    except ImportError:
        # Fallback para execução direta
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)
        
        from models.mlp_classifier import MLPEmbeddingClassifier, create_mlp_model
        from core.evaluator import ModelEvaluator, DataTypeConverter
        from core.data_loader import DataManager

logger = logging.getLogger(__name__)


class MLPEmbeddingPipeline:
    """
    Pipeline MLP que implementa EXATAMENTE a mesma funcionalidade
    da classe MLPEmbeddingPipeline do classifier.py original.
    
    Mantém todos os parâmetros, métodos e comportamentos idênticos.
    """
    
    def __init__(self, 
                 embeddings_path: str,
                 labels_path: str, 
                 batch_size: int = 64,
                 lr: float = 0.001,
                 epochs: int = 50,
                 test_split: float = 0.1,
                 val_split: float = 0.1, 
                 early_stopping_patience: int = 5,
                 model_output: str = "mlp_model.pth",
                 metrics_output: str = "training_metrics.json",
                 split_indices: Optional['SplitIndices'] = None):
        """
        Inicializa pipeline com EXATAMENTE os mesmos parâmetros do original.
        
        Args:
            embeddings_path: Caminho para o arquivo de embeddings (.npy)
            labels_path: Caminho para o arquivo de labels (.npy)
            batch_size: Tamanho do batch (default: 64)
            lr: Taxa de aprendizado (default: 0.001)
            epochs: Número de épocas (default: 50)
            test_split: Proporção do conjunto de teste (não usado diretamente)
            val_split: Proporção do conjunto de validação (não usado diretamente)
            early_stopping_patience: Paciência para early stopping (default: 5)
            model_output: Caminho para salvar o modelo (default: "mlp_model.pth")
            metrics_output: Caminho para salvar métricas (default: "training_metrics.json")
            split_indices: Optional SplitIndices object with pre-defined train/val/test splits.
                          If provided, these indices will be used instead of random splitting.
                          This ensures consistency with other pipelines (e.g., regression).
        """
        self.embeddings_path = embeddings_path
        self.labels_path = labels_path
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.model_output = model_output
        self.metrics_output = metrics_output
        self.split_indices = split_indices  # Store external splits if provided
        # Os parâmetros test_split e val_split não serão usados diretamente,
        # pois a divisão será feita estratificadamente para garantir a proporção exata.
        
        # Configurar dispositivo (prioridade: CUDA > MPS > CPU)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        self.input_dim = self.get_embedding_dim()
        
        # Componentes modularizados
        self.data_manager = DataManager(embeddings_path, labels_path, self.device)
        self.evaluator = ModelEvaluator(self.device)
        self.converter = DataTypeConverter()
        
        # Inicializa Spark - EXATAMENTE como no original
        self.spark = SparkSession.builder.appName("MLP Training Metrics").getOrCreate()
        
        # Data loaders (serão preenchidos no load_data)
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None

    def get_embedding_dim(self) -> int:
        """
        Obtém automaticamente a dimensão do embedding carregando a primeira amostra.
        
        IMPLEMENTAÇÃO IDÊNTICA ao classifier.py original.
        """
        embeddings = np.load(self.embeddings_path, allow_pickle=True)
        return embeddings.shape[1]
    
    def load_data(self, 
                  train_idx: Optional[np.ndarray] = None,
                  val_idx: Optional[np.ndarray] = None,
                  test_idx: Optional[np.ndarray] = None) -> None:
        """
        Carrega embeddings e rótulos binários, realizando a divisão estratificada em 
        80% treino, 10% validação e 10% teste.
        
        IMPLEMENTAÇÃO IDÊNTICA ao classifier.py original.
        
        Se os índices já forem fornecidos, eles serão usados; caso contrário, 
        a divisão padrão é aplicada.
        
        **NEW**: If split_indices was provided during initialization, those indices
        will be used by default (can be overridden by passing explicit train/val/test_idx).
        This ensures consistent splits across classification and regression pipelines.
        """
        # Use split_indices if available and no explicit indices provided
        if train_idx is None and val_idx is None and test_idx is None:
            if self.split_indices is not None:
                train_idx = self.split_indices.train_idx
                val_idx = self.split_indices.val_idx
                test_idx = self.split_indices.test_idx
                logger.info(f"Using external split indices: train={len(train_idx)}, "
                          f"val={len(val_idx)}, test={len(test_idx)}")
        
        self.train_loader, self.val_loader, self.test_loader = self.data_manager.create_data_loaders(
            train_idx, val_idx, test_idx, self.batch_size
        )
    
    def evaluate(self, model: nn.Module, dataloader: DataLoader) -> Dict[str, Any]:
        """
        Avalia o modelo e retorna um dicionário com as métricas calculadas.
        
        IMPLEMENTAÇÃO IDÊNTICA ao classifier.py original.
        """
        return self.evaluator.evaluate(model, dataloader)

    def convert_to_native(self, data: Any) -> Any:
        """
        Converte recursivamente dicionários e listas para usar tipos nativos do Python.
        
        IMPLEMENTAÇÃO IDÊNTICA ao classifier.py original.
        """
        return self.converter.convert_to_native(data)

    def train(self,
              train_idx: Optional[np.ndarray] = None,
              val_idx: Optional[np.ndarray] = None,
              test_idx: Optional[np.ndarray] = None,
              hyperparameters: Optional[Dict[str, Any]] = None) -> float:
        """
        Treina a MLP e avalia as métricas para treino, validação e teste.
        
        IMPLEMENTAÇÃO IDÊNTICA ao classifier.py original.
        
        O conjunto de teste permanece intocado até a última avaliação.
        Retorna a Loss de validação do último epoch para otimização.
        Se test_idx for None (modo CV), não são salvos arquivos nem criados DataFrames.
        O parâmetro 'hyperparameters' (dicionário) será incluído no arquivo de métricas se fornecido.
        """
        self.load_data(train_idx, val_idx, test_idx)
        
        # Criar modelo EXATAMENTE como no original
        model = MLPEmbeddingClassifier(self.input_dim).to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=2, factor=0.5
        )
        
        # Early stopping - EXATAMENTE como no original
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        epoch_metrics_log = []
        best_model_state = None
        
        # Loop de treinamento - EXATAMENTE como no original
        for epoch in range(self.epochs):
            model.train()
            running_loss = 0
            
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            # Avaliação - EXATAMENTE como no original
            train_metrics = self.evaluate(model, self.train_loader)
            val_metrics = self.evaluate(model, self.val_loader)
            scheduler.step(val_metrics["Loss"])
            
            # Log das métricas - EXATAMENTE como no original
            epoch_record = {"Epoch": epoch + 1}
            for key, value in train_metrics.items():
                epoch_record["Train " + key] = value
            for key, value in val_metrics.items():
                epoch_record["Validation " + key] = value
            epoch_metrics_log.append(epoch_record)
            
            # Logging - EXATAMENTE como no original
            logger.info(f"Epoch [{epoch+1}/{self.epochs}]")
            logger.info(f"  Treino -> Loss: {epoch_record['Train Loss']:.4f} - Acc: {epoch_record['Train Accuracy']:.4f} - Prec: {epoch_record['Train Precision']:.4f} - F1: {epoch_record['Train F1']:.4f} - AUC: {epoch_record['Train ROC_AUC']:.4f}")
            logger.info(f"  Validação -> Loss: {epoch_record['Validation Loss']:.4f} - Acc: {epoch_record['Validation Accuracy']:.4f} - Prec: {epoch_record['Validation Precision']:.4f} - F1: {epoch_record['Validation F1']:.4f} - AUC: {epoch_record['Validation ROC_AUC']:.4f}")
            
            # Early stopping - EXATAMENTE como no original
            if val_metrics["Loss"] < best_val_loss:
                best_val_loss = val_metrics["Loss"]
                epochs_without_improvement = 0
                best_model_state = model.state_dict()
            else:
                epochs_without_improvement += 1
            
            if epochs_without_improvement >= self.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Se test_idx for None, assume modo de Cross-Validation - EXATAMENTE como no original
        if self.test_loader is None:
            logger.info("Modo Cross-Validation: pulando salvamento de métricas e criação de DataFrames Spark.")
            torch.save(model.state_dict(), self.model_output)
            logger.info(f"✅ Modelo salvo em '{self.model_output}'.")
            return epoch_metrics_log[-1]["Validation Loss"]
        
        # Avaliação final no conjunto de teste (modo final) - EXATAMENTE como no original
        test_metrics = self.evaluate(model, self.test_loader)
        test_metrics_record = {"Test " + key: value for key, value in test_metrics.items()}
        
        logger.info("Avaliação final no conjunto de teste:")
        logger.info(f"  Test -> Loss: {test_metrics['Loss']:.4f} - Acc: {test_metrics['Accuracy']:.4f} - Prec: {test_metrics['Precision']:.4f} - F1: {test_metrics['F1']:.4f} - AUC: {test_metrics['ROC_AUC']:.4f}")
        
        # Salvar métricas - EXATAMENTE como no original
        epoch_metrics_log = self.convert_to_native(epoch_metrics_log)
        test_metrics_record = self.convert_to_native(test_metrics_record)
        final_metrics = {"epoch_metrics": epoch_metrics_log, "test_metrics": test_metrics_record}
        
        if hyperparameters is not None:
            final_metrics["best_hyperparameters"] = hyperparameters
        
        with open(self.metrics_output, "w") as f:
            json.dump(final_metrics, f, indent=4)
        logger.info(f"✅ Métricas de treino, validação e teste salvas em '{self.metrics_output}'.")
        
        # DataFrames Spark - EXATAMENTE como no original
        self._create_spark_dataframes(epoch_metrics_log, test_metrics_record)
        
        # Salvar modelo - EXATAMENTE como no original
        torch.save(model.state_dict(), self.model_output)
        logger.info(f"✅ Modelo salvo em '{self.model_output}'.")
        
        return epoch_metrics_log[-1]["Validation Loss"]
    
    def _create_spark_dataframes(self, epoch_metrics_log: List[Dict], test_metrics_record: Dict) -> None:
        """
        Cria DataFrames Spark EXATAMENTE como no classifier.py original.
        """
        # Schema para métricas de época - EXATAMENTE como no original
        schema = StructType([
            StructField("Epoch", IntegerType(), True),
            StructField("Train Loss", DoubleType(), True),
            StructField("Train Accuracy", DoubleType(), True),
            StructField("Train Precision", DoubleType(), True),
            StructField("Train Recall", DoubleType(), True),
            StructField("Train F1", DoubleType(), True),
            StructField("Train ROC_AUC", DoubleType(), True),
            StructField("Train True_Negatives", IntegerType(), True),
            StructField("Train False_Positives", IntegerType(), True),
            StructField("Train False_Negatives", IntegerType(), True),
            StructField("Train True_Positives", IntegerType(), True),
            StructField("Train Specificity", DoubleType(), True),
            StructField("Train Fbeta_0.5", DoubleType(), True),
            StructField("Train Fbeta_2", DoubleType(), True),
            StructField("Train MCC", DoubleType(), True),
            StructField("Train Average_Precision", DoubleType(), True),
            StructField("Train Brier_Score", DoubleType(), True),
            StructField("Validation Loss", DoubleType(), True),
            StructField("Validation Accuracy", DoubleType(), True),
            StructField("Validation Precision", DoubleType(), True),
            StructField("Validation Recall", DoubleType(), True),
            StructField("Validation F1", DoubleType(), True),
            StructField("Validation ROC_AUC", DoubleType(), True),
            StructField("Validation True_Negatives", IntegerType(), True),
            StructField("Validation False_Positives", IntegerType(), True),
            StructField("Validation False_Negatives", IntegerType(), True),
            StructField("Validation True_Positives", IntegerType(), True),
            StructField("Validation Specificity", DoubleType(), True),
            StructField("Validation Fbeta_0.5", DoubleType(), True),
            StructField("Validation Fbeta_2", DoubleType(), True),
            StructField("Validation MCC", DoubleType(), True),
            StructField("Validation Average_Precision", DoubleType(), True),
            StructField("Validation Brier_Score", DoubleType(), True)
        ])
        
        epoch_metrics_df = self.spark.createDataFrame(epoch_metrics_log, schema=schema)
        logger.info("DataFrame com métricas de treino e validação:")
        epoch_metrics_df.show(truncate=False)
        
        # Schema para métricas de teste - EXATAMENTE como no original
        test_schema = StructType([
            StructField("Test Loss", DoubleType(), True),
            StructField("Test Accuracy", DoubleType(), True),
            StructField("Test Precision", DoubleType(), True),
            StructField("Test Recall", DoubleType(), True),
            StructField("Test F1", DoubleType(), True),
            StructField("Test ROC_AUC", DoubleType(), True),
            StructField("Test True_Negatives", IntegerType(), True),
            StructField("Test False_Positives", IntegerType(), True),
            StructField("Test False_Negatives", IntegerType(), True),
            StructField("Test True_Positives", IntegerType(), True),
            StructField("Test Specificity", DoubleType(), True),
            StructField("Test Fbeta_0.5", DoubleType(), True),
            StructField("Test Fbeta_2", DoubleType(), True),
            StructField("Test MCC", DoubleType(), True),
            StructField("Test Average_Precision", DoubleType(), True),
            StructField("Test Brier_Score", DoubleType(), True)
        ])
        
        test_metrics_df = self.spark.createDataFrame([test_metrics_record], schema=test_schema)
        logger.info("DataFrame com métricas do conjunto de teste:")
        test_metrics_df.show(truncate=False)

    def cross_validate(self, k: int = 5) -> float:
        """
        Realiza validação cruzada (k-fold, com k=5 por padrão) sobre o conjunto de treino+validação.
        
        IMPLEMENTAÇÃO IDÊNTICA ao classifier.py original.
        
        Retorna a média da Loss de validação dos k folds.
        """
        embeddings, labels = self.data_manager.get_data_for_cv()
        indices = np.arange(len(labels))
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        val_losses = []
        fold = 1
        
        for train_idx, val_idx in skf.split(indices, labels):
            logger.info(f"Iniciando fold {fold} de {k}")
            # No modo CV, não usamos conjunto de teste (test_idx=None)
            current_val_loss = self.train(train_idx=train_idx, val_idx=val_idx, test_idx=None)
            logger.info(f"Fold {fold}: Validation Loss = {current_val_loss:.4f}")
            val_losses.append(current_val_loss)
            fold += 1
        
        avg_val_loss = np.mean(val_losses)
        logger.info(f"Média da Validation Loss nos {k} folds: {avg_val_loss:.4f}")
        return avg_val_loss


# Função de conveniência para criar pipeline
def create_pipeline(embeddings_path: str,
                   labels_path: str,
                   **kwargs) -> MLPEmbeddingPipeline:
    """
    Factory function para criar pipeline MLP.
    
    Args:
        embeddings_path: Caminho para embeddings
        labels_path: Caminho para labels
        **kwargs: Argumentos adicionais para o pipeline
        
    Returns:
        Pipeline MLP configurado
    """
    return MLPEmbeddingPipeline(embeddings_path, labels_path, **kwargs)


if __name__ == "__main__":
    # Teste básico do pipeline
    print("🧪 Testando MLPEmbeddingPipeline modularizado...")
    
    # Criar dados sintéticos para teste
    import tempfile
    
    n_samples = 500
    n_features = 256
    
    # Dados sintéticos
    embeddings = np.random.randn(n_samples, n_features).astype(np.float32)
    labels = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    
    # Salvar em arquivos temporários
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f_emb:
        np.save(f_emb.name, embeddings)
        emb_path = f_emb.name
    
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f_lab:
        np.save(f_lab.name, labels)
        lab_path = f_lab.name
    
    try:
        # Testar pipeline
        pipeline = MLPEmbeddingPipeline(
            embeddings_path=emb_path,
            labels_path=lab_path,
            batch_size=32,
            epochs=2,  # Rápido para teste
            model_output="test_model.pth",
            metrics_output="test_metrics.json"
        )
        
        print(f"✅ Pipeline criado - Embedding dim: {pipeline.input_dim}")
        
        # Testar cross-validation (rápido)
        avg_loss = pipeline.cross_validate(k=2)  # 2 folds para teste rápido
        print(f"✅ Cross-validation concluído - Loss média: {avg_loss:.4f}")
        
        print("🎯 MLPEmbeddingPipeline modularizado funcionando perfeitamente!")
        
    finally:
        # Limpar arquivos temporários
        import os
        if os.path.exists(emb_path):
            os.unlink(emb_path)
        if os.path.exists(lab_path):
            os.unlink(lab_path)
        if os.path.exists("test_model.pth"):
            os.unlink("test_model.pth")
        if os.path.exists("test_metrics.json"):
            os.unlink("test_metrics.json")
