import os
import json
import numpy as np
import random
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Subset
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType
from sklearn.metrics import roc_auc_score, confusion_matrix, fbeta_score, matthews_corrcoef, average_precision_score, brier_score_loss
from sklearn.model_selection import train_test_split, StratifiedKFold
import optuna

# ---------------------------
# Fixação de Sementes
# ---------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ---------------------------
# Configuração de Logging
# ---------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------
# Modelo MLPEmbeddingClassifier
# ---------------------------
class MLPEmbeddingClassifier(nn.Module):
    def __init__(self, input_dim):
        super(MLPEmbeddingClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 3328),
            nn.ReLU(),
            nn.Linear(3328, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# ---------------------------
# Pipeline de Treinamento
# ---------------------------
class MLPEmbeddingPipeline:
    def __init__(self, embeddings_path, labels_path, batch_size=64, lr=0.001, epochs=50, test_split=0.1, val_split=0.1, early_stopping_patience=5):
        self.embeddings_path = embeddings_path
        self.labels_path = labels_path
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        # Os parâmetros test_split e val_split não serão usados diretamente,
        # pois a divisão será feita estratificadamente para garantir a proporção exata.
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = self.get_embedding_dim()
        
        # Inicializa Spark
        self.spark = SparkSession.builder.appName("MLP Training Metrics").getOrCreate()

    def get_embedding_dim(self):
        """Obtém automaticamente a dimensão do embedding carregando a primeira amostra."""
        embeddings = np.load(self.embeddings_path, allow_pickle=True)
        return embeddings.shape[1]
    
    def load_data(self, train_idx=None, val_idx=None, test_idx=None):
        """
        Carrega embeddings e rótulos binários, realizando a divisão estratificada em 80% treino, 10% validação e 10% teste.
        Se os índices já forem fornecidos, eles serão usados; caso contrário, a divisão padrão é aplicada.
        """
        embeddings = np.load(self.embeddings_path, allow_pickle=True)
        labels = np.load(self.labels_path, allow_pickle=True)
        labels = labels.flatten()
        X = torch.tensor(embeddings, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        dataset = TensorDataset(X, y)
        if train_idx is None or val_idx is None or test_idx is None:
            indices = np.arange(len(dataset))
            train_idx, temp_idx = train_test_split(indices, test_size=0.2, stratify=labels, random_state=42)
            val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=labels[temp_idx], random_state=42)
        self.train_loader = DataLoader(Subset(dataset, train_idx), batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(Subset(dataset, val_idx), batch_size=self.batch_size, shuffle=False)
        if test_idx is not None:
            self.test_loader = DataLoader(Subset(dataset, test_idx), batch_size=self.batch_size, shuffle=False)
        else:
            self.test_loader = None
    
    def evaluate(self, model, dataloader):
        """Avalia o modelo e retorna um dicionário com as métricas calculadas."""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_probs = []
        all_labels = []
        criterion = nn.BCELoss()
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                total_loss += loss.item()
                preds = (outputs >= 0.5).float()
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
                all_probs.extend(outputs.cpu().numpy().flatten())
                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(y_batch.cpu().numpy().flatten())
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        acc = correct / total
        precision = np.sum((all_preds == 1) & (all_labels == 1)) / max(1, np.sum(all_preds == 1))
        recall = np.sum((all_preds == 1) & (all_labels == 1)) / max(1, np.sum(all_labels == 1))
        f1 = 2 * (precision * recall) / max(1e-6, precision + recall)
        roc_auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
        if len(np.unique(all_labels)) > 1:
            tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
        else:
            if all_labels[0] == 0:
                tn, fp, fn, tp = len(all_labels), 0, 0, 0
            else:
                tn, fp, fn, tp = 0, 0, 0, len(all_labels)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        fbeta_0_5 = fbeta_score(all_labels, all_preds, beta=0.5, zero_division=0)
        fbeta_2 = fbeta_score(all_labels, all_preds, beta=2, zero_division=0)
        mcc = matthews_corrcoef(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0
        avg_precision = average_precision_score(all_labels, all_probs)
        brier = brier_score_loss(all_labels, all_probs)
        metrics = {
            "Loss": total_loss / len(dataloader),
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
            "Average_Precision": avg_precision,
            "Brier_Score": brier
        }
        return metrics

    def convert_to_native(self, data):
        """Converte recursivamente dicionários e listas para usar tipos nativos do Python."""
        if isinstance(data, dict):
            return {k: self.convert_to_native(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.convert_to_native(v) for v in data]
        elif isinstance(data, (np.float64, np.float32)):
            return float(data)
        elif isinstance(data, (np.int64, np.int32)):
            return int(data)
        else:
            return data

    def train(self, train_idx=None, val_idx=None, test_idx=None):
        """
        Treina a MLP e avalia as métricas para treino, validação e teste.
        O conjunto de teste permanece intocado até a última avaliação.
        Retorna a Loss de validação do último epoch para otimização.
        Se test_idx for None (modo CV), não são salvos arquivos nem criados DataFrames.
        """
        self.load_data(train_idx, val_idx, test_idx)
        model = MLPEmbeddingClassifier(self.input_dim).to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        epoch_metrics_log = []
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
            train_metrics = self.evaluate(model, self.train_loader)
            val_metrics = self.evaluate(model, self.val_loader)
            scheduler.step(val_metrics["Loss"])
            epoch_record = {"Epoch": epoch + 1}
            for key, value in train_metrics.items():
                epoch_record["Train " + key] = value
            for key, value in val_metrics.items():
                epoch_record["Validation " + key] = value
            epoch_metrics_log.append(epoch_record)
            logger.info(f"Epoch [{epoch+1}/{self.epochs}]")
            logger.info(f"  Treino -> Loss: {epoch_record['Train Loss']:.4f} - Acc: {epoch_record['Train Accuracy']:.4f} - Prec: {epoch_record['Train Precision']:.4f} - F1: {epoch_record['Train F1']:.4f} - AUC: {epoch_record['Train ROC_AUC']:.4f}")
            logger.info(f"  Validação -> Loss: {epoch_record['Validation Loss']:.4f} - Acc: {epoch_record['Validation Accuracy']:.4f} - Prec: {epoch_record['Validation Precision']:.4f} - F1: {epoch_record['Validation F1']:.4f} - AUC: {epoch_record['Validation ROC_AUC']:.4f}")
            if val_metrics["Loss"] < best_val_loss:
                best_val_loss = val_metrics["Loss"]
                epochs_without_improvement = 0
                best_model_state = model.state_dict()
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= self.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        # Se test_idx for None, assume modo de Cross-Validation e não salvar resultados extras.
        if self.test_loader is None:
            logger.info("Modo Cross-Validation: pulando salvamento de métricas e criação de DataFrames Spark.")
            torch.save(model.state_dict(), "mlp_model.pth")
            logger.info("✅ Modelo salvo em 'mlp_model.pth'.")
            return epoch_metrics_log[-1]["Validation Loss"]
        # Avaliação final no conjunto de teste (modo final)
        test_metrics = self.evaluate(model, self.test_loader)
        test_metrics_record = {"Test " + key: value for key, value in test_metrics.items()}
        logger.info("Avaliação final no conjunto de teste:")
        logger.info(f"  Test -> Loss: {test_metrics['Loss']:.4f} - Acc: {test_metrics['Accuracy']:.4f} - Prec: {test_metrics['Precision']:.4f} - F1: {test_metrics['F1']:.4f} - AUC: {test_metrics['ROC_AUC']:.4f}")
        epoch_metrics_log = self.convert_to_native(epoch_metrics_log)
        test_metrics_record = self.convert_to_native(test_metrics_record)
        final_metrics = {"epoch_metrics": epoch_metrics_log, "test_metrics": test_metrics_record}
        with open("training_metrics.json", "w") as f:
            json.dump(final_metrics, f, indent=4)
        logger.info("✅ Métricas de treino, validação e teste salvas em 'training_metrics.json'.")
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
        torch.save(model.state_dict(), "mlp_model.pth")
        logger.info("✅ Modelo salvo em 'mlp_model.pth'.")
        return epoch_metrics_log[-1]["Validation Loss"]

    def cross_validate(self, k=5):
        """
        Realiza validação cruzada (k-fold, com k=5 por padrão) sobre o conjunto de treino+validação.
        Retorna a média da Loss de validação dos k folds.
        """
        embeddings = np.load(self.embeddings_path, allow_pickle=True)
        labels = np.load(self.labels_path, allow_pickle=True).flatten()
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

# ---------------------------
# Função Objetivo para o Optuna
# ---------------------------
def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    epochs = trial.suggest_int('epochs', 1, 100)
    pipeline = MLPEmbeddingPipeline(
        embeddings_path="concatenated_embeddings/concatenated_embeddings_normalized.npy",
        labels_path="concatenated_embeddings/binary_labels.npy",
        batch_size=batch_size,
        lr=lr,
        epochs=epochs,
        early_stopping_patience=5
    )
    # Utiliza 5-fold cross-validation e retorna a média da Loss de validação
    avg_val_loss = pipeline.cross_validate(k=5)
    return avg_val_loss

# ---------------------------
# Execução do Estudo com o Optuna
# ---------------------------
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
    logger.info("Melhores hiperparâmetros: %s", study.best_params)
    logger.info("Melhor métrica: %s", study.best_value)
