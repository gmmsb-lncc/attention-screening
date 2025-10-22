#!/usr/bin/env python3
"""
Comparação de Classificadores DockTKinase
==========================================

Script para treinar e comparar múltiplos classificadores:
- Random Forest
- Multi-Layer Perceptron (MLP)
- Support Vector Machine (SVM)
- Gradient Boosting
- Logistic Regression
- K-Nearest Neighbors (KNN)
- XGBoost (se disponível)

Uso:
    python compare_classifiers.py --dataset human --model esm2_t6_8M_UR50D --max-samples 1000
"""

import sys
import argparse
from pathlib import Path
import time
import json
import warnings
from datetime import datetime

# Visualização
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo
import matplotlib.pyplot as plt
import seaborn as sns

# Suprimir warnings
warnings.filterwarnings('ignore')

# Adicionar paths
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR / 'src'))
sys.path.insert(0, str(ROOT_DIR / 'ESM'))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)

# Importar pipeline completo
from run_complete_pipeline import CompletePipeline


class ClassifierComparison:
    """Comparação de múltiplos classificadores"""
    
    def __init__(self, random_state=42, verbose=True):
        """
        Inicializar comparação
        
        Args:
            random_state: Seed para reprodutibilidade
            verbose: Mostrar logs detalhados
        """
        self.random_state = random_state
        self.verbose = verbose
        self.results = []
        
    def get_classifiers(self):
        """
        Definir todos os classificadores a testar
        
        Returns:
            Dict com {nome: (modelo, params)}
        """
        classifiers = {
            'RandomForest': (
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=20,
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbose=0
                ),
                {
                    'n_estimators': 100,
                    'max_depth': 20,
                    'criterion': 'gini'
                }
            ),
            
            'MLP_Small': (
                MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    activation='relu',
                    solver='adam',
                    max_iter=500,
                    random_state=self.random_state,
                    early_stopping=True,
                    validation_fraction=0.1,
                    verbose=0
                ),
                {
                    'hidden_layers': [128, 64],
                    'activation': 'relu',
                    'solver': 'adam',
                    'max_iter': 500
                }
            ),
            
            'MLP_Large': (
                MLPClassifier(
                    hidden_layer_sizes=(256, 128, 64),
                    activation='relu',
                    solver='adam',
                    max_iter=500,
                    random_state=self.random_state,
                    early_stopping=True,
                    validation_fraction=0.1,
                    verbose=0
                ),
                {
                    'hidden_layers': [256, 128, 64],
                    'activation': 'relu',
                    'solver': 'adam',
                    'max_iter': 500
                }
            ),
            
            'SVM_Linear': (
                SVC(
                    kernel='linear',
                    probability=True,
                    random_state=self.random_state,
                    verbose=0
                ),
                {
                    'kernel': 'linear',
                    'C': 1.0
                }
            ),
            
            'SVM_RBF': (
                SVC(
                    kernel='rbf',
                    probability=True,
                    random_state=self.random_state,
                    verbose=0
                ),
                {
                    'kernel': 'rbf',
                    'C': 1.0,
                    'gamma': 'scale'
                }
            ),
            
            'GradientBoosting': (
                GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=self.random_state,
                    verbose=0
                ),
                {
                    'n_estimators': 100,
                    'max_depth': 5,
                    'learning_rate': 0.1
                }
            ),
            
            'LogisticRegression': (
                LogisticRegression(
                    max_iter=1000,
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                {
                    'max_iter': 1000,
                    'penalty': 'l2',
                    'solver': 'lbfgs'
                }
            ),
            
        }
        
        # Tentar adicionar KNN (pode falhar devido a bug do threadpoolctl)
        try:
            # Teste mais robusto para ver se KNN funciona
            test_knn = KNeighborsClassifier(n_neighbors=3)
            test_X = np.random.rand(50, 320)  # Usar dimensões realistas
            test_y = np.random.randint(0, 2, 50)
            test_knn.fit(test_X, test_y)
            # Testar predict_proba também
            test_proba = test_knn.predict_proba(test_X[:5])
            
            # Se funcionar, adicionar ao dict
            classifiers['KNN'] = (
                KNeighborsClassifier(
                    n_neighbors=5
                ),
                {
                    'n_neighbors': 5,
                    'weights': 'uniform',
                    'metric': 'euclidean'
                }
            )
            if self.verbose:
                print('   ✅ KNN disponível')
        except Exception as e:
            if self.verbose:
                print(f'   ⚠️  KNN não disponível (erro: {type(e).__name__})')
                print(f'       Solução: pip install -U threadpoolctl scikit-learn')
        
        # Tentar adicionar XGBoost se disponível
        try:
            import xgboost as xgb
            classifiers['XGBoost'] = (
                xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=self.random_state,
                    verbosity=0,
                    n_jobs=-1
                ),
                {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1
                }
            )
            if self.verbose:
                print('   ✅ XGBoost disponível')
        except ImportError:
            if self.verbose:
                print('   ⚠️  XGBoost não disponível (instale: pip install xgboost)')
        
        return classifiers
    
    def train_and_evaluate(self, clf, clf_name, clf_params, X_train, y_train, X_val, y_val, X_test, y_test):
        """
        Treinar e avaliar um classificador
        
        Args:
            clf: Modelo a treinar
            clf_name: Nome do modelo
            clf_params: Hiperparâmetros
            X_train, y_train: Dados de treino
            X_val, y_val: Dados de validação
            X_test, y_test: Dados de teste
            
        Returns:
            Dict com resultados
        """
        if self.verbose:
            print(f'\n{"="*60}')
            print(f'🤖 Treinando: {clf_name}')
            print(f'{"="*60}')
        
        result = {
            'name': clf_name,
            'params': clf_params,
            'start_time': datetime.now().isoformat()
        }
        
        try:
            # Treinar
            start_time = time.time()
            clf.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            result['train_time'] = train_time
            result['train_accuracy'] = float(clf.score(X_train, y_train))
            
            if self.verbose:
                print(f'   ✅ Treinado em {train_time:.2f}s')
                print(f'   📊 Acurácia Treino: {result["train_accuracy"]:.4f}')
            
            # Avaliar em TREINO (para ver se está aprendendo)
            train_metrics = self._evaluate(clf, X_train, y_train, 'Train')
            result['train'] = train_metrics
            
            # Avaliar em VALIDAÇÃO
            val_metrics = self._evaluate(clf, X_val, y_val, 'Validation')
            result['validation'] = val_metrics
            
            # Avaliar em TESTE (nunca visto!)
            test_metrics = self._evaluate(clf, X_test, y_test, 'Test')
            result['test'] = test_metrics
            
            result['status'] = 'success'
            result['end_time'] = datetime.now().isoformat()
            
            if self.verbose:
                print(f'\n   📊 RESUMO {clf_name}:')
                print(f'      Treino:    F1={train_metrics["f1"]:.4f}, Acc={train_metrics["accuracy"]:.4f}, Prec={train_metrics["precision"]:.4f}')
                print(f'      Validação: F1={val_metrics["f1"]:.4f}, Acc={val_metrics["accuracy"]:.4f}, Prec={val_metrics["precision"]:.4f}')
                print(f'      Teste:     F1={test_metrics["f1"]:.4f}, Acc={test_metrics["accuracy"]:.4f}, Prec={test_metrics["precision"]:.4f}')
        
        except Exception as e:
            result['status'] = 'failed'
            result['error'] = str(e)
            result['error_type'] = type(e).__name__
            result['end_time'] = datetime.now().isoformat()
            
            if self.verbose:
                print(f'   ❌ ERRO ({type(e).__name__}): {e}')
                # Mostrar traceback para debug
                import traceback
                print(f'   📋 Detalhes:')
                for line in traceback.format_exc().split('\n')[-4:-1]:
                    if line.strip():
                        print(f'      {line.strip()}')
        
        return result
    
    def _evaluate(self, clf, X, y, dataset_name):
        """Avaliar modelo em um conjunto"""
        start_time = time.time()
        
        # Predições
        y_pred = clf.predict(X)
        y_proba = clf.predict_proba(X)
        
        # Métricas
        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, y_pred, average='weighted', zero_division=0
        )
        
        # ROC AUC
        if len(np.unique(y)) == 2:
            roc_auc = roc_auc_score(y, y_proba[:, 1])
            avg_precision = average_precision_score(y, y_proba[:, 1])
        else:
            roc_auc = roc_auc_score(y, y_proba, multi_class='ovr', average='weighted')
            avg_precision = None
        
        eval_time = time.time() - start_time
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'roc_auc': float(roc_auc),
            'avg_precision': float(avg_precision) if avg_precision else None,
            'eval_time': eval_time
        }
    
    def compare(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """
        Comparar todos os classificadores
        
        Args:
            X_train, y_train: Dados de treino
            X_val, y_val: Dados de validação
            X_test, y_test: Dados de teste
            
        Returns:
            List de resultados, nome do melhor modelo
        """
        if self.verbose:
            print('\n' + '='*60)
            print('🏆 COMPARAÇÃO DE CLASSIFICADORES')
            print('='*60)
        
        classifiers = self.get_classifiers()
        
        if self.verbose:
            print(f'\n📋 Modelos a testar: {len(classifiers)}')
            for name in classifiers.keys():
                print(f'   • {name}')
        
        # Treinar e avaliar cada modelo
        results = []
        for clf_name, (clf, params) in classifiers.items():
            result = self.train_and_evaluate(
                clf, clf_name, params,
                X_train, y_train, X_val, y_val, X_test, y_test
            )
            results.append(result)
        
        # Encontrar melhor modelo (baseado em F1-score de validação)
        successful_results = [r for r in results if r['status'] == 'success']
        
        if not successful_results:
            if self.verbose:
                print('\n❌ Nenhum modelo foi treinado com sucesso!')
            return results, None
        
        best_model = max(successful_results, key=lambda r: r['validation']['f1'])
        
        if self.verbose:
            print('\n' + '='*100)
            print('🏆 RANKING DOS MODELOS (ordenado por F1-Score Validação)')
            print('='*100)
            
            # Ordenar por F1 de validação
            sorted_results = sorted(
                successful_results,
                key=lambda r: r['validation']['f1'],
                reverse=True
            )
            
            # Cabeçalho da tabela
            print(f'\n{"#":<3} {"Modelo":<20} {"Conjunto":<12} {"F1":>7} {"Acurácia":>9} {"Precisão":>9} {"Recall":>7} {"ROC-AUC":>8} {"Tempo":>8}')
            print('-'*100)
            
            for i, result in enumerate(sorted_results, 1):
                name = result['name']
                train_time = result['train_time']
                
                medal = '🥇' if i == 1 else '🥈' if i == 2 else '🥉' if i == 3 else f'{i}.'
                
                # Treino
                print(f'{medal:<3} {name:<20} {"Treino":<12} '
                      f'{result["train"]["f1"]:>7.4f} '
                      f'{result["train"]["accuracy"]:>9.4f} '
                      f'{result["train"]["precision"]:>9.4f} '
                      f'{result["train"]["recall"]:>7.4f} '
                      f'{result["train"]["roc_auc"]:>8.4f} '
                      f'{train_time:>7.2f}s')
                
                # Validação
                print(f'{"":3} {"":20} {"Validação":<12} '
                      f'{result["validation"]["f1"]:>7.4f} '
                      f'{result["validation"]["accuracy"]:>9.4f} '
                      f'{result["validation"]["precision"]:>9.4f} '
                      f'{result["validation"]["recall"]:>7.4f} '
                      f'{result["validation"]["roc_auc"]:>8.4f}')
                
                # Teste
                print(f'{"":3} {"":20} {"Teste":<12} '
                      f'{result["test"]["f1"]:>7.4f} '
                      f'{result["test"]["accuracy"]:>9.4f} '
                      f'{result["test"]["precision"]:>9.4f} '
                      f'{result["test"]["recall"]:>7.4f} '
                      f'{result["test"]["roc_auc"]:>8.4f}')
                
                if i < len(sorted_results):  # Não adicionar linha após o último
                    print()
            
            print('-'*100)
            print(f'\n{"🏆 MELHOR MODELO":^100}')
            print('='*100)
            print(f'Modelo: {best_model["name"]}')
            print(f'Tempo de Treinamento: {best_model["train_time"]:.2f}s')
            print()
            print(f'{"Conjunto":<12} {"F1":>7} {"Acurácia":>9} {"Precisão":>9} {"Recall":>7} {"ROC-AUC":>8}')
            print('-'*100)
            print(f'{"Treino":<12} '
                  f'{best_model["train"]["f1"]:>7.4f} '
                  f'{best_model["train"]["accuracy"]:>9.4f} '
                  f'{best_model["train"]["precision"]:>9.4f} '
                  f'{best_model["train"]["recall"]:>7.4f} '
                  f'{best_model["train"]["roc_auc"]:>8.4f}')
            print(f'{"Validação":<12} '
                  f'{best_model["validation"]["f1"]:>7.4f} '
                  f'{best_model["validation"]["accuracy"]:>9.4f} '
                  f'{best_model["validation"]["precision"]:>9.4f} '
                  f'{best_model["validation"]["recall"]:>7.4f} '
                  f'{best_model["validation"]["roc_auc"]:>8.4f}')
            print(f'{"Teste":<12} '
                  f'{best_model["test"]["f1"]:>7.4f} '
                  f'{best_model["test"]["accuracy"]:>9.4f} '
                  f'{best_model["test"]["precision"]:>9.4f} '
                  f'{best_model["test"]["recall"]:>7.4f} '
                  f'{best_model["test"]["roc_auc"]:>8.4f}')
            print('='*100)
        
        self.results = results
        return results, best_model['name']
    
    def save_results(self, results, output_dir):
        """Salvar resultados da comparação"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Salvar JSON completo
        results_file = output_dir / 'classifier_comparison.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'random_state': self.random_state,
                'results': results,
                'summary': {
                    'total_models': len(results),
                    'successful': len([r for r in results if r['status'] == 'success']),
                    'failed': len([r for r in results if r['status'] == 'failed'])
                }
            }, f, indent=2, ensure_ascii=False)
        
        # Criar tabela comparativa detalhada
        comparison_file = output_dir / 'comparison_table.txt'
        with open(comparison_file, 'w', encoding='utf-8') as f:
            f.write('='*100 + '\n')
            f.write(' '*35 + 'COMPARAÇÃO DE CLASSIFICADORES\n')
            f.write('='*100 + '\n\n')
            f.write(f'Data: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write(f'Random State: {self.random_state}\n\n')
            
            # Dados
            successful = [r for r in results if r['status'] == 'success']
            failed = [r for r in results if r['status'] == 'failed']
            sorted_results = sorted(successful, key=lambda r: r['validation']['f1'], reverse=True)
            
            f.write(f'Total de modelos: {len(results)}\n')
            f.write(f'  ✓ Bem-sucedidos: {len(successful)}\n')
            f.write(f'  ✗ Falharam: {len(failed)}\n\n')
            
            if failed:
                f.write('Modelos que falharam:\n')
                for r in failed:
                    f.write(f'  • {r["name"]}: {r.get("error_type", "Unknown error")}\n')
                f.write('\n')
            
            # Tabela de resultados
            f.write('='*100 + '\n')
            f.write('RANKING POR F1-SCORE (VALIDAÇÃO)\n')
            f.write('='*100 + '\n\n')
            
            # Cabeçalho
            f.write(f'{"#":<3} {"Modelo":<20} {"Conjunto":<12} {"F1":>7} {"Acurácia":>9} {"Precisão":>9} {"Recall":>7} {"ROC-AUC":>8} {"Tempo":>8}\n')
            f.write('-'*100 + '\n')
            
            for i, result in enumerate(sorted_results, 1):
                name = result['name'][:19]
                train_time = result['train_time']
                
                # Treino
                f.write(f'{i:<3} {name:<20} {"Treino":<12} '
                        f'{result["train"]["f1"]:>7.4f} '
                        f'{result["train"]["accuracy"]:>9.4f} '
                        f'{result["train"]["precision"]:>9.4f} '
                        f'{result["train"]["recall"]:>7.4f} '
                        f'{result["train"]["roc_auc"]:>8.4f} '
                        f'{train_time:>7.2f}s\n')
                
                # Validação
                f.write(f'{"":3} {"":20} {"Validação":<12} '
                        f'{result["validation"]["f1"]:>7.4f} '
                        f'{result["validation"]["accuracy"]:>9.4f} '
                        f'{result["validation"]["precision"]:>9.4f} '
                        f'{result["validation"]["recall"]:>7.4f} '
                        f'{result["validation"]["roc_auc"]:>8.4f}\n')
                
                # Teste
                f.write(f'{"":3} {"":20} {"Teste":<12} '
                        f'{result["test"]["f1"]:>7.4f} '
                        f'{result["test"]["accuracy"]:>9.4f} '
                        f'{result["test"]["precision"]:>9.4f} '
                        f'{result["test"]["recall"]:>7.4f} '
                        f'{result["test"]["roc_auc"]:>8.4f}\n')
                
                if i < len(sorted_results):
                    f.write('\n')
            
            # Melhor modelo
            if sorted_results:
                f.write('\n' + '='*100 + '\n')
                f.write(f'{"🏆 MELHOR MODELO":^100}\n')
                f.write('='*100 + '\n')
                best = sorted_results[0]
                f.write(f'Modelo: {best["name"]}\n')
                f.write(f'Tempo de Treinamento: {best["train_time"]:.2f}s\n\n')
                
                f.write(f'{"Conjunto":<12} {"F1":>7} {"Acurácia":>9} {"Precisão":>9} {"Recall":>7} {"ROC-AUC":>8}\n')
                f.write('-'*100 + '\n')
                f.write(f'{"Treino":<12} '
                        f'{best["train"]["f1"]:>7.4f} '
                        f'{best["train"]["accuracy"]:>9.4f} '
                        f'{best["train"]["precision"]:>9.4f} '
                        f'{best["train"]["recall"]:>7.4f} '
                        f'{best["train"]["roc_auc"]:>8.4f}\n')
                f.write(f'{"Validação":<12} '
                        f'{best["validation"]["f1"]:>7.4f} '
                        f'{best["validation"]["accuracy"]:>9.4f} '
                        f'{best["validation"]["precision"]:>9.4f} '
                        f'{best["validation"]["recall"]:>7.4f} '
                        f'{best["validation"]["roc_auc"]:>8.4f}\n')
                f.write(f'{"Teste":<12} '
                        f'{best["test"]["f1"]:>7.4f} '
                        f'{best["test"]["accuracy"]:>9.4f} '
                        f'{best["test"]["precision"]:>9.4f} '
                        f'{best["test"]["recall"]:>7.4f} '
                        f'{best["test"]["roc_auc"]:>8.4f}\n')
                f.write('='*100 + '\n')
        
        # Gerar visualizações
        self.plot_comparison(results, output_dir)
        
        if self.verbose:
            print(f'\n💾 Resultados salvos em: {output_dir}')
            print(f'   📄 JSON detalhado: classifier_comparison.json')
            print(f'   📊 Tabela resumo:  comparison_table.txt')
            print(f'   📈 Visualizações:  comparison_*.png')
    
    def plot_comparison(self, results, output_dir):
        """
        Criar visualizações da comparação de modelos
        
        Args:
            results: Lista de resultados dos modelos
            output_dir: Diretório para salvar as visualizações
        """
        try:
            output_dir = Path(output_dir)
            
            # Filtrar apenas modelos bem-sucedidos
            successful = [r for r in results if r['status'] == 'success']
            if not successful:
                return
            
            # Ordenar por F1 de validação
            successful = sorted(successful, key=lambda r: r['validation']['f1'], reverse=True)
            
            # Configurar estilo
            sns.set_style("whitegrid")
            plt.rcParams['figure.facecolor'] = 'white'
            
            # ===== FIGURA 1: Comparação de Métricas =====
            fig1 = plt.figure(figsize=(16, 10))
            
            model_names = [r['name'] for r in successful]
            
            metrics = ['f1', 'accuracy', 'precision', 'recall', 'roc_auc']
            metric_labels = ['F1-Score', 'Acurácia', 'Precisão', 'Recall', 'ROC-AUC']
            
            for idx, (metric, label) in enumerate(zip(metrics, metric_labels), 1):
                ax = plt.subplot(2, 3, idx)
                
                train_vals = [r['train'][metric] for r in successful]
                val_vals = [r['validation'][metric] for r in successful]
                test_vals = [r['test'][metric] for r in successful]
                
                x = np.arange(len(model_names))
                width = 0.25
                
                ax.bar(x - width, train_vals, width, label='Train', color='#3498db', alpha=0.8)
                ax.bar(x, val_vals, width, label='Validation', color='#f39c12', alpha=0.8)
                ax.bar(x + width, test_vals, width, label='Test', color='#e74c3c', alpha=0.8)
                
                ax.set_ylabel(label, fontsize=11, fontweight='bold')
                ax.set_title(f'Comparação: {label}', fontsize=12, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
                ax.legend(loc='lower right', fontsize=9)
                ax.set_ylim([0, 1.0])
                ax.grid(axis='y', alpha=0.3)
            
            # Subplot 6: Tempo de Treinamento
            ax = plt.subplot(2, 3, 6)
            times = [r['train_time'] for r in successful]
            bars = ax.barh(model_names, times, color='#9b59b6', alpha=0.8)
            ax.set_xlabel('Tempo (segundos)', fontsize=11, fontweight='bold')
            ax.set_title('Tempo de Treinamento', fontsize=12, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Adicionar valores nas barras
            for bar, time_val in zip(bars, times):
                ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                       f' {time_val:.2f}s',
                       ha='left', va='center', fontsize=9, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'comparison_metrics.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # ===== FIGURA 2: Ranking =====
            fig2 = plt.figure(figsize=(12, 8))
            
            # Comparação F1-Score (principal métrica)
            val_f1 = [r['validation']['f1'] for r in successful]
            test_f1 = [r['test']['f1'] for r in successful]
            
            y_pos = np.arange(len(model_names))
            
            plt.barh(y_pos - 0.2, val_f1, 0.4, label='Validation F1', 
                    color='#f39c12', alpha=0.8)
            plt.barh(y_pos + 0.2, test_f1, 0.4, label='Test F1', 
                    color='#e74c3c', alpha=0.8)
            
            plt.xlabel('F1-Score', fontsize=12, fontweight='bold')
            plt.ylabel('Modelo', fontsize=12, fontweight='bold')
            plt.title('Ranking de Modelos por F1-Score', fontsize=14, fontweight='bold')
            plt.yticks(y_pos, model_names)
            plt.legend(loc='lower right', fontsize=10)
            plt.xlim([0, 1.0])
            plt.grid(axis='x', alpha=0.3)
            
            # Adicionar medalhas para top 3
            for i in range(min(3, len(model_names))):
                medal = '🥇' if i == 0 else '🥈' if i == 1 else '🥉'
                plt.text(-0.05, i, medal, fontsize=20, ha='right', va='center')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'comparison_ranking.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # ===== FIGURA 3: Overfitting Analysis =====
            fig3 = plt.figure(figsize=(14, 6))
            
            # Subplot 1: Train vs Validation
            ax1 = plt.subplot(1, 2, 1)
            train_f1 = [r['train']['f1'] for r in successful]
            val_f1 = [r['validation']['f1'] for r in successful]
            
            ax1.scatter(train_f1, val_f1, s=100, alpha=0.6, c=range(len(successful)), 
                       cmap='viridis')
            ax1.plot([0, 1], [0, 1], 'r--', lw=2, alpha=0.5, label='Ideal (sem overfitting)')
            
            for i, name in enumerate(model_names):
                ax1.annotate(name, (train_f1[i], val_f1[i]), fontsize=8, 
                           xytext=(5, 5), textcoords='offset points')
            
            ax1.set_xlabel('F1-Score Train', fontsize=11, fontweight='bold')
            ax1.set_ylabel('F1-Score Validation', fontsize=11, fontweight='bold')
            ax1.set_title('Análise de Overfitting (Train vs Validation)', 
                         fontsize=12, fontweight='bold')
            ax1.legend()
            ax1.grid(alpha=0.3)
            ax1.set_xlim([0, 1])
            ax1.set_ylim([0, 1])
            
            # Subplot 2: Validation vs Test
            ax2 = plt.subplot(1, 2, 2)
            test_f1 = [r['test']['f1'] for r in successful]
            
            ax2.scatter(val_f1, test_f1, s=100, alpha=0.6, c=range(len(successful)), 
                       cmap='viridis')
            ax2.plot([0, 1], [0, 1], 'r--', lw=2, alpha=0.5, label='Ideal (boa generalização)')
            
            for i, name in enumerate(model_names):
                ax2.annotate(name, (val_f1[i], test_f1[i]), fontsize=8,
                           xytext=(5, 5), textcoords='offset points')
            
            ax2.set_xlabel('F1-Score Validation', fontsize=11, fontweight='bold')
            ax2.set_ylabel('F1-Score Test', fontsize=11, fontweight='bold')
            ax2.set_title('Análise de Generalização (Validation vs Test)', 
                         fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(alpha=0.3)
            ax2.set_xlim([0, 1])
            ax2.set_ylim([0, 1])
            
            plt.tight_layout()
            plt.savefig(output_dir / 'comparison_overfitting.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            if self.verbose:
                print(f'   📊 Visualizações geradas com sucesso!')
        
        except Exception as e:
            if self.verbose:
                print(f'   ⚠️  Erro ao gerar visualizações: {e}')
                import traceback
                traceback.print_exc()


def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description='Comparar múltiplos classificadores',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Comparar todos os modelos no dataset humano
  python compare_classifiers.py --dataset human --max-samples 1000
  
  # Usar modelo ESM-2 grande
  python compare_classifiers.py --dataset human --model esm2_t33_650M_UR50D
  
  # Com GPU
  python compare_classifiers.py --dataset all --device cuda
        """
    )
    
    parser.add_argument('--dataset', type=str, default='human',
                       choices=['human', 'non_human', 'all'])
    parser.add_argument('--model', type=str, default='esm2_t6_8M_UR50D')
    parser.add_argument('--val-size', type=float, default=0.1)
    parser.add_argument('--test-size', type=float, default=0.1)
    parser.add_argument('--label-method', type=str, default='auto',
                       choices=['pchembl', 'ic50', 'ki', 'kd', 'auto'])
    parser.add_argument('--label-threshold', type=float, default=None)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--device', type=str, default='auto',
                       choices=['cpu', 'cuda', 'auto'])
    parser.add_argument('--output-dir', type=str, default='tests/comparison_output',
                       help='Diretório para salvar resultados (default: tests/comparison_output)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quiet', action='store_true')
    
    args = parser.parse_args()
    
    # ETAPA 1: Carregar dados e gerar embeddings usando pipeline existente
    print('='*60)
    print('📊 ETAPA 1: Preparação dos Dados')
    print('='*60)
    
    pipeline = CompletePipeline(
        dataset_name=args.dataset,
        esm_model=args.model,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.seed,
        max_samples=args.max_samples,
        device=args.device,
        output_dir=args.output_dir,
        label_method=args.label_method,
        label_threshold=args.label_threshold,
        verbose=not args.quiet
    )
    
    # Carregar dataset
    df = pipeline.load_dataset()
    
    # Criar labels
    y, df = pipeline.create_labels(df)
    
    # Gerar embeddings
    X = pipeline.generate_embeddings(df, batch_size=8)
    
    # Split estratificado
    X_train, X_val, X_test, y_train, y_val, y_test = pipeline.stratified_split(X, y)
    
    # ETAPA 2: Comparar classificadores
    print('\n' + '='*60)
    print('📊 ETAPA 2: Comparação de Classificadores')
    print('='*60)
    
    comparison = ClassifierComparison(
        random_state=args.seed,
        verbose=not args.quiet
    )
    
    results, best_model = comparison.compare(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    
    # ETAPA 3: Salvar resultados
    comparison.save_results(results, args.output_dir)
    
    print('\n✅ Comparação concluída!')
    if best_model:
        print(f'🏆 Melhor modelo: {best_model}')


if __name__ == '__main__':
    main()
