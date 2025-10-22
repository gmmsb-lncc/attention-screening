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
                print(f'      Val Acc:  {val_metrics["accuracy"]:.4f}')
                print(f'      Val F1:   {val_metrics["f1"]:.4f}')
                print(f'      Test Acc: {test_metrics["accuracy"]:.4f}')
                print(f'      Test F1:  {test_metrics["f1"]:.4f}')
        
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
            print('\n' + '='*60)
            print('🏆 RANKING DOS MODELOS (por F1-Score Validação)')
            print('='*60)
            
            # Ordenar por F1 de validação
            sorted_results = sorted(
                successful_results,
                key=lambda r: r['validation']['f1'],
                reverse=True
            )
            
            for i, result in enumerate(sorted_results, 1):
                name = result['name']
                val_f1 = result['validation']['f1']
                test_f1 = result['test']['f1']
                val_acc = result['validation']['accuracy']
                test_acc = result['test']['accuracy']
                train_time = result['train_time']
                
                medal = '🥇' if i == 1 else '🥈' if i == 2 else '🥉' if i == 3 else f'{i}.'
                
                print(f'\n{medal} {name}')
                print(f'   Val:  F1={val_f1:.4f}, Acc={val_acc:.4f}')
                print(f'   Test: F1={test_f1:.4f}, Acc={test_acc:.4f}')
                print(f'   Time: {train_time:.2f}s')
            
            print('\n' + '='*60)
            print(f'🏆 MELHOR MODELO: {best_model["name"]}')
            print('='*60)
            print(f'   Val F1:   {best_model["validation"]["f1"]:.4f}')
            print(f'   Val Acc:  {best_model["validation"]["accuracy"]:.4f}')
            print(f'   Test F1:  {best_model["test"]["f1"]:.4f}')
            print(f'   Test Acc: {best_model["test"]["accuracy"]:.4f}')
            print('='*60)
        
        self.results = results
        return results, best_model['name']
    
    def save_results(self, results, output_dir):
        """Salvar resultados da comparação"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Salvar JSON completo
        results_file = output_dir / 'classifier_comparison.json'
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'random_state': self.random_state,
                'results': results
            }, f, indent=2)
        
        # Criar tabela comparativa
        comparison_file = output_dir / 'comparison_table.txt'
        with open(comparison_file, 'w') as f:
            f.write('COMPARAÇÃO DE CLASSIFICADORES\n')
            f.write('='*80 + '\n\n')
            
            # Cabeçalho
            f.write(f'{"Modelo":<20} {"Val F1":>8} {"Val Acc":>8} {"Test F1":>8} {"Test Acc":>8} {"Tempo":>8}\n')
            f.write('-'*80 + '\n')
            
            # Dados
            successful = [r for r in results if r['status'] == 'success']
            sorted_results = sorted(successful, key=lambda r: r['validation']['f1'], reverse=True)
            
            for result in sorted_results:
                name = result['name'][:19]
                val_f1 = result['validation']['f1']
                val_acc = result['validation']['accuracy']
                test_f1 = result['test']['f1']
                test_acc = result['test']['accuracy']
                train_time = result['train_time']
                
                f.write(f'{name:<20} {val_f1:>8.4f} {val_acc:>8.4f} {test_f1:>8.4f} {test_acc:>8.4f} {train_time:>7.2f}s\n')
        
        if self.verbose:
            print(f'\n💾 Resultados salvos:')
            print(f'   • {results_file}')
            print(f'   • {comparison_file}')


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
