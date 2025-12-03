#!/usr/bin/env python3
"""
Treinador de Modelos de Regressão - DockTKinase
================================================

Gerencia treinamento de múltiplos modelos com validação.
"""

import time
import warnings
import numpy as np
from pathlib import Path
import joblib

# Suppress harmless sklearn/LGBM warnings
# Feature names warning (happens when training with DataFrame but predicting with numpy array)
warnings.filterwarnings('ignore', message='X does not have valid feature names')
warnings.filterwarnings('ignore', message='.*was fitted with feature names.*')
# Convergence warnings (LinearSVC may not converge with default iterations)
warnings.filterwarnings('ignore', category=UserWarning, message='.*Liblinear failed to converge.*')
warnings.filterwarnings('ignore', message='.*ConvergenceWarning.*')
# Suppress scipy ConstantInputWarning
warnings.filterwarnings('ignore', message='An input array is constant')

# Import relativo corrigido
try:
    from ..models import RegressionModels
except ImportError:
    from regression.models import RegressionModels

from .evaluator import RegressionEvaluator


class RegressionTrainer:
    """
    Treina múltiplos modelos de regressão e compara resultados.
    
    Suporta treinamento paralelo e seleção automática do melhor modelo.
    """
    
    def __init__(self, models_dict=None, device='auto', verbose=True, random_state=42):
        """
        Inicializar trainer.
        
        Args:
            models_dict: Dict {nome: modelo} ou None para usar todos
            device: Device para modelos que suportam GPU
            verbose: Mostrar progresso
            random_state: Seed para reprodutibilidade
        """
        if models_dict is None:
            models_dict = RegressionModels.get_all_models(random_state=random_state)
        
        self.models = models_dict
        self.device = device
        self.verbose = verbose
        self.random_state = random_state
        
        # Resultados
        self.trained_models = {}
        self.train_results = {}
        self.val_results = {}
        self.test_results = {}
        self.training_times = {}
    
    def train_all(self, X_train, y_train, X_val, y_val):
        """
        Treina todos os modelos e valida.
        
        Args:
            X_train: Features de treino
            y_train: Targets de treino
            X_val: Features de validação
            y_val: Targets de validação
            
        Returns:
            Dict com métricas de validação de cada modelo
        """
        if self.verbose:
            print('🤖 TREINAMENTO DE MODELOS DE REGRESSÃO')
            print('='*60)
            print(f'   Modelos a treinar: {len(self.models)}')
            print(f'   Amostras treino: {len(X_train):,}')
            print(f'   Amostras validação: {len(X_val):,}')
            print('='*60)
            print()
        
        for model_name in self.models.keys():
            if self.verbose:
                print(f'🔄 Treinando {model_name}...')
            
            self.train_single(model_name, X_train, y_train, X_val, y_val)
        
        if self.verbose:
            print()
            print('='*60)
            print('✅ TREINAMENTO E VALIDAÇÃO COMPLETOS!')
            print('='*60)
            print()
        
        return self.val_results
    
    def train_single(self, model_name, X_train, y_train, X_val, y_val):
        """
        Treina um modelo específico.
        
        Args:
            model_name: Nome do modelo
            X_train, y_train: Dados de treino
            X_val, y_val: Dados de validação
        """
        if model_name not in self.models:
            raise ValueError(f'Modelo "{model_name}" não encontrado')
        
        model = self.models[model_name]
        
        # Treinar
        start_time = time.time()
        
        try:
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Predições
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            
            # Métricas
            train_metrics = RegressionEvaluator.calculate_metrics(
                y_train, y_train_pred, model_name
            )
            val_metrics = RegressionEvaluator.calculate_metrics(
                y_val, y_val_pred, model_name
            )
            
            # Salvar resultados
            self.trained_models[model_name] = model
            self.train_results[model_name] = train_metrics
            self.val_results[model_name] = val_metrics
            self.training_times[model_name] = training_time
            
            if self.verbose:
                print(f'   ✅ {model_name} - Treino: MAE={train_metrics["MAE"]:.2f} | '
                      f'Val: MAE={val_metrics["MAE"]:.2f} R²={val_metrics["R2"]:.4f} | '
                      f'Tempo: {training_time:.2f}s')
        
        except Exception as e:
            if self.verbose:
                print(f'   ❌ {model_name} - ERRO: {str(e)}')
            self.trained_models[model_name] = None
            self.train_results[model_name] = None
            self.val_results[model_name] = None
            self.training_times[model_name] = None
    
    def evaluate_on_test(self, X_test, y_test):
        """
        Avalia todos os modelos treinados no conjunto de teste.
        
        Args:
            X_test: Features de teste
            y_test: Targets de teste
            
        Returns:
            Dict com métricas de teste de cada modelo
        """
        if not self.trained_models:
            raise RuntimeError('Nenhum modelo foi treinado ainda!')
        
        if self.verbose:
            print()
            print('='*60)
            print('📊 AVALIAÇÃO NO CONJUNTO DE TESTE')
            print('='*60)
            print(f'   Amostras teste: {len(X_test):,}')
            print('='*60)
            print()
        
        for model_name, model in self.trained_models.items():
            if model is None:
                continue
            
            try:
                # Predição
                y_test_pred = model.predict(X_test)
                
                # Métricas
                test_metrics = RegressionEvaluator.calculate_metrics(
                    y_test, y_test_pred, model_name
                )
                
                self.test_results[model_name] = test_metrics
                
                if self.verbose:
                    print(f'   {model_name}: MAE={test_metrics["MAE"]:.2f} '
                          f'RMSE={test_metrics["RMSE"]:.2f} R²={test_metrics["R2"]:.4f}')
            
            except Exception as e:
                if self.verbose:
                    print(f'   ❌ {model_name} - ERRO: {str(e)}')
                self.test_results[model_name] = None
        
        if self.verbose:
            print()
            print('='*60)
            print('✅ AVALIAÇÃO NO CONJUNTO DE TESTE COMPLETA!')
            print('='*60)
            
            # Mostrar ranking no teste
            self._print_test_ranking()
        
        return self.test_results
    
    def get_best_model(self, metric='MAE', dataset='val'):
        """
        Retorna o melhor modelo baseado na métrica.
        
        Args:
            metric: Métrica para seleção (MAE, RMSE, R2, etc)
            dataset: Dataset para comparar ('train', 'val', 'test')
            
        Returns:
            Tuple (nome_modelo, modelo, métricas)
        """
        if dataset == 'train':
            results = self.train_results
        elif dataset == 'val':
            results = self.val_results
        elif dataset == 'test':
            results = self.test_results
        else:
            raise ValueError(f'Dataset inválido: {dataset}')
        
        # Filtrar modelos válidos
        valid_results = {k: v for k, v in results.items() if v is not None}
        
        if not valid_results:
            raise RuntimeError(f'Nenhum resultado disponível para dataset "{dataset}"')
        
        # Determinar se menor é melhor
        ascending = metric not in ['R2']  # R² maior é melhor
        
        # Encontrar melhor
        best_name = RegressionEvaluator.get_best_model(
            valid_results, metric=metric, ascending=ascending
        )
        
        return best_name, self.trained_models[best_name], valid_results[best_name]
    
    def save_models(self, output_dir, save_all=True):
        """
        Salva modelos treinados.
        
        Args:
            output_dir: Diretório para salvar
            save_all: Se True, salva todos. Se False, salva apenas o melhor.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if save_all:
            # Salvar todos os modelos
            for model_name, model in self.trained_models.items():
                if model is not None:
                    model_path = output_dir / f'{model_name}.pkl'
                    joblib.dump(model, model_path)
                    
                    if self.verbose:
                        print(f'   💾 Salvo: {model_path.name}')
        
        # Salvar melhor modelo
        try:
            best_name, best_model, _ = self.get_best_model(metric='MAE', dataset='val')
            best_filename = f'{best_name}_best_model.pkl'
            best_path = output_dir / best_filename
            joblib.dump(best_model, best_path)
            
            if self.verbose:
                print(f'   ⭐ Melhor modelo salvo: {best_name} → {best_filename}')
        except Exception as e:
            if self.verbose:
                print(f'   ⚠️  Erro ao salvar melhor modelo: {e}')
    
    def _print_test_ranking(self):
        """Imprime ranking de modelos baseado no teste (formato tabela)."""
        print('\n📊 RESUMO DOS RESULTADOS (Conjunto de Teste)')
        print('=' * 80)
        
        df = RegressionEvaluator.compare_models(self.test_results, metric='MAE')
        
        # Cabeçalho
        header = f"{'Modelo':<20} {'MAE':>12} {'RMSE':>12} {'R²':>10}"
        print(header)
        print('-' * 80)
        
        for idx, (model_name, row) in enumerate(df.iterrows(), 1):
            mae = row['MAE'] if row['MAE'] is not None else float('inf')
            r2 = row['R2'] if row['R2'] is not None else float('-inf')
            rmse = row['RMSE'] if row['RMSE'] is not None else float('inf')
            
            # Formatar valores com tratamento para inf
            mae_str = f'{mae:>12.2f}' if mae != float('inf') else '         N/A'
            rmse_str = f'{rmse:>12.2f}' if rmse != float('inf') else '         N/A'
            r2_str = f'{r2:>10.4f}' if r2 != float('-inf') else '       N/A'
            
            row_str = f"{model_name:<20} {mae_str} {rmse_str} {r2_str}"
            
            # Destacar top 3
            if idx == 1:
                print(f'🥇 {row_str}')
            elif idx == 2:
                print(f'🥈 {row_str}')
            elif idx == 3:
                print(f'🥉 {row_str}')
            else:
                print(f'   {row_str}')
        
        print('=' * 80)


if __name__ == '__main__':
    # Test
    from sklearn.datasets import make_regression
    
    X, y = make_regression(n_samples=1000, n_features=100, noise=10, random_state=42)
    
    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Treinar
    trainer = RegressionTrainer(verbose=True)
    trainer.train_all(X_train, y_train, X_val, y_val)
    trainer.evaluate_on_test(X_test, y_test)
    
    # Melhor modelo
    best_name, best_model, best_metrics = trainer.get_best_model()
    print(f'\n🏆 Melhor modelo: {best_name}')
    print(f'   MAE: {best_metrics["MAE"]:.4f}')
    print(f'   R²: {best_metrics["R2"]:.4f}')
