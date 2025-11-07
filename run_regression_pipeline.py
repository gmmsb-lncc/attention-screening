#!/usr/bin/env python3
"""
Pipeline de Regressão - DockTKinase
====================================

Pipeline completo para previsão de valores de atividade (Ki, Kd, IC50).

🎯 DUAS INTERFACES DISPONÍVEIS:

1. **Pipeline Tradicional**: Reutiliza embeddings e splits do classificador
2. **Pipeline Modular** ⭐ NOVO: Standalone com interface simplificada

Uso Tradicional:
    python run_regression_pipeline.py --dataset all --model esm2_t36_3B_UR50D

Uso Modular (Standalone):
    python src/regression/modular_regression.py embeddings.npy targets.npy

Ver documentação completa:
    - README.md (pipeline tradicional)
    - src/regression/README_MODULAR.md (pipeline modular)

Autor: DockTKinase Team
Data: 2024-2025
"""

import argparse
import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import gc

# Imports para geração de embeddings (opcionais, carregados quando necessário)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Adicionar ESM ao sys.path
esm_path = Path(__file__).parent / 'ESM'
if esm_path.exists() and str(esm_path) not in sys.path:
    sys.path.insert(0, str(esm_path))

# Imports do projeto
from src.regression import (
    RegressionModels,
    RegressionTrainer,
    RegressionEvaluator,
    RegressionVisualizer,
    prepare_regression_targets,
    load_embeddings_cache,
    save_embeddings_cache
)

# Root directory
ROOT_DIR = Path(__file__).resolve().parent


def load_data(dataset_name):
    """Carregar dataset de teste."""
    dataset_files = {
        'human': 'kinase_human_compounds.tsv',
        'non_human': 'kinase_non_human_compounds.tsv',
        'all': 'kinase_all_compounds.tsv'
    }
    
    if dataset_name not in dataset_files:
        raise ValueError(f"Dataset '{dataset_name}' inválido. Use: {list(dataset_files.keys())}")
    
    dataset_path = ROOT_DIR / 'tests' / 'datasets' / dataset_files[dataset_name]
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {dataset_path}")
    
    df = pd.read_csv(dataset_path, sep='\t')
    return df


class RegressionPipeline:
    """
    Pipeline completo de regressão para predição de atividade.
    
    Etapas:
    1. Carregar dados e preparar targets (Ki > Kd > IC50)
    2. Carregar embeddings (do cache ou gerar novos)
    3. Carregar splits do pipeline de classificação
    4. Treinar 11 modelos de regressão
    5. Avaliar e comparar modelos
    6. Gerar visualizações e relatórios
    """
    
    def __init__(self, 
                 dataset='all',
                 model_name='esm2_t36_3B_UR50D',
                 output_dir='results/regression',
                 classification_stats=None,
                 embeddings_cache=None,
                 device='auto',
                 verbose=True,
                 random_state=42):
        """
        Inicializar pipeline de regressão.
        
        Args:
            dataset: Nome do dataset ('all', 'egfr', etc)
            model_name: Modelo ESM usado
            output_dir: Diretório para salvar resultados
            classification_stats: Path para pipeline_stats.json da classificação
            embeddings_cache: Path para cache de embeddings (.npz)
            device: 'cuda', 'mps' ou 'cpu'
            verbose: Mostrar progresso
            random_state: Seed para reprodutibilidade
        """
        self.dataset = dataset
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.classification_stats = classification_stats
        self.embeddings_cache = embeddings_cache
        self.device = device
        self.verbose = verbose
        self.random_state = random_state
        
        # Criar diretórios de saída
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'models').mkdir(exist_ok=True)
        (self.output_dir / 'predictions').mkdir(exist_ok=True)
        (self.output_dir / 'metrics').mkdir(exist_ok=True)
        (self.output_dir / 'visualizations').mkdir(exist_ok=True)
        
        # Stats do pipeline
        self.stats = {
            'pipeline': 'regression',
            'dataset': dataset,
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'device': device,
            'random_state': random_state
        }
        
    def load_data_and_prepare_targets(self, max_samples=None):
        """
        Carregar dados e preparar targets de regressão.
        
        Args:
            max_samples: Número máximo de amostras a carregar (None = todas)
        
        Returns:
            df: DataFrame com targets preparados
            y: Array com valores em nM
            measure_types: Tipos de medida usados
        """
        if self.verbose:
            print('📊 ETAPA 1: Carregamento de Dados e Preparação de Targets')
            print('='*70)
        
        start_time = time.time()
        
        # Carregar dados
        df = load_data(self.dataset)
        
        # Limitar amostras se necessário
        if max_samples and len(df) > max_samples:
            df = df.head(max_samples).copy()
            if self.verbose:
                print(f'   ⚠️  Dataset limitado a {max_samples:,} amostras para teste')
        
        # Preparar targets de regressão (Ki > Kd > IC50)
        # IMPORTANTE: keep_all=True para manter mesmo número de amostras que os splits
        y, df_filtered, measure_types, kept_indices = prepare_regression_targets(
            df, 
            priority=['Ki', 'Kd', 'IC50'],
            keep_all=True,  # Manter todas as medidas para compatibilidade com splits
            verbose=self.verbose
        )
        
        load_time = time.time() - start_time
        
        # Estatísticas dos targets
        measure_counts = pd.Series(measure_types).value_counts()
        
        if self.verbose:
            print(f'\n   ✅ Targets preparados:')
            print(f'      Total de amostras: {len(y):,}')
            print(f'      Valor mínimo: {np.min(y):.2f} nM')
            print(f'      Valor máximo: {np.max(y):.2f} nM')
            print(f'      Média: {np.mean(y):.2f} nM')
            print(f'      Mediana: {np.median(y):.2f} nM')
            print(f'\n   📊 Distribuição de medidas:')
            for measure, count in measure_counts.items():
                pct = count / len(y) * 100
                print(f'      {measure}: {count:,} ({pct:.1f}%)')
            print(f'\n   ⏱️  Tempo: {load_time:.2f}s')
            print()
        
        # Salvar stats
        self.stats['data_loading'] = {
            'load_time': float(load_time),
            'total_samples': int(len(y)),
            'min_value': float(np.min(y)),
            'max_value': float(np.max(y)),
            'mean_value': float(np.mean(y)),
            'median_value': float(np.median(y)),
            'measure_distribution': {k: int(v) for k, v in measure_counts.items()}
        }
        
        return df_filtered, y, measure_types, kept_indices
    
    def load_or_generate_embeddings(self, df):
        """
        Carregar embeddings do cache ou gerar novos.
        
        Args:
            df: DataFrame com coluna 'sequence'
            
        Returns:
            embeddings: Array numpy com embeddings
        """
        if self.verbose:
            print('🧬 ETAPA 2: Carregamento de Embeddings')
            print('='*70)
        
        # Tentar carregar do cache
        if self.embeddings_cache:
            cache_path = Path(self.embeddings_cache)
            
            if cache_path.exists():
                try:
                    data = np.load(cache_path, allow_pickle=True)
                    embeddings = data['embeddings']
                    
                    # Validar dimensões
                    if len(embeddings) == len(df):
                        if self.verbose:
                            print(f'   ✅ Embeddings carregados do cache: {cache_path}')
                            print(f'   📊 Shape: {embeddings.shape}')
                            print()
                        return embeddings
                    else:
                        if self.verbose:
                            print(f'   ⚠️  Dimensões incompatíveis: {len(embeddings)} vs {len(df)}')
                
                except Exception as e:
                    if self.verbose:
                        print(f'   ⚠️  Erro ao carregar cache: {e}')
                        print(f'   🔄 Será necessário gerar embeddings novos')
        
        # Se não conseguiu carregar, gerar novos embeddings
        if self.verbose:
            print(f'   ❌ Cache não encontrado')
            print(f'   🔄 Gerando embeddings usando modelo {self.model_name}...')
        
        # Verificar se torch está disponível
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch não está instalado. É necessário para gerar embeddings.\n"
                "Instale com: pip install torch"
            )
        
        # Importar biblioteca ESM
        import esm
        
        # Carregar modelo
        model_func = getattr(esm.pretrained, self.model_name)
        model, alphabet = model_func()
        model = model.to(self.device)
        model.eval()
        batch_converter = alphabet.get_batch_converter()
        
        if self.verbose:
            print(f'   ✅ Modelo carregado')
            print(f'   📊 Gerando embeddings para {len(df):,} sequências...')
        
        # Gerar embeddings em batches
        embeddings = []
        # Verificar qual coluna de sequência está disponível
        seq_col = 'seq' if 'seq' in df.columns else 'sequence'
        sequences = df[seq_col].tolist()
        batch_size = 8
        
        for i in range(0, len(sequences), batch_size):
            batch_end = min(i + batch_size, len(sequences))
            batch_seqs = sequences[i:batch_end]
            batch_data = [(f'seq_{j}', seq) for j, seq in enumerate(batch_seqs)]
            
            # Converter batch
            _, _, batch_tokens = batch_converter(batch_data)
            batch_tokens = batch_tokens.to(self.device)
            
            # Gerar embeddings
            with torch.no_grad():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                results = model(batch_tokens, repr_layers=[model.num_layers])
                token_embeddings = results['representations'][model.num_layers]
                batch_embeddings = token_embeddings[:, 1:-1, :].mean(1)
                batch_embeddings_cpu = batch_embeddings.cpu().numpy()
            
            embeddings.append(batch_embeddings_cpu)
            
            # Limpar memória
            del batch_tokens, results, token_embeddings, batch_embeddings, batch_embeddings_cpu
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # Progresso
            if self.verbose and (i + batch_size) % 50 == 0:
                processed = min(i + batch_size, len(sequences))
                pct = 100 * processed / len(sequences)
                print(f'      {processed}/{len(sequences)} ({pct:.1f}%)')
        
        # Concatenar
        embeddings = np.vstack(embeddings)
        
        # Liberar modelo da memória
        del model, alphabet, batch_converter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Salvar no cache se path foi especificado
        if self.embeddings_cache:
            cache_path = Path(self.embeddings_cache)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            np.savez_compressed(
                cache_path,
                embeddings=embeddings,
                model=self.model_name,
                dataset=self.dataset,
                n_samples=len(embeddings)
            )
            
            if self.verbose:
                print(f'   � Embeddings salvos no cache: {cache_path}')
        
        if self.verbose:
            print(f'   ✅ Embeddings gerados!')
            print(f'   📊 Shape: {embeddings.shape}')
            print()
        
        return embeddings
    
    def load_split_indices(self):
        """
        Carregar índices dos splits do pipeline de classificação.
        
        Returns:
            idx_train, idx_val, idx_test: Arrays com índices
        """
        if self.verbose:
            print('🔀 ETAPA 3: Carregamento dos Splits')
            print('='*70)
        
        if not self.classification_stats:
            raise ValueError(
                'Path para classification_stats não fornecido. '
                'Use --classification-stats para especificar.'
            )
        
        stats_path = Path(self.classification_stats)
        if not stats_path.exists():
            raise FileNotFoundError(
                f'Arquivo de stats não encontrado: {stats_path}\n'
                f'Execute o pipeline de classificação primeiro.'
            )
        
        # Carregar stats
        with open(stats_path, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        
        # Extrair índices
        if 'split_indices' not in stats:
            raise KeyError(
                'Stats não contém "split_indices". '
                'Execute o pipeline de classificação novamente com a versão atualizada.'
            )
        
        idx_train = np.array(stats['split_indices']['train'])
        idx_val = np.array(stats['split_indices']['val'])
        idx_test = np.array(stats['split_indices']['test'])
        
        if self.verbose:
            print(f'   ✅ Splits carregados:')
            print(f'      Arquivo: {stats_path}')
            print(f'      Train: {len(idx_train):,} amostras')
            print(f'      Val:   {len(idx_val):,} amostras')
            print(f'      Test:  {len(idx_test):,} amostras')
            print(f'      Total: {len(idx_train) + len(idx_val) + len(idx_test):,} amostras')
            print()
        
        # Salvar stats
        self.stats['splits'] = {
            'source': str(stats_path),
            'train_size': int(len(idx_train)),
            'val_size': int(len(idx_val)),
            'test_size': int(len(idx_test))
        }
        
        return idx_train, idx_val, idx_test
    
    def train_models(self, X_train, y_train, X_val, y_val):
        """
        Treinar todos os modelos de regressão.
        
        Args:
            X_train, y_train: Dados de treino
            X_val, y_val: Dados de validação
            
        Returns:
            trainer: RegressionTrainer com modelos treinados
        """
        if self.verbose:
            print('🤖 ETAPA 4: Treinamento de Modelos')
            print('='*70)
        
        # Criar trainer
        trainer = RegressionTrainer(
            models_dict=None,  # Usa todos os modelos disponíveis
            device=self.device,
            verbose=self.verbose,
            random_state=self.random_state
        )
        
        # Treinar todos os modelos
        start_time = time.time()
        val_results = trainer.train_all(X_train, y_train, X_val, y_val)
        train_time = time.time() - start_time
        
        if self.verbose:
            print(f'\n   ⏱️  Tempo total de treinamento: {train_time:.2f}s')
            print()
        
        # Salvar stats
        self.stats['training'] = {
            'total_time': float(train_time),
            'n_models': len(trainer.trained_models),
            'models_trained': list(trainer.trained_models.keys())
        }
        
        return trainer
    
    def evaluate_and_save(self, trainer, X_test, y_test):
        """
        Avaliar modelos no conjunto de teste e salvar resultados.
        
        Args:
            trainer: RegressionTrainer com modelos treinados
            X_test, y_test: Dados de teste
        """
        if self.verbose:
            print('📊 ETAPA 5: Avaliação no Conjunto de Teste')
            print('='*70)
        
        # Avaliar no test set
        test_results = trainer.evaluate_on_test(X_test, y_test)
        
        # Comparar modelos
        comparison_df = RegressionEvaluator.compare_models(test_results, metric='RMSE')
        
        if self.verbose:
            print('\n   🏆 RANKING FINAL (Test Set - RMSE):')
            print('   ' + '='*66)
            for rank, (model_name, row) in enumerate(comparison_df.head(5).iterrows()):
                medal = ['🥇', '🥈', '🥉', '  ', '  '][rank] if rank < 5 else '  '
                print(f'   {medal} {rank+1}. {model_name:20s} | '
                      f'RMSE: {row["RMSE"]:8.2f} | R²: {row["R2"]:6.4f} | '
                      f'MAE: {row["MAE"]:8.2f}')
            print()
        
        # Salvar métricas
        metrics_file = self.output_dir / 'metrics' / 'test_metrics.json'
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)
        
        # Salvar comparação em CSV
        comparison_file = self.output_dir / 'metrics' / 'models_comparison.csv'
        comparison_df.to_csv(comparison_file, index=False)
        
        if self.verbose:
            print(f'   💾 Métricas salvas em: {metrics_file}')
            print(f'   💾 Comparação salva em: {comparison_file}')
            print()
        
        # Salvar stats
        self.stats['evaluation'] = {
            'test_metrics_file': str(metrics_file),
            'comparison_file': str(comparison_file),
            'n_models_evaluated': len(test_results)
        }
        
        return test_results, comparison_df
    
    def generate_visualizations(self, trainer, X_test, y_test, comparison_df):
        """
        Gerar todas as visualizações.
        
        Args:
            trainer: RegressionTrainer com modelos treinados
            X_test, y_test: Dados de teste
            comparison_df: DataFrame com comparação de modelos
        """
        if self.verbose:
            print('📈 ETAPA 6: Geração de Visualizações')
            print('='*70)
        
        viz_dir = self.output_dir / 'visualizations'
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Pegar melhor modelo
        best_model_name, best_model, best_metrics = trainer.get_best_model(metric='RMSE', dataset='test')
        
        # Predições do melhor modelo
        y_pred = best_model.predict(X_test)
        
        if self.verbose:
            print(f'   🎨 Gerando visualizações para modelo: {best_model_name}')
        
        # 1. Predictions vs Actual
        RegressionVisualizer.plot_predictions_vs_actual(
            y_test, y_pred, 
            model_name=best_model_name,
            save_path=viz_dir / 'predictions_vs_actual.png'
        )
        
        # 2. Residuals
        RegressionVisualizer.plot_residuals(
            y_test, y_pred,
            model_name=best_model_name,
            save_path=viz_dir / 'residuals_analysis.png'
        )
        
        # 3. Models Comparison
        RegressionVisualizer.plot_models_comparison(
            trainer.test_results,  # Usar resultados de teste do trainer
            metric='RMSE',
            save_path=viz_dir / 'models_comparison_rmse.png'
        )
        
        # 4. Error Distribution
        RegressionVisualizer.plot_error_distribution(
            y_test, y_pred,
            model_name=best_model_name,
            save_path=viz_dir / 'error_distribution.png'
        )
        
        # 5. Feature Importance (se aplicável)
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = best_model.feature_importances_
            RegressionVisualizer.plot_feature_importance(
                feature_importance,
                model_name=best_model_name,
                save_path=viz_dir / 'feature_importance.png',
                top_n=30
            )
        
        if self.verbose:
            print(f'   ✅ Visualizações salvas em: {self.output_dir / "visualizations"}')
            print()
    
    def save_predictions(self, trainer, X_test, y_test):
        """
        Salvar predições detalhadas de todos os modelos.
        
        Args:
            trainer: RegressionTrainer com modelos treinados
            X_test, y_test: Dados de teste
        """
        if self.verbose:
            print('💾 Salvando Predições Detalhadas')
            print('='*70)
        
        for model_name, model in trainer.trained_models.items():
            # Pular modelos que falharam (None)
            if model is None:
                if self.verbose:
                    print(f'   ⚠️  {model_name}: modelo falhou, pulando')
                continue
            
            y_pred = model.predict(X_test)
            
            # Salvar predições em CSV
            pred_file = self.output_dir / 'predictions' / f'{model_name}_predictions.csv'
            RegressionEvaluator.save_predictions_csv(
                y_test, y_pred, pred_file, model_name
            )
            
            if self.verbose:
                print(f'   ✅ {model_name}: {pred_file.name}')
        
        print()
    
    def save_models(self, trainer):
        """
        Salvar modelos treinados.
        
        Args:
            trainer: RegressionTrainer com modelos treinados
        """
        models_dir = self.output_dir / 'models'
        trainer.save_models(models_dir, save_all=True)
        
        if self.verbose:
            print(f'   💾 Modelos salvos em: {models_dir}')
            print()
    
    def save_pipeline_stats(self):
        """Salvar estatísticas do pipeline."""
        stats_file = self.output_dir / 'regression_stats.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        
        if self.verbose:
            print(f'   📊 Estatísticas salvas em: {stats_file}')
            print()
    
    def run(self):
        """Executar pipeline completo de regressão."""
        if self.verbose:
            print('\n' + '='*70)
            print(' '*15 + '🧪 PIPELINE DE REGRESSÃO - DOCKTKINASE')
            print('='*70)
            print(f'Dataset: {self.dataset}')
            print(f'Modelo: {self.model_name}')
            print(f'Device: {self.device}')
            print(f'Output: {self.output_dir}')
            print('='*70)
            print()
        
        pipeline_start = time.time()
        
        try:
            # 1. Carregar splits primeiro (para saber quais amostras usar)
            idx_train, idx_val, idx_test = self.load_split_indices()
            
            # Combinar todos os índices dos splits
            all_split_indices = np.concatenate([idx_train, idx_val, idx_test])
            n_samples_needed = len(all_split_indices)
            
            if self.verbose:
                print(f'📊 Total de amostras nos splits: {n_samples_needed:,}')
                print(f'   Train: {len(idx_train):,}')
                print(f'   Val:   {len(idx_val):,}')
                print(f'   Test:  {len(idx_test):,}')
                print()
            
            # 2. Carregar apenas as amostras necessárias
            df, y, measure_types, kept_indices = self.load_data_and_prepare_targets(max_samples=n_samples_needed)
            
            # 3. Carregar/gerar embeddings apenas para as amostras necessárias
            embeddings = self.load_or_generate_embeddings(df)
            
            # 4. Aplicar splits
            X_train = embeddings[idx_train]
            X_val = embeddings[idx_val]
            X_test = embeddings[idx_test]
            y_train = y[idx_train]
            y_val = y[idx_val]
            y_test = y[idx_test]
            
            # 4. Treinar modelos
            trainer = self.train_models(X_train, y_train, X_val, y_val)
            
            # 5. Avaliar e salvar resultados
            test_results, comparison_df = self.evaluate_and_save(trainer, X_test, y_test)
            
            # 6. Gerar visualizações
            self.generate_visualizations(trainer, X_test, y_test, comparison_df)
            
            # 7. Salvar predições
            self.save_predictions(trainer, X_test, y_test)
            
            # 8. Salvar modelos
            self.save_models(trainer)
            
            # 9. Salvar stats do pipeline
            pipeline_time = time.time() - pipeline_start
            self.stats['total_pipeline_time'] = float(pipeline_time)
            self.save_pipeline_stats()
            
            if self.verbose:
                print('='*70)
                print('✅ PIPELINE DE REGRESSÃO CONCLUÍDO COM SUCESSO!')
                print('='*70)
                print(f'⏱️  Tempo total: {pipeline_time:.2f}s ({pipeline_time/60:.1f} min)')
                print(f'📁 Resultados em: {self.output_dir}')
                print('='*70)
                print()
            
            return True
            
        except Exception as e:
            if self.verbose:
                print('='*70)
                print('❌ ERRO NO PIPELINE DE REGRESSÃO')
                print('='*70)
                print(f'Erro: {e}')
                print('='*70)
            raise


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description='Pipeline de Regressão - DockTKinase',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:

  # Pipeline TRADICIONAL (reutiliza embeddings do classificador):
  
  # 1. Regressão usando embeddings e splits da classificação
  python run_regression_pipeline.py \\
      --dataset all \\
      --model esm2_t36_3B_UR50D \\
      --classification-stats results/pipeline_stats.json \\
      --embeddings-cache results/embeddings_esm2_t36_3B_UR50D.npz

  # 2. Gerar embeddings automaticamente (sem cache)
  python run_regression_pipeline.py \\
      --dataset human \\
      --model esm2_t6_8M_UR50D \\
      --classification-stats results/test_small/pipeline_stats.json \\
      --output-dir results/test_small/regression

  # 3. Com GPU CUDA
  python run_regression_pipeline.py \\
      --dataset egfr \\
      --device cuda

  # 4. Com Apple Silicon (M1/M2)
  python run_regression_pipeline.py \\
      --dataset all \\
      --device mps

  # Pipeline MODULAR ⭐ NOVO (standalone, interface simplificada):
  
  # 1. Execução básica
  python src/regression/modular_regression.py embeddings.npy targets.npy
  
  # 2. Com modelos específicos
  python src/regression/modular_regression.py embeddings.npy targets.npy \\
      --models RandomForest XGBoost KNN
  
  # 3. Customizar output e splits
  python src/regression/modular_regression.py embeddings.npy targets.npy \\
      --output results/my_experiment \\
      --test-size 0.15 --val-size 0.15
  
  Ver documentação completa em:
    - README.md (pipeline tradicional)
    - src/regression/README_MODULAR.md (pipeline modular)
        """
    )
    
    parser.add_argument('--dataset', type=str, default='all',
                       help='Dataset a usar (all, egfr, etc)')
    parser.add_argument('--model', type=str, default='esm2_t36_3B_UR50D',
                       help='Modelo ESM usado para embeddings')
    parser.add_argument('--classification-stats', type=str,
                       default='results/pipeline_stats.json',
                       help='Path para pipeline_stats.json da classificação')
    parser.add_argument('--embeddings-cache', type=str, default=None,
                       help='Path para cache de embeddings (opcional, será gerado se não fornecido)')
    parser.add_argument('--output-dir', type=str, default='results/regression',
                       help='Diretório para salvar resultados')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'mps', 'cpu'],
                       help='Device para computação')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Seed para reprodutibilidade')
    parser.add_argument('--quiet', action='store_true',
                       help='Suprimir output verbose')
    
    args = parser.parse_args()
    
    # Criar pipeline
    pipeline = RegressionPipeline(
        dataset=args.dataset,
        model_name=args.model,
        output_dir=args.output_dir,
        classification_stats=args.classification_stats,
        embeddings_cache=args.embeddings_cache,
        device=args.device,
        verbose=not args.quiet,
        random_state=args.random_state
    )
    
    # Executar
    success = pipeline.run()
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
