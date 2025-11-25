#!/usr/bin/env python3
"""
DockTKinase Integrated Pipeline
================================

Sistema de integração end-to-end que orquestra todos os módulos:
- build: Geração de embeddings e matrizes
- classifier: Classificação binária (ativo/inativo)
- regression: Predição quantitativa (pKi/IC50)

Uso:
    # CLI
    python -m integrated_pipeline --input data.tsv --output results/

    # Python API
    from integrated_pipeline import IntegratedPipeline
    
    pipeline = IntegratedPipeline(
        input_tsv="data.tsv",
        output_dir="results/"
    )
    results = pipeline.run()
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Adicionar paths
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / 'src'))


@dataclass
class IntegratedConfig:
    """Configuração integrada para todos os módulos."""
    
    # Input/Output
    input_tsv: str
    output_dir: str = "results/integrated"
    use_checkpoints: bool = True  # Usar checkpoints para evitar recálculo
    
    # Build module
    esm_model: str = "esm2_t6_8M_UR50D"
    esm_dim: Optional[int] = None  # None = usar dimensão padrão do modelo
    ligand_model: str = "smi-ted-large"
    batch_size: int = 8
    device: str = "cpu"
    
    # Embedding directories (reutilização de embeddings pré-computados)
    protein_embeddings_dir: Optional[str] = None  # Se especificado, usa embeddings existentes
    ligand_embeddings_dir: Optional[str] = None   # Se especificado, usa embeddings existentes (compartilhável)
    
    # Data split
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    
    # Stratification settings
    stratifier_auto_threshold: bool = True  # Use automatic threshold detection
    stratifier_threshold: Optional[float] = None  # Manual threshold (0.0-1.0) - overrides auto
    stratifier_method: str = 'target'  # Auto-threshold method: silhouette, elbow, target, percentile
    
    # Classification
    run_classification: bool = True
    use_multi_model_classification: bool = False  # True = 10 modelos, False = MLP apenas
    classification_models: Optional[List[str]] = None  # None = todos, ou lista específica
    classifier_epochs: int = 50  # Apenas para MLP
    classifier_cv_folds: int = 5  # Apenas para MLP
    
    # Regression
    run_regression: bool = True
    regression_models: Optional[List[str]] = None  # None = todos os 10 modelos
    regression_cv_folds: int = 5
    
    # Binary threshold for classification labels
    binary_threshold: float = 1000.0  # nM
    
    # Options
    verbose: bool = True
    save_models: bool = True
    create_visualizations: bool = True


class IntegratedPipeline:
    """
    Pipeline integrado end-to-end do DockTKinase.
    
    Orquestra todos os módulos em sequência:
    1. Build: Gera embeddings (ligand + protein) e matrizes
    2. Classifier: Treina modelo de classificação binária
    3. Regression: Treina modelos de regressão quantitativa
    """
    
    def __init__(self, config: Union[IntegratedConfig, Dict[str, Any]]):
        """
        Inicializar pipeline integrado.
        
        Args:
            config: IntegratedConfig ou dict com configurações
        """
        if isinstance(config, dict):
            config = IntegratedConfig(**config)
        
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories
        self.build_dir = self.output_dir / "build"
        self.classifier_dir = self.output_dir / "classifier"
        self.regression_dir = self.output_dir / "regression"
        
        for dir_path in [self.build_dir, self.classifier_dir, self.regression_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Results storage
        self.results = {
            'config': self.config.__dict__ if hasattr(config, '__dict__') else config,
            'build': {},
            'classifier': {},
            'regression': {},
            'status': 'initialized',
            'timestamp_start': None,
            'timestamp_end': None,
            'total_time_seconds': None
        }
    
    def run(self) -> Dict[str, Any]:
        """
        Executar pipeline completo integrado.
        
        Returns:
            Dict com resultados de todos os módulos
        """
        start_time = time.time()
        self.results['timestamp_start'] = datetime.now().isoformat()
        
        if self.config.verbose:
            self._print_header()
        
        try:
            # Phase 1: Build embeddings and matrices
            if self.config.verbose:
                print("\n" + "="*80)
                print("PHASE 1: BUILD - Embedding Generation & Matrix Construction")
                print("="*80)
            
            # Tentar carregar checkpoint
            build_results = self._load_checkpoint('build')
            
            if build_results is None:
                # Executar build phase
                build_results = self._run_build_phase()
                self._save_checkpoint('build', build_results)
            else:
                if self.config.verbose:
                    print("📂 Usando checkpoint da fase de Build")
                
                # Atualizar checkpoint antigo se necessário (adicionar n_samples e embedding_dim)
                if 'n_samples' not in build_results or 'embedding_dim' not in build_results:
                    import numpy as np
                    embedding_matrix_path = self.build_dir / "embedding_matrix.npy"
                    if embedding_matrix_path.exists():
                        embedding_matrix = np.load(embedding_matrix_path)
                        build_results['n_samples'] = embedding_matrix.shape[0]
                        build_results['embedding_dim'] = embedding_matrix.shape[1]
                        # Salvar checkpoint atualizado
                        self._save_checkpoint('build', build_results)
                        if self.config.verbose:
                            print(f"   Updated checkpoint with statistics: {build_results['n_samples']} samples, {build_results['embedding_dim']} features")
            
            self.results['build'] = build_results
            
            # Phase 2: Classification (optional)
            if self.config.run_classification:
                if self.config.verbose:
                    print("\n" + "="*80)
                    print("PHASE 2: CLASSIFICATION - Binary Activity Prediction")
                    print("="*80)
                
                # Tentar carregar checkpoint
                classifier_results = self._load_checkpoint('classifier')
                
                if classifier_results is None:
                    # Executar classification phase
                    classifier_results = self._run_classification_phase(build_results)
                    self._save_checkpoint('classifier', classifier_results)
                else:
                    if self.config.verbose:
                        print("📂 Usando checkpoint da fase de Classification")
                    
                    # Se o checkpoint não tem as chaves esperadas, processar
                    if 'best_model' not in classifier_results:
                        # Checkpoint antigo - processar para encontrar melhor modelo
                        best_model_name = None
                        best_roc_auc = -1.0
                        best_metrics = {}
                        individual_results = {}
                        
                        for model_name, metrics in classifier_results.items():
                            if isinstance(metrics, dict):
                                roc_auc = metrics.get('ROC_AUC', -1.0)
                                if roc_auc > best_roc_auc:
                                    best_roc_auc = roc_auc
                                    best_model_name = model_name
                                    best_metrics = metrics
                                
                                individual_results[model_name] = {
                                    'roc_auc': float(metrics.get('ROC_AUC', 0)),
                                    'accuracy': float(metrics.get('Accuracy', 0)),
                                    'f1': float(metrics.get('F1', 0)),
                                    'precision': float(metrics.get('Precision', 0)),
                                    'recall': float(metrics.get('Recall', 0))
                                }
                        
                        # Reconstruir com estrutura esperada
                        classifier_results = {
                            'success': True,
                            'mode': 'MultiModel',
                            'n_models_trained': len(individual_results),
                            'best_model': best_model_name,
                            'best_metrics': {
                                'ROC_AUC': float(best_metrics.get('ROC_AUC', 0)),
                                'Accuracy': float(best_metrics.get('Accuracy', 0)),
                                'F1': float(best_metrics.get('F1', 0)),
                                'Precision': float(best_metrics.get('Precision', 0)),
                                'Recall': float(best_metrics.get('Recall', 0))
                            },
                            'individual_results': individual_results
                        }
                    
                    if self.config.verbose and classifier_results.get('best_model'):
                        print(f"   Best model: {classifier_results['best_model']}")
                        print(f"   Best ROC-AUC: {classifier_results['best_metrics']['ROC_AUC']:.4f}")
                
                self.results['classifier'] = classifier_results
            
            # Phase 3: Regression (optional)
            if self.config.run_regression:
                if self.config.verbose:
                    print("\n" + "="*80)
                    print("PHASE 3: REGRESSION - Quantitative Activity Prediction")
                    print("="*80)
                
                # Tentar carregar checkpoint
                regression_results = self._load_checkpoint('regression')
                
                if regression_results is None:
                    # Executar regression phase
                    regression_results = self._run_regression_phase(build_results)
                    self._save_checkpoint('regression', regression_results)
                else:
                    if self.config.verbose:
                        print("📂 Usando checkpoint da fase de Regression")
                
                self.results['regression'] = regression_results
            
            # Success
            self.results['status'] = 'completed'
            
        except Exception as e:
            self.results['status'] = 'failed'
            self.results['error'] = str(e)
            
            if self.config.verbose:
                print(f"\n❌ Pipeline failed: {e}")
            
            raise
        
        finally:
            end_time = time.time()
            self.results['timestamp_end'] = datetime.now().isoformat()
            self.results['total_time_seconds'] = end_time - start_time
            
            # Save final results
            self._save_results()
            
            if self.config.verbose:
                self._print_summary()
        
        return self.results
    
    def _run_build_phase(self) -> Dict[str, Any]:
        """
        Phase 1: Gerar embeddings e construir matrizes.
        
        Returns:
            Dict com paths dos arquivos gerados
        """
        from build.pipeline import BuildPipeline
        from build.core import BuildConfig
        
        # Configurar build
        build_config = BuildConfig(
            input_tsv=self.config.input_tsv,
            output_dir=str(self.build_dir),
            esm_model=self.config.esm_model,
            esm_dim=self.config.esm_dim,  # Dimensão customizada
            ligand_model=self.config.ligand_model,
            batch_size=self.config.batch_size,
            device=self.config.device,
            binary_threshold=self.config.binary_threshold,
            test_size=self.config.test_size,
            val_size=self.config.val_size,
            random_state=self.config.random_state,
            # Diretórios de embeddings pré-existentes (reutilização)
            protein_embeddings_dir=self.config.protein_embeddings_dir,
            ligand_embeddings_dir=self.config.ligand_embeddings_dir
        )
        
        # Executar build pipeline
        build_pipeline = BuildPipeline(build_config)
        success = build_pipeline.run_complete_pipeline(
            input_tsv_path=self.config.input_tsv,
            output_dir=self.build_dir,
            matrix_type='embedding',
            binary_threshold=self.config.binary_threshold,
            run_validation=True
        )
        
        if not success:
            raise RuntimeError("Build phase failed")
        
        # Load stratified splits (NEW: use stratification system)
        from build.pipeline.split_indices import SplitIndices
        
        splits_file = self.build_dir / "splits" / "stratified_splits.npz"
        split_indices = None
        
        if splits_file.exists():
            try:
                split_indices = SplitIndices.load(str(splits_file))
                if self.config.verbose:
                    print(f"✅ Loaded stratified splits from: {splits_file}")
                    print(f"   Train: {len(split_indices.train_idx)} samples")
                    print(f"   Val:   {len(split_indices.val_idx)} samples")
                    print(f"   Test:  {len(split_indices.test_idx)} samples")
            except Exception as e:
                if self.config.verbose:
                    print(f"⚠️  Warning: Could not load stratified splits: {e}")
                    print("   Pipelines will use default splitting")
        
        # Coletar paths dos arquivos gerados
        results = {
            'success': True,
            'embeddings': {
                'protein': str(self.build_dir / "embeddings" / "protein_embeddings.npy"),
                'ligand': str(self.build_dir / "embeddings" / "ligand_embeddings.npy"),
                'concatenated': str(self.build_dir / "embedding_matrix.npy")
            },
            'labels': {
                'binary': str(self.build_dir / "binary_labels.npy"),
                'regression': str(self.build_dir / "interaction_labels.npy")
            },
            'splits': {
                'train_indices': str(self.build_dir / "splits" / "train_indices.npy"),
                'val_indices': str(self.build_dir / "splits" / "val_indices.npy"),
                'test_indices': str(self.build_dir / "splits" / "test_indices.npy")
            },
            'split_indices': split_indices  # NEW: pass SplitIndices object
        }
        
        # Load embedding matrix to get statistics
        import numpy as np
        embedding_matrix_path = self.build_dir / "embedding_matrix.npy"
        if embedding_matrix_path.exists():
            embedding_matrix = np.load(embedding_matrix_path)
            results['n_samples'] = embedding_matrix.shape[0]
            results['embedding_dim'] = embedding_matrix.shape[1]
        else:
            results['n_samples'] = 0
            results['embedding_dim'] = 0
        
        if self.config.verbose:
            print("✅ Build phase completed successfully")
            print(f"   Embeddings saved to: {self.build_dir / 'embeddings'}")
            print(f"   Matrix saved to: {self.build_dir / 'matrix'}")
            print(f"   Labels saved to: {self.build_dir / 'labels'}")
        
        return results
    
    def _run_classification_phase(self, build_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Phase 2: Treinar classificador binário.
        
        Args:
            build_results: Resultados do build phase
        
        Returns:
            Dict com métricas do classificador
        """
        # Paths dos dados
        embeddings_path = build_results['embeddings']['concatenated']
        labels_path = build_results['labels']['binary']
        split_indices = build_results.get('split_indices')  # NEW: get stratified splits
        
        # Escolher pipeline: Multi-modelo ou MLP único
        if self.config.use_multi_model_classification:
            return self._run_multi_model_classification(embeddings_path, labels_path, split_indices)
        else:
            return self._run_mlp_classification(embeddings_path, labels_path, split_indices)
    
    def _run_mlp_classification(self, embeddings_path: str, labels_path: str, split_indices=None) -> Dict[str, Any]:
        """
        Executar classificação com MLP único (modo legado).
        
        Args:
            embeddings_path: Path dos embeddings concatenados
            labels_path: Path dos labels binários
            split_indices: Optional SplitIndices object for stratified splits
            
        Returns:
            Dict com métricas do MLP
        """
        from classifier.modular_pipeline import MLPEmbeddingPipeline
        
        # Criar pipeline de classificação
        classifier = MLPEmbeddingPipeline(
            embeddings_path=embeddings_path,
            labels_path=labels_path,
            batch_size=32,
            lr=0.001,
            epochs=self.config.classifier_epochs,
            test_split=self.config.test_size,
            val_split=self.config.val_size,
            early_stopping_patience=10,
            model_output=str(self.classifier_dir / "mlp_model.pth"),
            metrics_output=str(self.classifier_dir / "metrics.json"),
            split_indices=split_indices  # NEW: pass stratified splits
        )
        
        # Carregar dados
        classifier.load_data()
        
        # Treinar
        val_loss = classifier.train()
        
        # Cross-validation
        cv_results = classifier.cross_validate(k=self.config.classifier_cv_folds)
        
        # Avaliar no test set
        test_metrics = classifier.evaluate(
            classifier.model,
            classifier.test_loader
        )
        
        results = {
            'success': True,
            'mode': 'MLP',
            'val_loss': float(val_loss),
            'test_metrics': {
                'accuracy': float(test_metrics.get('accuracy', 0)),
                'precision': float(test_metrics.get('precision', 0)),
                'recall': float(test_metrics.get('recall', 0)),
                'f1': float(test_metrics.get('f1', 0)),
                'roc_auc': float(test_metrics.get('roc_auc', 0))
            },
            'cv_results': {
                'mean_roc_auc': float(cv_results.get('mean_roc_auc', 0)),
                'std_roc_auc': float(cv_results.get('std_roc_auc', 0)),
                'n_folds': self.config.classifier_cv_folds
            },
            'model_path': str(self.classifier_dir / "mlp_model.pth")
        }
        
        return results
    
    def _run_multi_model_classification(self, embeddings_path: str, labels_path: str, split_indices=None) -> Dict[str, Any]:
        """
        Executar classificação com múltiplos modelos sklearn.
        
        Args:
            embeddings_path: Path dos embeddings concatenados
            labels_path: Path dos labels binários
            split_indices: Optional SplitIndices object for stratified splits
            
        Returns:
            Dict com métricas de todos os modelos
        """
        from classifier.multi_model_pipeline import MultiModelClassificationPipeline
        
        # Criar pipeline multi-modelo
        # TODO: Add split_indices support to MultiModelClassificationPipeline
        if split_indices and self.config.verbose:
            print("⚠️  Note: MultiModelClassificationPipeline doesn't support split_indices yet")
            print("   Using automatic stratification within the pipeline")
        
        pipeline = MultiModelClassificationPipeline(
            embeddings_path=embeddings_path,
            labels_path=labels_path,
            output_dir=str(self.classifier_dir),
            models_to_train=self.config.classification_models,  # None = todos
            test_size=self.config.test_size,
            val_size=self.config.val_size,
            random_state=self.config.random_state,
            verbose=self.config.verbose
        )
        
        # Executar pipeline completo
        test_metrics = pipeline.run()
        
        # Encontrar melhor modelo com base em ROC-AUC
        best_model_name = None
        best_roc_auc = -1.0
        best_metrics = {}
        
        for model_name, metrics in test_metrics.items():
            roc_auc = metrics.get('ROC_AUC', -1.0)
            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc
                best_model_name = model_name
                best_metrics = metrics
        
        # Compilar resultados
        results = {
            'success': True,
            'mode': 'MultiModel',
            'n_models_trained': len(test_metrics),
            'best_model': best_model_name,
            'best_metrics': {
                'ROC_AUC': float(best_metrics.get('ROC_AUC', 0)),
                'Accuracy': float(best_metrics.get('Accuracy', 0)),
                'F1': float(best_metrics.get('F1', 0)),
                'Precision': float(best_metrics.get('Precision', 0)),
                'Recall': float(best_metrics.get('Recall', 0))
            },
            'individual_results': {}
        }
        
        # Adicionar métricas individuais
        for model_name, metrics in test_metrics.items():
            results['individual_results'][model_name] = {
                'roc_auc': float(metrics.get('ROC_AUC', 0)),
                'accuracy': float(metrics.get('Accuracy', 0)),
                'f1': float(metrics.get('F1', 0)),
                'precision': float(metrics.get('Precision', 0)),
                'recall': float(metrics.get('Recall', 0))
            }
        
        if self.config.verbose:
            print("✅ Classification phase completed successfully")
            print(f"   Best model: {best_model_name}")
            print(f"   Best ROC-AUC: {best_roc_auc:.4f}")
        
        return results
    
    def _run_regression_phase(self, build_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Phase 3: Treinar modelos de regressão.
        
        Args:
            build_results: Resultados do build phase
        
        Returns:
            Dict com métricas dos modelos de regressão
        """
        from regression.modular_pipeline import RegressionPipeline
        
        # Paths dos dados
        embeddings_path = build_results['embeddings']['concatenated']
        targets_path = build_results['labels']['regression']
        
        # Extrair split_indices do build phase
        split_indices = build_results.get('split_indices')
        
        # Criar pipeline de regressão
        regression = RegressionPipeline(
            embeddings_path=embeddings_path,
            targets_path=targets_path,
            output_dir=str(self.regression_dir),
            models_to_train=self.config.regression_models,
            test_size=self.config.test_size,
            val_size=self.config.val_size,
            random_state=self.config.random_state,
            verbose=self.config.verbose,
            split_indices=split_indices
        )
        
        # FASE 1: Carregar dados (com checkpoint)
        data_checkpoint = self._load_checkpoint('regression_data')
        if data_checkpoint is None:
            if self.config.verbose:
                print("📊 Carregando dados de regressão...")
            regression.load_data()
            
            # Salvar checkpoint de dados
            data_info = {
                'n_samples': len(regression.y_train) + len(regression.y_val) + len(regression.y_test),
                'n_train': len(regression.y_train),
                'n_val': len(regression.y_val),
                'n_test': len(regression.y_test),
                'n_features': regression.X_train.shape[1]
            }
            self._save_checkpoint('regression_data', data_info)
        else:
            if self.config.verbose:
                print("📂 Checkpoint de dados carregado")
                print(f"   Samples: {data_checkpoint['n_samples']} ({data_checkpoint['n_train']}/{data_checkpoint['n_val']}/{data_checkpoint['n_test']})")
            # Recarregar dados
            regression.load_data()
        
        # FASE 2: Treinar modelos (com checkpoint)
        train_checkpoint = self._load_checkpoint('regression_train')
        if train_checkpoint is None:
            if self.config.verbose:
                print("🎯 Treinando modelos de regressão...")
            train_results = regression.train_models()
            self._save_checkpoint('regression_train', train_results)
        else:
            if self.config.verbose:
                print("📂 Checkpoint de treinamento carregado")
            train_results = train_checkpoint
            # Recarregar modelos treinados no regression pipeline
            regression.val_metrics = train_results
        
        # FASE 3: Avaliar no conjunto de teste
        test_checkpoint = self._load_checkpoint('regression_test')
        if test_checkpoint is None:
            if self.config.verbose:
                print("📊 Avaliando modelos no conjunto de teste...")
            test_results = regression.evaluate_on_test()
            self._save_checkpoint('regression_test', test_results)
        else:
            if self.config.verbose:
                print("📂 Checkpoint de teste carregado")
            test_results = test_checkpoint
            regression.test_metrics = test_results
        
        # Encontrar melhor modelo com base em MAE do conjunto de validação
        best_model_name = None
        best_mae = float('inf')
        best_r2 = -float('inf')
        
        for model_name, metrics in train_results.items():
            mae = metrics.get('MAE', float('inf'))
            if mae < best_mae:
                best_mae = mae
                best_model_name = model_name
                best_r2 = metrics.get('R2', 0.0)
        
        if self.config.verbose and train_checkpoint:
            print(f"   Best model: {best_model_name}")
            print(f"   Best MAE: {best_mae:.3f}")
            print(f"   Best R²: {best_r2:.4f}")
        
        # Cross-validation (opcional, para modelos selecionados)
        from regression.core import quick_cross_validate
        
        cv_results = {}
        # CV apenas para poucos modelos (se modelos específicos foram escolhidos)
        if self.config.regression_models and len(self.config.regression_models) <= 3:
            cv_results = quick_cross_validate(
                regression.X_train,
                regression.y_train,
                model_names=self.config.regression_models,
                n_splits=self.config.regression_cv_folds,
                random_state=self.config.random_state
            )
        
        # Compilar resultados
        models_trained = len(self.config.regression_models) if self.config.regression_models else len(train_results)
        results = {
            'success': True,
            'best_model': best_model_name,
            'best_mae': float(best_mae),
            'best_r2': float(best_r2),
            'models_trained': models_trained,
            'individual_results': {},
            'test_results': {}
        }
        
        # Adicionar métricas individuais de validação
        for model_name, metrics in train_results.items():
            results['individual_results'][model_name] = {
                'mae': float(metrics.get('MAE', 0)),
                'rmse': float(metrics.get('RMSE', 0)),
                'r2': float(metrics.get('R2', 0))
            }
        
        # Adicionar métricas de teste
        if test_results:
            for model_name, metrics in test_results.items():
                if metrics:  # Verificar se não é None
                    results['test_results'][model_name] = {
                        'mae': float(metrics.get('MAE', 0)),
                        'rmse': float(metrics.get('RMSE', 0)),
                        'r2': float(metrics.get('R2', 0))
                    }
        
        # Adicionar CV se disponível
        if cv_results:
            results['cv_results'] = {}
            for model_name, cv_result in cv_results.items():
                results['cv_results'][model_name] = {
                    'mae_mean': float(cv_result.get_mean_metric('mae')),
                    'mae_std': float(cv_result.get_std_metric('mae')),
                    'r2_mean': float(cv_result.get_mean_metric('r2')),
                    'r2_std': float(cv_result.get_std_metric('r2'))
                }
        
        if self.config.verbose:
            print("✅ Regression phase completed successfully")
            print(f"   Best model: {results['best_model']}")
            print(f"   Best MAE: {results['best_mae']:.3f}")
            print(f"   Best R²: {results['best_r2']:.4f}")
        
        return results
    
    def _save_results(self) -> None:
        """Salvar resultados finais em JSON."""
        results_file = self.output_dir / "integrated_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        if self.config.verbose:
            print(f"\n📁 Results saved to: {results_file}")
    
    def _print_header(self) -> None:
        """Imprimir cabeçalho do pipeline."""
        print("\n" + "="*80)
        print(" " * 20 + "🧬 DOCKTKINASE INTEGRATED PIPELINE 🧬")
        print("="*80)
        print(f"Input TSV: {self.config.input_tsv}")
        print(f"Output Dir: {self.config.output_dir}")
        print(f"ESM Model: {self.config.esm_model}")
        print(f"Device: {self.config.device}")
        print(f"Random Seed: {self.config.random_state}")
        print("\nModules to run:")
        print(f"  • Build: ✅ (always required)")
        if self.config.run_classification:
            mode = "Multi-Model (10 models)" if self.config.use_multi_model_classification else "MLP only"
            print(f"  • Classification: ✅ ({mode})")
        else:
            print(f"  • Classification: ❌")
        print(f"  • Regression: {'✅ (10 models)' if self.config.run_regression else '❌'}")
        print("="*80)
    
    def _print_summary(self) -> None:
        """Imprimir resumo final."""
        print("\n" + "="*80)
        print(" " * 25 + "🎉 PIPELINE SUMMARY 🎉")
        print("="*80)
        
        print(f"\n📊 Status: {self.results['status'].upper()}")
        print(f"⏱️  Total time: {self.results['total_time_seconds']:.2f} seconds")
        
        # Build results
        if self.results.get('build', {}).get('success'):
            print("\n✅ Build Phase: SUCCESS")
        
        # Classification results
        if self.config.run_classification and self.results.get('classifier', {}).get('success'):
            clf = self.results['classifier']
            print("\n✅ Classification Phase: SUCCESS")
            
            # Multi-model mode
            if clf.get('mode') == 'MultiModel':
                print(f"   Mode: Multi-Model ({clf.get('n_models_trained', 0)} models)")
                print(f"   Best model: {clf.get('best_model', 'Unknown')}")
                if 'best_metrics' in clf:
                    print(f"   Best ROC-AUC: {clf['best_metrics'].get('ROC_AUC', 0):.4f}")
                    print(f"   Best F1: {clf['best_metrics'].get('F1', 0):.4f}")
            # MLP mode
            else:
                print(f"   Mode: MLP (single model)")
                print(f"   Test ROC-AUC: {clf['test_metrics']['roc_auc']:.4f}")
                print(f"   Test Accuracy: {clf['test_metrics']['accuracy']:.4f}")
                print(f"   CV ROC-AUC: {clf['cv_results']['mean_roc_auc']:.4f} ± {clf['cv_results']['std_roc_auc']:.4f}")
        
        # Regression results
        if self.config.run_regression and self.results.get('regression', {}).get('success'):
            reg = self.results['regression']
            print("\n✅ Regression Phase: SUCCESS")
            print(f"   Best model: {reg['best_model']}")
            print(f"   Best MAE: {reg['best_mae']:.3f}")
            print(f"   Best R²: {reg['best_r2']:.4f}")
        
        print("\n" + "="*80)
        print(f"📁 All results saved to: {self.output_dir}")
        print("="*80 + "\n")
    
    def _save_checkpoint(self, phase_name: str, phase_results: Dict[str, Any]) -> None:
        """
        Salva checkpoint de uma fase específica.
        
        Args:
            phase_name: Nome da fase ('build', 'classifier', 'regression')
            phase_results: Resultados da fase
        """
        if not self.config.use_checkpoints:
            return
        
        checkpoint_dir = self.output_dir / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_file = checkpoint_dir / f'{phase_name}_checkpoint.json'
        
        with open(checkpoint_file, 'w') as f:
            json.dump(phase_results, f, indent=2)
        
        if self.config.verbose:
            print(f"✅ Checkpoint salvo: {checkpoint_file}")
    
    def _load_checkpoint(self, phase_name: str) -> Optional[Dict[str, Any]]:
        """
        Carrega checkpoint de uma fase se existir.
        
        Args:
            phase_name: Nome da fase
            
        Returns:
            Resultados da fase ou None se checkpoint não existe
        """
        if not self.config.use_checkpoints:
            return None
        
        checkpoint_file = self.output_dir / 'checkpoints' / f'{phase_name}_checkpoint.json'
        
        if not checkpoint_file.exists():
            return None
        
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            
            if self.config.verbose:
                print(f"📂 Checkpoint carregado: {checkpoint_file}")
            
            return checkpoint_data
        except Exception as e:
            if self.config.verbose:
                print(f"⚠️  Erro ao carregar checkpoint: {e}")
            return None


def main():
    """Entry point de linha de comando."""
    parser = argparse.ArgumentParser(
        description="DockTKinase Integrated Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Complete workflow (classification + regression)
  python -m integrated_pipeline --input data.tsv --output results/

  # Classification only
  python -m integrated_pipeline --input data.tsv --no-regression

  # Regression only
  python -m integrated_pipeline --input data.tsv --no-classification

  # Custom models
  python -m integrated_pipeline --input data.tsv \\
      --regression-models Ridge Lasso XGBoost \\
      --esm-model esm2_t33_650M_UR50D
        """
    )
    
    # Required
    parser.add_argument(
        '--input',
        required=True,
        help='Input TSV file with kinase data'
    )
    
    parser.add_argument(
        '--output',
        default='results/integrated',
        help='Output directory (default: results/integrated)'
    )
    
    # Build options
    parser.add_argument(
        '--esm-model',
        default='esm2_t6_8M_UR50D',
        help='ESM model name (default: esm2_t6_8M_UR50D)'
    )
    
    parser.add_argument(
        '--device',
        default='cpu',
        choices=['cpu', 'cuda', 'mps'],
        help='Device for computation (default: cpu)'
    )
    
    # Module selection
    parser.add_argument(
        '--no-classification',
        action='store_true',
        help='Skip classification phase'
    )
    
    parser.add_argument(
        '--no-regression',
        action='store_true',
        help='Skip regression phase'
    )
    
    # Regression options
    parser.add_argument(
        '--regression-models',
        nargs='+',
        default=['Ridge', 'Lasso', 'ElasticNet', 'RandomForest', 'XGBoost'],
        help='Regression models to train (default: Ridge Lasso ElasticNet RandomForest XGBoost)'
    )
    
    # General options
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    # Criar configuração
    config = IntegratedConfig(
        input_tsv=args.input,
        output_dir=args.output,
        esm_model=args.esm_model,
        device=args.device,
        run_classification=not args.no_classification,
        run_regression=not args.no_regression,
        regression_models=args.regression_models,
        random_state=args.random_state,
        verbose=not args.quiet
    )
    
    # Executar pipeline
    pipeline = IntegratedPipeline(config)
    results = pipeline.run()
    
    # Status de saída
    return 0 if results['status'] == 'completed' else 1


if __name__ == '__main__':
    sys.exit(main())
