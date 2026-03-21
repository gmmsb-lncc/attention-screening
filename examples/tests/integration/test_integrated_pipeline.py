"""
Testes de integração para o pipeline unificado.

Valida o fluxo end-to-end: build → classifier → regression
"""

import pytest
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

# Adicionar path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / 'src'))

from integrated_pipeline import IntegratedPipeline, IntegratedConfig


@pytest.fixture
def sample_tsv(tmp_path):
    """Criar TSV de exemplo para testes."""
    data = {
        'Smiles': [
            'CCO',  # Ethanol
            'CC(C)O',  # Isopropanol
            'CCCC',  # Butane
            'C1=CC=CC=C1',  # Benzene
            'CC(=O)O'  # Acetic acid
        ],
        'Sequence': [
            'MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK',
            'MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK',
            'MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK',
            'MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK',
            'MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK'
        ],
        'IC50_nM': [50.0, 500.0, 5000.0, 10.0, 100.0],
        'pKi': [7.3, 6.3, 5.3, 8.0, 7.0]
    }
    
    df = pd.DataFrame(data)
    tsv_path = tmp_path / "test_data.tsv"
    df.to_csv(tsv_path, sep='\t', index=False)
    
    return tsv_path


class TestIntegratedPipelineBasic:
    """Testes básicos do pipeline integrado."""
    
    def test_pipeline_initialization(self, tmp_path):
        """Testar inicialização do pipeline."""
        config = IntegratedConfig(
            input_tsv="dummy.tsv",
            output_dir=str(tmp_path / "output")
        )
        
        pipeline = IntegratedPipeline(config)
        
        assert pipeline.config.input_tsv == "dummy.tsv"
        assert pipeline.output_dir.exists()
        assert pipeline.build_dir.exists()
        assert pipeline.classifier_dir.exists()
        assert pipeline.regression_dir.exists()
        assert pipeline.results['status'] == 'initialized'
    
    def test_config_from_dict(self, tmp_path):
        """Testar criação de config a partir de dict."""
        config_dict = {
            'input_tsv': 'test.tsv',
            'output_dir': str(tmp_path / 'output'),
            'esm_model': 'esm2_t6_8M_UR50D',
            'run_classification': False
        }
        
        pipeline = IntegratedPipeline(config_dict)
        
        assert pipeline.config.input_tsv == 'test.tsv'
        assert pipeline.config.esm_model == 'esm2_t6_8M_UR50D'
        assert pipeline.config.run_classification is False
    
    def test_output_directories_created(self, tmp_path):
        """Testar criação de diretórios de saída."""
        config = IntegratedConfig(
            input_tsv="dummy.tsv",
            output_dir=str(tmp_path / "output")
        )
        
        pipeline = IntegratedPipeline(config)
        
        # Verificar diretórios
        assert (tmp_path / "output").exists()
        assert (tmp_path / "output" / "build").exists()
        assert (tmp_path / "output" / "classifier").exists()
        assert (tmp_path / "output" / "regression").exists()


class TestIntegratedPipelineConfig:
    """Testes de configuração."""
    
    def test_default_config_values(self, tmp_path):
        """Testar valores padrão da configuração."""
        config = IntegratedConfig(
            input_tsv="test.tsv",
            output_dir=str(tmp_path)
        )
        
        assert config.esm_model == "esm2_t6_8M_UR50D"
        assert config.ligand_model == "smi-ted-large"
        assert config.batch_size == 8
        assert config.device == "cpu"
        assert config.test_size == 0.2
        assert config.val_size == 0.1
        assert config.random_state == 42
        assert config.run_classification is True
        assert config.run_regression is True
        assert config.verbose is True
    
    def test_custom_config_values(self, tmp_path):
        """Testar valores customizados."""
        config = IntegratedConfig(
            input_tsv="test.tsv",
            output_dir=str(tmp_path),
            esm_model="esm2_t12_35M_UR50D",
            batch_size=16,
            device="cuda",
            test_size=0.3,
            random_state=123,
            run_classification=False,
            verbose=False
        )
        
        assert config.esm_model == "esm2_t12_35M_UR50D"
        assert config.batch_size == 16
        assert config.device == "cuda"
        assert config.test_size == 0.3
        assert config.random_state == 123
        assert config.run_classification is False
        assert config.verbose is False
    
    def test_regression_models_config(self, tmp_path):
        """Testar configuração de modelos de regressão."""
        config = IntegratedConfig(
            input_tsv="test.tsv",
            output_dir=str(tmp_path),
            regression_models=['Ridge', 'Lasso']
        )
        
        assert len(config.regression_models) == 2
        assert 'Ridge' in config.regression_models
        assert 'Lasso' in config.regression_models


@pytest.mark.integration
class TestIntegratedPipelineExecution:
    """
    Testes de execução do pipeline integrado.
    
    NOTA: Estes testes são marcados como @pytest.mark.integration
    e podem ser pulados em CI/CD rápido.
    """
    
    @pytest.mark.slow
    def test_complete_pipeline_execution(self, sample_tsv, tmp_path):
        """
        Testar execução completa do pipeline (build + classifier + regression).
        
        AVISO: Este teste pode levar vários minutos.
        """
        config = IntegratedConfig(
            input_tsv=str(sample_tsv),
            output_dir=str(tmp_path / "output"),
            esm_model="esm2_t6_8M_UR50D",  # Modelo menor para teste
            batch_size=2,
            classifier_epochs=5,  # Poucos epochs para teste
            classifier_cv_folds=2,  # Poucos folds para teste
            regression_models=['Ridge', 'Lasso'],  # Poucos modelos para teste
            regression_cv_folds=2,
            verbose=False
        )
        
        pipeline = IntegratedPipeline(config)
        results = pipeline.run()
        
        # Verificar status
        assert results['status'] == 'completed'
        assert 'timestamp_start' in results
        assert 'timestamp_end' in results
        assert results['total_time_seconds'] > 0
        
        # Verificar build phase
        assert results['build']['success'] is True
        assert Path(results['build']['embeddings']['concatenated']).exists()
        assert Path(results['build']['labels']['binary']).exists()
        assert Path(results['build']['labels']['regression']).exists()
        
        # Verificar classification phase
        assert results['classifier']['success'] is True
        assert 'test_metrics' in results['classifier']
        assert results['classifier']['test_metrics']['roc_auc'] >= 0
        assert results['classifier']['test_metrics']['roc_auc'] <= 1
        
        # Verificar regression phase
        assert results['regression']['success'] is True
        assert 'best_model' in results['regression']
        assert results['regression']['best_mae'] >= 0
        assert -1 <= results['regression']['best_r2'] <= 1
        
        # Verificar arquivo de resultados
        results_file = tmp_path / "output" / "integrated_results.json"
        assert results_file.exists()
        
        with open(results_file) as f:
            saved_results = json.load(f)
        
        assert saved_results['status'] == 'completed'
    
    @pytest.mark.slow
    def test_build_only_execution(self, sample_tsv, tmp_path):
        """Testar execução apenas do build (sem classifier/regression)."""
        config = IntegratedConfig(
            input_tsv=str(sample_tsv),
            output_dir=str(tmp_path / "output"),
            esm_model="esm2_t6_8M_UR50D",
            batch_size=2,
            run_classification=False,
            run_regression=False,
            verbose=False
        )
        
        pipeline = IntegratedPipeline(config)
        results = pipeline.run()
        
        assert results['status'] == 'completed'
        assert results['build']['success'] is True
        assert not results.get('classifier', {})
        assert not results.get('regression', {})
    
    @pytest.mark.slow
    def test_build_and_classification_only(self, sample_tsv, tmp_path):
        """Testar execução de build + classification (sem regression)."""
        config = IntegratedConfig(
            input_tsv=str(sample_tsv),
            output_dir=str(tmp_path / "output"),
            esm_model="esm2_t6_8M_UR50D",
            batch_size=2,
            classifier_epochs=5,
            run_classification=True,
            run_regression=False,
            verbose=False
        )
        
        pipeline = IntegratedPipeline(config)
        results = pipeline.run()
        
        assert results['status'] == 'completed'
        assert results['build']['success'] is True
        assert results['classifier']['success'] is True
        assert not results.get('regression', {})
    
    @pytest.mark.slow
    def test_build_and_regression_only(self, sample_tsv, tmp_path):
        """Testar execução de build + regression (sem classification)."""
        config = IntegratedConfig(
            input_tsv=str(sample_tsv),
            output_dir=str(tmp_path / "output"),
            esm_model="esm2_t6_8M_UR50D",
            batch_size=2,
            run_classification=False,
            run_regression=True,
            regression_models=['Ridge'],
            verbose=False
        )
        
        pipeline = IntegratedPipeline(config)
        results = pipeline.run()
        
        assert results['status'] == 'completed'
        assert results['build']['success'] is True
        assert not results.get('classifier', {})
        assert results['regression']['success'] is True


class TestIntegratedPipelineErrorHandling:
    """Testes de tratamento de erros."""
    
    def test_invalid_input_file(self, tmp_path):
        """Testar com arquivo de entrada inexistente."""
        config = IntegratedConfig(
            input_tsv="nonexistent_file.tsv",
            output_dir=str(tmp_path / "output")
        )
        
        pipeline = IntegratedPipeline(config)
        
        with pytest.raises(Exception):
            pipeline.run()
        
        # Verificar que o erro foi registrado
        assert pipeline.results['status'] == 'failed'
        assert 'error' in pipeline.results
    
    def test_results_saved_on_failure(self, tmp_path):
        """Testar que resultados são salvos mesmo em caso de falha."""
        config = IntegratedConfig(
            input_tsv="nonexistent.tsv",
            output_dir=str(tmp_path / "output")
        )
        
        pipeline = IntegratedPipeline(config)
        
        try:
            pipeline.run()
        except Exception:
            pass
        
        # Verificar que arquivo de resultados foi criado
        results_file = tmp_path / "output" / "integrated_results.json"
        assert results_file.exists()
        
        with open(results_file) as f:
            results = json.load(f)
        
        assert results['status'] == 'failed'


class TestIntegratedPipelineOutputs:
    """Testes de saídas e resultados."""
    
    def test_results_structure(self, tmp_path):
        """Testar estrutura do dict de resultados."""
        config = IntegratedConfig(
            input_tsv="dummy.tsv",
            output_dir=str(tmp_path / "output")
        )
        
        pipeline = IntegratedPipeline(config)
        
        # Verificar estrutura inicial
        assert 'config' in pipeline.results
        assert 'build' in pipeline.results
        assert 'classifier' in pipeline.results
        assert 'regression' in pipeline.results
        assert 'status' in pipeline.results
        assert pipeline.results['status'] == 'initialized'
    
    def test_json_serialization(self, tmp_path):
        """Testar que resultados são serializáveis em JSON."""
        config = IntegratedConfig(
            input_tsv="dummy.tsv",
            output_dir=str(tmp_path / "output")
        )
        
        pipeline = IntegratedPipeline(config)
        
        # Adicionar alguns resultados mockados
        pipeline.results['build'] = {
            'success': True,
            'embeddings': {'protein': 'path/to/embeddings.npy'}
        }
        pipeline.results['classifier'] = {
            'success': True,
            'test_metrics': {'roc_auc': 0.85}
        }
        
        pipeline._save_results()
        
        # Verificar que arquivo foi criado
        results_file = tmp_path / "output" / "integrated_results.json"
        assert results_file.exists()
        
        # Verificar que pode ser carregado
        with open(results_file) as f:
            loaded_results = json.load(f)
        
        assert loaded_results['build']['success'] is True
        assert loaded_results['classifier']['test_metrics']['roc_auc'] == 0.85


def test_integration_suite():
    """
    Test suite summary para integração.
    
    Esta função documenta todos os testes de integração.
    """
    tests = {
        'basic': [
            'test_pipeline_initialization',
            'test_config_from_dict',
            'test_output_directories_created'
        ],
        'config': [
            'test_default_config_values',
            'test_custom_config_values',
            'test_regression_models_config'
        ],
        'execution': [
            'test_complete_pipeline_execution',
            'test_build_only_execution',
            'test_build_and_classification_only',
            'test_build_and_regression_only'
        ],
        'error_handling': [
            'test_invalid_input_file',
            'test_results_saved_on_failure'
        ],
        'outputs': [
            'test_results_structure',
            'test_json_serialization'
        ]
    }
    
    total_tests = sum(len(v) for v in tests.values())
    
    print(f"\nIntegrated Pipeline Test Suite: {total_tests} tests")
    for category, test_list in tests.items():
        print(f"  {category}: {len(test_list)} tests")
    
    return tests


if __name__ == '__main__':
    test_integration_suite()
    pytest.main([__file__, '-v'])
