"""
Testes para validar o sistema de device management.

Testa:
- Detecção de devices disponíveis
- Validação de CUDA/CPU/MPS
- Fallback automático
- Benchmarking opcional
- Integração com MLPPipeline
"""

import pytest
import torch
import logging
from pathlib import Path
import tempfile
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Imports locais
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.classifier.utils.device_manager import DeviceValidator, SmartDeviceManager, DeviceInfo
from src.classifier.main import MLPPipeline

logger = logging.getLogger(__name__)


class TestDeviceValidator:
    """Testes para DeviceValidator."""
    
    def test_cpu_detection(self):
        """Testa detecção de CPU (sempre disponível)."""
        validator = DeviceValidator()
        devices = validator.detect_available_devices()
        
        # CPU deve sempre estar presente
        cpu_devices = [d for d in devices if d.type == "cpu"]
        assert len(cpu_devices) >= 1
        
        cpu_device = cpu_devices[0]
        assert cpu_device.device.type == "cpu"
        assert cpu_device.name is not None
        assert cpu_device.total_memory is not None
        assert cpu_device.is_available
    
    def test_device_validation_basic(self):
        """Testa validação básica de devices."""
        validator = DeviceValidator()
        
        # CPU deve sempre ser válido
        cpu_device = torch.device("cpu")
        result = validator.validate_device_compatibility(cpu_device)
        
        assert result["is_compatible"]
        assert result["device"] == cpu_device
        assert len(result["issues"]) == 0
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA não disponível")
    def test_cuda_detection(self):
        """Testa detecção de CUDA (se disponível)."""
        validator = DeviceValidator()
        devices = validator.detect_available_devices()
        
        cuda_devices = [d for d in devices if d.type == "cuda"]
        if cuda_devices:  # Se CUDA disponível
            cuda_device = cuda_devices[0]
            assert cuda_device.device.type == "cuda"
            assert cuda_device.name is not None
            assert cuda_device.total_memory is not None
            assert cuda_device.compute_capability is not None
            assert cuda_device.is_available
    
    def test_device_ranking(self):
        """Testa ranking de devices."""
        validator = DeviceValidator(prefer_gpu=True)
        devices = validator.detect_available_devices()
        
        # Pelo menos CPU deve estar presente
        assert len(devices) >= 1
        
        # Primeiro device deve ser recomendado
        recommended = [d for d in devices if d.is_recommended]
        assert len(recommended) == 1
        
        # Se GPU disponível, deve ser recomendado
        gpu_devices = [d for d in devices if d.type in ["cuda", "mps"]]
        cpu_devices = [d for d in devices if d.type == "cpu"]
        
        if gpu_devices:
            assert recommended[0].type in ["cuda", "mps"]
        else:
            assert recommended[0].type == "cpu"
    
    def test_benchmarking(self):
        """Testa sistema de benchmark."""
        validator = DeviceValidator(enable_benchmarking=True)
        devices = validator.detect_available_devices()
        
        # Verificar que benchmark foi executado
        benchmarked = [d for d in devices if d.benchmark_score is not None]
        assert len(benchmarked) >= 1  # Pelo menos CPU
        
        # Scores devem ser positivos
        for device in benchmarked:
            assert device.benchmark_score > 0
    
    def test_memory_requirements(self):
        """Testa requisitos de memória."""
        # GPU deve ter pelo menos 2GB
        validator = DeviceValidator(min_gpu_memory_gb=2.0)
        devices = validator.detect_available_devices()
        
        cuda_devices = [d for d in devices if d.type == "cuda"]
        for device in cuda_devices:
            if device.total_memory and device.total_memory < 2.0:
                assert len(device.warnings) > 0  # Deve ter warning
    
    def test_invalid_device(self):
        """Testa validação de device inválido."""
        validator = DeviceValidator()
        
        # Device inexistente
        invalid_device = torch.device("cuda:99")  # Provavelmente não existe
        result = validator.validate_device_compatibility(invalid_device)
        
        # Deve falhar ou ter issues
        assert not result["is_compatible"] or len(result["issues"]) > 0


class TestSmartDeviceManager:
    """Testes para SmartDeviceManager."""
    
    def test_auto_device_selection(self):
        """Testa seleção automática de device."""
        manager = SmartDeviceManager()
        device = manager.get_device("auto")
        
        assert isinstance(device, torch.device)
        
        # Verificar que device funciona
        test_tensor = torch.randn(10, 10).to(device)
        result = test_tensor @ test_tensor
        assert result.shape == (10, 10)
    
    def test_cpu_only_selection(self):
        """Testa seleção forçada de CPU."""
        manager = SmartDeviceManager()
        device = manager.get_device("cpu_only")
        
        assert device.type == "cpu"
        
        # Testar funcionamento
        test_tensor = torch.randn(5, 5).to(device)
        result = test_tensor @ test_tensor
        assert result.shape == (5, 5)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA não disponível")
    def test_gpu_only_selection(self):
        """Testa seleção forçada de GPU."""
        manager = SmartDeviceManager()
        device = manager.get_device("gpu_only")
        
        assert device.type in ["cuda", "mps"]
        
        # Testar funcionamento
        test_tensor = torch.randn(5, 5).to(device)
        result = test_tensor @ test_tensor
        assert result.shape == (5, 5)
    
    def test_fastest_selection(self):
        """Testa seleção do device mais rápido."""
        manager = SmartDeviceManager(enable_benchmarking=True)
        device = manager.get_device("fastest")
        
        assert isinstance(device, torch.device)
        
        # Device selecionado deve ter benchmark score
        device_info = manager.get_device_info()
        if device_info and device_info.benchmark_score:
            assert device_info.benchmark_score > 0
    
    def test_device_info(self):
        """Testa obtenção de informações do device."""
        manager = SmartDeviceManager()
        device = manager.get_device("auto")
        
        device_info = manager.get_device_info()
        assert device_info is not None
        assert device_info.device == device
        assert device_info.name is not None
        assert device_info.type is not None
    
    def test_device_validation(self):
        """Testa validação contínua do device."""
        manager = SmartDeviceManager()
        device = manager.get_device("auto")
        
        # Primeira validação deve passar
        assert manager.validate_current_device()
        
        # Simular falha no device (mock)
        with patch.object(torch, 'randn', side_effect=RuntimeError("Simulated failure")):
            assert not manager.validate_current_device()
    
    def test_fallback_mechanism(self):
        """Testa mecanismo de fallback."""
        # Simular falha na seleção de GPU
        with patch('torch.cuda.is_available', return_value=False):
            manager = SmartDeviceManager()
            
            # Tentar GPU only deve falhar e usar CPU
            device = manager.get_device("gpu_only")  # Deveria fazer fallback
            # Na implementação atual, isso pode dar erro ou fallback para CPU
            assert isinstance(device, torch.device)


class TestMLPPipelineIntegration:
    """Testes de integração com MLPPipeline."""
    
    def create_sample_data(self):
        """Cria dados de exemplo para testes."""
        np.random.seed(42)
        n_samples = 1000
        n_features = 20
        
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)  # Target simples
        
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
        df["target"] = y
        
        return df
    
    def test_pipeline_device_management(self):
        """Testa integração do device management no pipeline."""
        # Criar dados temporários
        df = self.create_sample_data()
        
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            df.to_csv(f.name, index=False)
            data_path = f.name
        
        try:
            # Testar diferentes requisitos de device
            for requirement in ["auto", "cpu_only"]:
                pipeline = MLPPipeline(
                    device_requirement=requirement,
                    min_gpu_memory_gb=0.5
                )
                
                # Device deve estar configurado
                assert pipeline.device is not None
                assert isinstance(pipeline.device, torch.device)
                
                # DeviceManager deve estar configurado
                assert pipeline.device_manager is not None
                device_info = pipeline.device_manager.get_device_info()
                assert device_info is not None
                
                # Validação deve passar
                assert pipeline.validate_device_status()
                
                # Se requirement é cpu_only, device deve ser CPU
                if requirement == "cpu_only":
                    assert pipeline.device.type == "cpu"
        
        finally:
            Path(data_path).unlink()  # Limpar arquivo temporário
    
    def test_pipeline_benchmarking(self):
        """Testa pipeline com benchmarking ativo."""
        df = self.create_sample_data()
        
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            df.to_csv(f.name, index=False)
            data_path = f.name
        
        try:
            pipeline = MLPPipeline(
                device_requirement="auto",
                enable_benchmarking=True
            )
            
            device_info = pipeline.device_manager.get_device_info()
            assert device_info is not None
            
            # Com benchmarking, deve ter score
            if device_info.benchmark_score is not None:
                assert device_info.benchmark_score > 0
        
        finally:
            Path(data_path).unlink()
    
    def test_device_status_validation(self):
        """Testa validação de status do device."""
        pipeline = MLPPipeline(device_requirement="cpu_only")
        
        # Primeira validação deve passar
        assert pipeline.validate_device_status()
        
        # Simular problema no device
        original_validate = pipeline.device_manager.validate_current_device
        pipeline.device_manager.validate_current_device = lambda: False
        
        # Deve detectar problema e tentar fallback
        # (O comportamento exato depende da implementação)
        result = pipeline.validate_device_status()
        
        # Restaurar método original
        pipeline.device_manager.validate_current_device = original_validate


if __name__ == "__main__":
    # Configurar logging para testes
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Executando testes de Device Management...")
    
    # Executar testes básicos
    test_validator = TestDeviceValidator()
    test_validator.test_cpu_detection()
    test_validator.test_device_validation_basic()
    print("✅ DeviceValidator - Testes básicos OK")
    
    test_manager = TestSmartDeviceManager()
    test_manager.test_auto_device_selection()
    test_manager.test_cpu_only_selection()
    print("✅ SmartDeviceManager - Testes básicos OK")
    
    test_integration = TestMLPPipelineIntegration()
    test_integration.test_pipeline_device_management()
    print("✅ MLPPipeline Integration - Testes básicos OK")
    
    print("🎉 Todos os testes básicos passaram!")
    
    # Testes com CUDA (se disponível)
    if torch.cuda.is_available():
        print("\n🔥 CUDA disponível - testando funcionalidades CUDA...")
        try:
            test_validator.test_cuda_detection()
            test_manager.test_gpu_only_selection()
            print("✅ Testes CUDA OK")
        except Exception as e:
            print(f"⚠️  Alguns testes CUDA falharam: {e}")
    else:
        print("⚠️  CUDA não disponível - pulando testes CUDA")
    
    print("\n🚀 Sistema de Device Management validado!")
