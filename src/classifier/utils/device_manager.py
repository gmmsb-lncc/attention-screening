"""
Device Manager Simplificado - Apenas o essencial para MLP
"""
import torch
import logging

logger = logging.getLogger(__name__)


class SimpleDeviceManager:
    """Device manager minimalista - foco apenas na funcionalidade essencial."""
    
    def __init__(self):
        self.device = self._get_best_device()
        logger.info(f"Device selecionado: {self.device}")
    
    def _get_best_device(self) -> torch.device:
        """Seleciona o melhor device disponível."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def get_device(self, requirement: str = "auto") -> torch.device:
        """Retorna device configurado."""
        return self.device
    
    def get_device_info(self) -> dict:
        """Retorna informações do device."""
        device_type = str(self.device.type)
        return {
            'type': device_type,
            'name': device_type,
            'device': self.device,
            'available_memory': 'N/A',
            'total_memory': 'N/A'
        }
    
    def validate_current_device(self) -> bool:
        """Valida se o device atual está funcionando."""
        try:
            # Teste simples de funcionamento do device
            test_tensor = torch.ones(1).to(self.device)
            return True
        except Exception as e:
            logger.warning(f"Device validation failed: {e}")
            return False
    
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor para device."""
        return tensor.to(self.device)


# Para compatibilidade com código existente
SmartDeviceManager = SimpleDeviceManager
