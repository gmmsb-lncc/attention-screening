# 🧬 DockTKinase - Guia Completo do Usuário

## 📋 Índice
1. [Instalação](#instalação)
2. [Configuração](#configuração)
3. [Uso Básico](#uso-básico)
4. [Configurações Avançadas](#configurações-avançadas)
5. [Exemplos Práticos](#exemplos-práticos)
6. [Solução de Problemas](#solução-de-problemas)
7. [API Reference](#api-reference)

---

## 🚀 Instalação

### Instalação Automática (Recomendada)
```bash
# Clone o repositório
git clone https://github.com/gmmsb-lncc/docktkinase.git
cd docktkinase

# Execute o setup automático
python setup_completo.py
```

### Instalação Manual
```bash
# Criar ambiente virtual
python -m venv env
source env/bin/activate  # Linux/Mac
# ou env\Scripts\activate  # Windows

# Instalar dependências
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn jupyter

# Verificar instalação
python launch_docktkinase.py
```

---

## ⚙️ Configuração

### Configuração Básica
```python
from classifier.main import MLPPipeline
from classifier.utils.config_manager import ConfigManager

# Criar pipeline com configuração padrão
pipeline = MLPPipeline()

# Ou usar gerenciador de configuração
config_mgr = ConfigManager()
config = config_mgr.create_config("development")
pipeline = MLPPipeline(config=config)
```

### Templates de Configuração Disponíveis

#### 1. **Development** (Desenvolvimento)
```python
config = config_mgr.create_config("development")
# - Learning rate alto (0.01)
# - Poucas épocas (10)
# - Batch size pequeno (32)
# - Ideal para testes rápidos
```

#### 2. **Production** (Produção)
```python
config = config_mgr.create_config("production")
# - Learning rate baixo (0.001)
# - Muitas épocas (100)
# - Batch size otimizado (128)
# - Configurações estáveis
```

#### 3. **Research** (Pesquisa)
```python
config = config_mgr.create_config("research")
# - Configurações experimentais
# - Dropout alto (0.5)
# - Arquiteturas complexas
# - Para exploração de hiperparâmetros
```

---

## 🎯 Uso Básico

### 1. Inicialização Rápida
```python
# Launcher automático
python launch_docktkinase.py

# Em Python
from classifier.main import MLPPipeline

# Criar pipeline
pipeline = MLPPipeline()
print("Sistema pronto!")
```

### 2. Detecção Automática de Hardware
```python
from classifier.utils.device_manager import SmartDeviceManager

device_mgr = SmartDeviceManager()
device = device_mgr.get_device()
print(f"Usando: {device}")

# GPU NVIDIA: cuda
# GPU Apple: mps  
# CPU: cpu
```

### 3. Carregamento de Dados
```python
from classifier.core.data_manager import DataManager

# Criar gerenciador de dados
data_mgr = DataManager()

# Carregar dados (implementar conforme seus dados)
# data_mgr.load_dataset("path/to/data")
```

### 4. Treinamento Simples
```python
# Pipeline completo
pipeline = MLPPipeline()

# Configurar dados
# pipeline.set_data(X_train, y_train, X_test, y_test)

# Treinar modelo
# pipeline.train()

# Avaliar
# results = pipeline.evaluate()
```

---

## 🔧 Configurações Avançadas

### Criação de Configuração Customizada
```python
from classifier.config.mlp_config import MLPConfig

# Configuração customizada
custom_config = MLPConfig(
    hidden_layers=[512, 256, 128],
    learning_rate=0.005,
    dropout_rate=0.3,
    batch_size=64,
    epochs=50,
    optimizer="adam",
    loss_function="cross_entropy"
)

# Usar com pipeline
pipeline = MLPPipeline(config=custom_config)
```

### Gestão Avançada de Memória
```python
from classifier.core.memory_manager import MemoryManager

# Monitoramento de memória
memory_mgr = MemoryManager()

# Configurar limites
memory_mgr.set_memory_limit(8)  # 8GB

# Usar com dados grandes
with memory_mgr.memory_context():
    # Operações que consomem muita memória
    large_computation()
```

### Configurações de Device Específicas
```python
from classifier.utils.device_manager import DeviceValidator

# Validar capabilities
validator = DeviceValidator()
capabilities = validator.validate_device_capabilities()

print(f"CUDA disponível: {capabilities['cuda_available']}")
print(f"MPS disponível: {capabilities['mps_available']}")
print(f"Memória GPU: {capabilities['gpu_memory']} MB")
```

---

## 📚 Exemplos Práticos

### Exemplo 1: Classificação Simples
```python
#!/usr/bin/env python3
"""
Exemplo básico de classificação com DockTKinase.
"""
import sys
from pathlib import Path

# Setup do ambiente
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from classifier.main import MLPPipeline
from classifier.utils.config_manager import ConfigManager

def exemplo_classificacao():
    print("🧬 Exemplo de Classificação DockTKinase")
    
    # Configuração
    config_mgr = ConfigManager()
    config = config_mgr.create_config("development")
    
    # Pipeline
    pipeline = MLPPipeline(config=config)
    
    # Seus dados aqui
    # X_train, y_train = carregar_dados_treino()
    # X_test, y_test = carregar_dados_teste()
    
    # Treino
    # pipeline.set_data(X_train, y_train, X_test, y_test)
    # pipeline.train()
    
    # Avaliação
    # results = pipeline.evaluate()
    # print(f"Acurácia: {results['accuracy']:.3f}")
    
    print("✅ Pipeline configurado com sucesso!")

if __name__ == "__main__":
    exemplo_classificacao()
```

### Exemplo 2: Otimização de Hiperparâmetros
```python
#!/usr/bin/env python3
"""
Exemplo de otimização de hiperparâmetros.
"""
from classifier.utils.config_manager import ConfigManager
from classifier.config.mlp_config import MLPConfig

def otimizar_hiperparametros():
    print("🔍 Otimização de Hiperparâmetros")
    
    # Configurações para testar
    configs_para_testar = [
        {"hidden_layers": [128, 64], "learning_rate": 0.01},
        {"hidden_layers": [256, 128], "learning_rate": 0.005},
        {"hidden_layers": [512, 256, 128], "learning_rate": 0.001},
    ]
    
    melhores_resultados = []
    
    for i, params in enumerate(configs_para_testar):
        print(f"🧪 Testando configuração {i+1}/3...")
        
        # Criar configuração
        config = MLPConfig(**params)
        
        # Treinar e avaliar
        # pipeline = MLPPipeline(config=config)
        # accuracy = pipeline.train_and_evaluate()
        # melhores_resultados.append((params, accuracy))
        
        print(f"   Configuração: {params}")
        # print(f"   Acurácia: {accuracy:.3f}")
    
    # Encontrar melhor
    # melhor = max(melhores_resultados, key=lambda x: x[1])
    # print(f"🏆 Melhor configuração: {melhor[0]}")
    # print(f"🎯 Melhor acurácia: {melhor[1]:.3f}")

if __name__ == "__main__":
    otimizar_hiperparametros()
```

### Exemplo 3: Análise de Performance
```python
#!/usr/bin/env python3
"""
Exemplo de análise de performance do sistema.
"""
import time
from classifier.utils.device_manager import SmartDeviceManager
from classifier.core.memory_manager import MemoryManager

def analisar_performance():
    print("📊 Análise de Performance")
    
    # Setup
    device_mgr = SmartDeviceManager()
    memory_mgr = MemoryManager()
    
    # Informações do sistema
    device = device_mgr.get_device()
    print(f"🖥️  Device: {device}")
    
    # Benchmark simples
    start_time = time.time()
    
    # Simular operações
    for i in range(100):
        # Operações de exemplo
        config_mgr = ConfigManager()
        config = config_mgr.create_config("development")
        
        if (i + 1) % 20 == 0:
            print(f"   Processado: {i+1}/100")
    
    elapsed = time.time() - start_time
    ops_per_sec = 100 / elapsed
    
    print(f"⏱️  Tempo total: {elapsed:.2f}s")
    print(f"🚀 Operações/seg: {ops_per_sec:.1f}")
    
    # Uso de memória
    memory_info = memory_mgr.get_memory_usage()
    print(f"💾 Memória usada: {memory_info['used_mb']:.1f}MB")

if __name__ == "__main__":
    analisar_performance()
```

---

## 🐛 Solução de Problemas

### Problemas Comuns

#### 1. **Erro de Import**
```
ModuleNotFoundError: No module named 'classifier'
```
**Solução:**
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))
```

#### 2. **CUDA não encontrado**
```
RuntimeError: CUDA not available
```
**Solução:**
```python
# O sistema detecta automaticamente e usa CPU
device_mgr = SmartDeviceManager()
device = device_mgr.get_device()  # Retorna 'cpu' se CUDA indisponível
```

#### 3. **Memória Insuficiente**
```
RuntimeError: CUDA out of memory
```
**Solução:**
```python
# Usar gestão de memória
memory_mgr = MemoryManager()
memory_mgr.set_memory_limit(4)  # Limitar a 4GB

# Ou reduzir batch_size
config.training.batch_size = 32  # Ao invés de 128
```

#### 4. **Configuração Inválida**
```
ValidationError: Invalid configuration
```
**Solução:**
```python
# Usar templates validados
config_mgr = ConfigManager()
config = config_mgr.create_config("development")  # Sempre válido
```

### Debugging

#### Ativar Logs Detalhados
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Agora todos os módulos mostrarão logs detalhados
```

#### Verificar Estado do Sistema
```python
# Script de diagnóstico
python test_integrity.py  # Testa integridade
python test_performance.py  # Testa performance
python test_typing_validation.py  # Valida tipos
```

---

## 📖 API Reference

### MLPPipeline
```python
class MLPPipeline:
    def __init__(self, config: Optional[MLPConfig] = None)
    def set_data(self, X_train, y_train, X_test, y_test)
    def train(self) -> Dict[str, Any]
    def evaluate(self) -> Dict[str, Any]
    def predict(self, X) -> np.ndarray
    def save_model(self, path: str)
    def load_model(self, path: str)
```

### ConfigManager
```python
class ConfigManager:
    def create_config(self, template: str) -> MLPConfig
    def save_config(self, config: MLPConfig, path: str)
    def load_config(self, path: str) -> MLPConfig
    def validate_config(self, config: MLPConfig) -> bool
    def get_available_templates(self) -> List[str]
```

### SmartDeviceManager
```python
class SmartDeviceManager:
    def get_device(self) -> str
    def set_device(self, device: str)
    def get_device_info(self) -> Dict[str, Any]
    def optimize_for_device(self, model) -> model
```

### MemoryManager
```python
class MemoryManager:
    def set_memory_limit(self, limit_gb: float)
    def get_memory_usage(self) -> Dict[str, float]
    def clear_cache(self)
    def memory_context(self) -> ContextManager
```

---

## 🎯 Próximos Passos

1. **Execute o setup**: `python setup_completo.py`
2. **Teste o sistema**: `python launch_docktkinase.py`
3. **Execute exemplos**: Use os scripts de exemplo acima
4. **Desenvolva seu modelo**: Adapte para seus dados

## 🆘 Suporte

- **Issues**: [GitHub Issues](https://github.com/gmmsb-lncc/docktkinase/issues)
- **Documentação**: Este arquivo + docstrings no código
- **Exemplos**: Diretório `examples/` (quando disponível)

---

**🧬 DockTKinase** - Sistema de classificação molecular inteligente para pesquisa em quinases.
