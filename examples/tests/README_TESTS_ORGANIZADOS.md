# 🧪 Testes do attention-screening Pipeline

Este diretório contém todos os testes do sistema attention-screening, organizados e executáveis a partir da pasta `tests/`.

## 📁 Estrutura dos Testes

### 🔧 Testes Principais (Reorganizados)

Os seguintes testes foram movidos da raiz do projeto para `tests/`:

- **`test_basic_functionality.py`** - Teste básico de funcionalidades essenciais
- **`test_device_manager.py`** - Teste do sistema consolidado de device management  
- **`test_complete_pipeline.py`** - Teste completo do pipeline principal
- **`test_final_report.py`** - Relatório abrangente de validação do sistema
- **`test_config_quick.py`** - Teste rápido do sistema de configuração
- **`test_auto_adaptive.py`** - Teste de adaptação automática
- **`test_final_auto_adaptive.py`** - Teste final de adaptação

### 🧪 Testes Específicos

- **`simple_test_robust_split.py`** - Teste da divisão robusta train/test
- **`test_realistic_pipeline.py`** - Teste com cenários realísticos
- **`test_models.py`** - Teste dos modelos MLP
- **`test_memory_management.py`** - Teste de gerenciamento de memória
- **`test_device_management.py`** - Teste de gerenciamento de dispositivos
- **`test_performance.py`** - Teste de performance
- **`test_integrity.py`** - Teste de integridade do sistema

### 🏃‍♂️ Executor de Testes

- **`run_all_tests.py`** - Suite completa que executa os testes principais

## 🚀 Como Executar

### Teste Individual
```bash
cd tests/
python test_basic_functionality.py
```

### Suite Completa
```bash
cd tests/
python run_all_tests.py
```

### Com pytest
```bash
cd tests/
python -m pytest -v
```

## 📊 Cobertura dos Testes

### ✅ Componentes Testados

1. **Imports e Dependências** - Verificação de todas as bibliotecas necessárias
2. **Device Manager** - Sistema consolidado com 3 modos (simple/smart/complex)
3. **Configuração MLP** - Sistema de configuração centralizada
4. **Modelo MLP** - Criação, forward pass e validação
5. **Train/Test Split** - Divisão robusta com validação estatística
6. **Pipeline Principal** - Integração completa de todos os componentes
7. **Operações com Tensores** - Movimentação e processamento em devices
8. **Sistema de Logging** - Validação de logs e mensagens

### 📈 Status Atual

- **100%** dos testes principais passando
- **5/5** testes na suite completa
- **Tempo de execução:** ~13 segundos
- **Dispositivos suportados:** CPU, CUDA, Apple MPS

## 🔍 Testes por Categoria

### 🏗️ Infraestrutura
- Device Manager (3 modos de operação)
- Sistema de configuração
- Gerenciamento de memória

### 🧠 Modelos
- Criação de modelos MLP
- Forward pass e validação
- Compatibilidade de devices

### 📊 Dados
- Divisão train/test estratificada
- Validação estatística
- DataLoaders e Datasets

### 🔗 Integração
- Pipeline completo end-to-end
- Interação entre componentes
- Fallback automático

## 🛠️ Manutenção

### Adicionar Novo Teste
1. Criar arquivo `test_nova_funcionalidade.py`
2. Incluir path correction: `project_root = Path(__file__).parent.parent`
3. Adicionar à lista em `run_all_tests.py` se for teste principal

### Debug de Falhas
1. Executar teste individual para detalhes
2. Verificar logs no terminal
3. Validar paths e imports
4. Confirmar dependências instaladas

## 📝 Notas

- Todos os testes foram ajustados para executar corretamente da pasta `tests/`
- Sistema de path automático garante imports corretos
- Device MPS (Apple Silicon) totalmente suportado com warnings apropriados
- Fallback automático para CPU em caso de problemas com GPU
