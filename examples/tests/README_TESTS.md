# Testes do Sistema attention-screening

Este diretório contém todos os testes do sistema de classificação MLP com auto-adaptação dimensional.

## 🆕 Testes de Auto-Adaptação

### `test_auto_adaptive.py`
Testa o sistema básico de auto-adaptação dimensional:
- ✅ Detecta automaticamente input_size (256, 512, 1024, 2048 features)
- ✅ Cria rede dinamicamente no primeiro forward pass
- ✅ Mantém consistência em múltiplos forward passes

**Executar:**
```bash
cd tests
python test_auto_adaptive.py
```

### `test_final_auto_adaptive.py`
Teste completo e integrado do sistema auto-adaptativo:
- ✅ 4 casos de teste com diferentes dimensões
- ✅ Integração com otimizadores PyTorch
- ✅ Mini-treinamento funcional
- ✅ Verificação de parâmetros e consistência

**Executar:**
```bash
cd tests
python test_final_auto_adaptive.py
```

### `test_realistic_pipeline.py`
Teste do pipeline completo com auto-adaptação:
- ✅ Usa MLPPipeline com métodos corretos
- ✅ ScalableDataset integration
- ✅ Cross-validation (com limitações conhecidas)

**Executar:**
```bash
cd tests
python test_realistic_pipeline.py
```

## 📊 Resultados Esperados

### Auto-Adaptação Funcional:
```
256 features   → 297,729 parâmetros
512 features   → 428,801 parâmetros  
1024 features  → 690,945 parâmetros
2048 features  → 1,215,233 parâmetros
```

### Logs de Sucesso:
```
[MLPEmbeddingClassifier] Auto-detectado input_size: 512
✅ AUTO-DETECÇÃO FUNCIONOU!
```

## 🔧 Testes Existentes

### Validação do Sistema:
- `test_integrity.py` - Testes de integridade geral
- `test_models.py` - Testes de modelos
- `test_device_management.py` - Gestão de dispositivos
- `test_memory_management.py` - Gestão de memória
- `test_performance.py` - Performance e benchmarks

### Validação de Dados:
- `test_robust_split.py` - Divisão robusta de dados
- `simple_test_robust_split.py` - Teste simples de split
- `test_additional_checks.py` - Verificações adicionais
- `test_typing_validation.py` - Validação de tipos

### Relatórios:
- `validacao_final.py` - Validação final do sistema
- `relatorio_final_revisao.py` - Relatório de revisão

## 🚀 Como Executar Todos os Testes

### Testes de Auto-Adaptação (Novos):
```bash
cd /Users/sulfierry/docktkinase/tests
python test_auto_adaptive.py
python test_final_auto_adaptive.py
python test_realistic_pipeline.py
```

### Testes de Sistema (Existentes):
```bash
cd /Users/sulfierry/docktkinase/tests
python test_integrity.py
python test_models.py
python validacao_final.py
```

## ⚡ Recursos Testados

### ✅ Auto-Adaptação:
- [x] Detecção automática de input_size
- [x] Construção dinâmica da rede
- [x] Múltiplas dimensões (256-2048)
- [x] Integração com otimizadores
- [x] Consistência de parâmetros

### ✅ Pipeline Integrado:
- [x] MLPPipeline com auto-config
- [x] ScalableDataset support
- [x] Device management (CPU/MPS/CUDA)
- [x] Cross-validation pipeline

### ✅ Sistema Robusto:
- [x] Import fallbacks
- [x] Error handling
- [x] Memory management
- [x] Device compatibility

---

**Nota:** Os testes usam paths relativos e funcionam corretamente da pasta `tests/`. O sistema de auto-adaptação elimina completamente a necessidade de especificar manualmente o `input_size`, adaptando-se automaticamente aos dados fornecidos.
