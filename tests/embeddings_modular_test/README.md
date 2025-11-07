# Modular Embeddings - Test Suite

Testes completos para validar a implementação modular de embeddings.

## 📋 Estrutura dos Testes

```
tests/embeddings_modular_test/
├── __init__.py
├── run_all_tests.py          # Master test runner
├── test_1_validators.py      # Validação de sequências/SMILES
├── test_2_data_loader.py     # Carregamento de dados
├── test_3_model_registry.py  # Registro de modelos
├── test_4_cache.py           # Sistema de cache
└── test_5_integration.py     # Testes de integração (modelos reais)
```

## 🚀 Como Executar

### Todos os testes (recomendado)
```bash
cd tests/embeddings_modular_test
python run_all_tests.py
```

### Testes individuais
```bash
# Apenas validators
python test_1_validators.py

# Apenas data loader
python test_2_data_loader.py

# Apenas model registry
python test_3_model_registry.py

# Apenas cache
python test_4_cache.py

# Apenas integração (modelos reais)
python test_5_integration.py
```

## 📊 Níveis de Teste

### Level 1: Validators (Rápido, ~1s)
- ✅ Validação de sequências de proteínas
- ✅ Validação de SMILES

### Level 2: Data Loader (Rápido, ~2s)
- ✅ Carregamento de listas
- ✅ Carregamento de FASTA
- ✅ Carregamento de CSV/TSV
- ✅ Carregamento de DataFrames

### Level 3: Model Registry (Rápido, ~1s)
- ✅ Operações básicas do registro
- ✅ Informações de modelos
- ✅ Validação de nomes
- ✅ Info de todos os modelos ESM

### Level 4: Cache Manager (Médio, ~3s)
- ✅ Inicialização
- ✅ Cache em memória
- ✅ Cache em disco
- ✅ Cache miss
- ✅ Limpeza de cache

### Level 5: Integration - REAL MODELS (Demorado, ~30s primeira vez)
- ✅ Embeddings de proteínas (ESM2 8M - menor modelo)
- ✅ Dataset real (kinase_test_small.tsv)
- ✅ Embeddings de ligantes (FM4M)
- ✅ Tratamento de erros

## ⚠️ Notas Importantes

### Primeira Execução
- Level 5 irá **baixar modelos** (~32MB para ESM2 8M, ~500MB para FM4M)
- Pode demorar alguns minutos
- Downloads subsequentes usarão cache

### Requisitos
```bash
# Certifique-se de ter as dependências instaladas
pip install torch transformers pandas numpy tqdm
```

### Recursos
- **Level 1-4**: Testes rápidos, sem modelos reais
- **Level 5**: Usa modelos reais, requer GPU/CPU e memória

## 📈 Saída Esperada

```
==================================================================
 🧪 MODULAR EMBEDDINGS - COMPLETE TEST SUITE 
==================================================================

==================================================================
 LEVEL 1: VALIDATORS 
==================================================================
✅ 1.1 Protein Validation PASSED
✅ 1.2 SMILES Validation PASSED

[... mais testes ...]

==================================================================
 TEST SUMMARY 
==================================================================

📊 Total Tests: 18
✅ Passed: 18
❌ Failed: 0

==================================================================
✅ ALL TESTS PASSED!
==================================================================

🎉 Successfully validated modular embeddings implementation!
   - 18 tests passed
   - All components working correctly
   - Ready for production use
```

## 🐛 Troubleshooting

### Erro: "Module not found"
```bash
# Certifique-se de estar no diretório correto
cd tests/embeddings_modular_test
```

### Erro: "CUDA out of memory"
```bash
# Use CPU forçando no código ou reduza batch_size
# Os testes já usam batch_size pequeno
```

### Erro: "Model download failed"
```bash
# Verifique conexão com internet
# Modelos são baixados do HuggingFace
```

## ✅ Critérios de Sucesso

Para considerar a modularização bem-sucedida, todos os 18 testes devem passar:

- ✅ 2 testes de validação
- ✅ 5 testes de data loader
- ✅ 4 testes de model registry
- ✅ 5 testes de cache
- ✅ 4 testes de integração

## 🎯 Próximos Passos

Após todos os testes passarem:

1. ✅ Commit do código testado
2. ✅ Atualizar documentação principal
3. ✅ Considerar adicionar testes de performance
4. ✅ Integrar com CI/CD se disponível
