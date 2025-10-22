# Padronização de Diretórios de Teste

## 📋 Resumo

Todos os outputs de testes, validações e comparações agora são criados **exclusivamente** dentro da pasta `tests/` na raiz do projeto.

## ✅ Mudanças Implementadas

### 1. Scripts Atualizados

#### `run_complete_pipeline.py`
```python
# ANTES
parser.add_argument('--output-dir', type=str, default='pipeline_output')

# DEPOIS
parser.add_argument('--output-dir', type=str, default='tests/pipeline_output')
```

#### `compare_classifiers.py`
```python
# ANTES
parser.add_argument('--output-dir', type=str, default='comparison_output')

# DEPOIS
parser.add_argument('--output-dir', type=str, default='tests/comparison_output')
```

### 2. Documentação Atualizada

#### `PIPELINE_GUIDE.md`
Todos os 5 exemplos atualizados:
- `test_output` → `tests/quick_test`
- `dev_test` → `tests/dev_test`
- `validation_results` → `tests/validation_results`
- `production_human` → `tests/production_human`
- `production_all` → `tests/production_all`

#### `CLASSIFIER_COMPARISON_GUIDE.md`
Todos os 3 exemplos atualizados:
- `comparison_results` → `tests/comparison_quick`
- `comparison_1k` → `tests/comparison_1k`
- `comparison_full` → `tests/comparison_full`

### 3. `.gitignore` Atualizado

```gitignore
# Diretórios na pasta tests/ (esperados)
tests/pipeline_output*/
tests/comparison_*/

# ⚠️ IMPORTANTE: Pastas de teste na RAIZ não devem ser criadas!
# Mas se forem criadas acidentalmente, serão ignoradas:
/test_*/
/comparison_*/
/pipeline_output/
```

### 4. Pastas Antigas Removidas

As seguintes pastas foram removidas da raiz do projeto:
- `comparison_results/`
- `test_final_fix/`
- `test_knn_final/`
- `test_knn_fix/`
- `test_knn_fix2/`
- `test_pipeline_output/`

## 📁 Estrutura de Diretórios Atualizada

```
docktkinase/
├── tests/
│   ├── test_*.py                    # Testes unitários
│   ├── pipeline_output/             # Outputs do pipeline (padrão)
│   ├── comparison_output/           # Outputs da comparação (padrão)
│   ├── quick_test/                  # Testes rápidos customizados
│   ├── dev_test/                    # Testes de desenvolvimento
│   ├── validation_results/          # Resultados de validação
│   ├── production_human/            # Produção (humanos)
│   ├── production_all/              # Produção (todos)
│   └── ...                          # Outros outputs customizados
├── src/                             # Código fonte
├── docs/                            # Documentação
├── ESM/                             # Modelo ESM-2 local
├── scripts/                         # Scripts de setup
└── ...
```

## 🎯 Uso Correto

### Pipeline Completo

```bash
# Teste rápido (usa padrão: tests/pipeline_output)
python run_complete_pipeline.py --dataset human --max-samples 50

# Teste customizado
python run_complete_pipeline.py \
    --dataset human \
    --max-samples 100 \
    --output-dir tests/my_custom_test
```

### Comparação de Classificadores

```bash
# Comparação rápida (usa padrão: tests/comparison_output)
python compare_classifiers.py --dataset human --max-samples 50

# Comparação customizada
python compare_classifiers.py \
    --dataset human \
    --max-samples 1000 \
    --output-dir tests/my_comparison
```

## ⚠️ Importante

1. **NÃO criar pastas de teste na raiz do projeto**
   - ❌ `./test_output/`
   - ❌ `./comparison_results/`
   - ❌ `./pipeline_output/`

2. **SEMPRE usar o prefixo `tests/`**
   - ✅ `tests/pipeline_output/`
   - ✅ `tests/comparison_results/`
   - ✅ `tests/my_custom_test/`

3. **Vantagens desta estrutura**
   - ✅ Organização centralizada
   - ✅ Facilita backup e limpeza
   - ✅ Evita poluição da raiz
   - ✅ Padrão profissional
   - ✅ Compatível com .gitignore

## 🔄 Migração de Outputs Antigos

Se você tem outputs em pastas antigas na raiz, migre-os para `tests/`:

```bash
# Exemplo: migrar outputs antigos
mv comparison_results tests/comparison_results_old
mv pipeline_output tests/pipeline_output_old
```

## 📊 Status

- ✅ Scripts atualizados (2 arquivos)
- ✅ Documentação atualizada (2 guias, 8 exemplos)
- ✅ .gitignore configurado
- ✅ Pastas antigas removidas (6 pastas)
- ✅ Estrutura validada

---

**Data de Implementação**: $(date)  
**Versão**: 1.0  
**Responsável**: Padronização de diretórios de teste
