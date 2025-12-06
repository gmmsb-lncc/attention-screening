# Sincronização Automática de Dimensões com Modelos

## 📊 Problema Resolvido

Anteriormente, as dimensões de embeddings eram **fixas** e não sincronizavam com os modelos escolhidos:

```python
# ❌ ANTES: Dimensões fixas, podia causar incompatibilidade
DEFAULT_PROTEIN_DIM = 2560  # Fixo para ESM-2 t36 3B
DEFAULT_LIGAND_DIM = 768    # Fixo para FM4M

# Problema: Se usar esm2_t6_8M_UR50D (320 dims), 
# mas protein_dim=2560, ocorre erro!
```

## ✅ Solução Implementada

Agora o `BuildConfig` **sincroniza automaticamente** as dimensões baseado nos modelos:

### Modelos Suportados

#### ESM (Proteínas)

| Modelo | Dimensões | Parâmetros |
|--------|-----------|------------|
| `esm2_t48_15B_UR50D` | 5120 | 15B |
| `esm2_t36_3B_UR50D` | 2560 | 3B |
| `esm2_t33_650M_UR50D` | 1280 | 650M |
| `esm2_t30_150M_UR50D` | 640 | 150M |
| `esm2_t12_35M_UR50D` | 480 | 35M |
| `esm2_t6_8M_UR50D` | 320 | 8M |

#### ESM-C (EvolutionaryScale Cambrian)

| Modelo | Dimensões | Parâmetros |
|--------|-----------|------------|
| `esmc-300m-2024-12` | 960 | 300M |
| `esmc-600m-2024-12` | 1152 | 600M |
| `esmc-6b-2024-12` | 3072 | 6B |

#### Outros Modelos de Proteína

| Modelo | Dimensões | Tipo |
|--------|-----------|------|
| `openfold3` | 384 | Structure-aware |
| `boltz2` | 384 | Structure + Affinity |

#### FM4M (Ligantes)

| Modelo | Dimensões | Tipo |
|--------|-----------|------|
| `SMI-TED` | 768 | Transformer |
| `SELFIES-TED` | 768 | Transformer |
| `SMI-SSED` | 768 | Encoder |
| `MHG` | 768 | Graph |
| `MOL-MOE` | 768 | Mixture |

## 🚀 Uso

### 1. Automático na Inicialização

```python
from build.core import BuildConfig

# Dimensões são automaticamente ajustadas baseado no modelo padrão
config = BuildConfig()
print(config.protein_dim)  # 320 (esm2_t6_8M_UR50D padrão)
print(config.ligand_dim)   # 768 (SMI-TED padrão)
```

### 2. Com Modelo Específico

```python
# Dimensões sincronizam com o modelo especificado
config = BuildConfig(esm_model='esm2_t36_3B_UR50D')
print(config.protein_dim)  # 2560 (sincronizado automaticamente!)
```

### 3. Alteração Dinâmica

```python
config = BuildConfig()
print(config.protein_dim)  # 320

# Alterar modelo re-sincroniza automaticamente
config.set('esm_model', 'esm2_t33_650M_UR50D')
print(config.protein_dim)  # 1280 (atualizado!)
```

### 4. Método Explícito

```python
# Obter todas as dimensões de uma vez
dims = config.get_model_dimensions()
print(dims)
# {'protein_dim': 2560, 'ligand_dim': 768, 'total_dim': 3328}
```

## 🔧 Como Funciona

### 1. Sincronização no `__init__`

```python
def __init__(self, config_data=None, **kwargs):
    self._config = self._load_default_config()
    # ... carrega configurações ...
    
    # ✨ Sincronização automática após todas as configurações
    self._sync_model_dimensions()
```

### 2. Re-sincronização em `set()` e `update()`

```python
def set(self, key: str, value: Any) -> None:
    self._config[key] = value
    
    # ✨ Re-sincroniza se modelo foi alterado
    if key in ('esm_model', 'fm4m_model'):
        self._sync_model_dimensions()
    
    self._validate_config()
```

### 3. Método `_sync_model_dimensions()`

```python
def _sync_model_dimensions(self) -> None:
    """Sincroniza dimensões com os modelos escolhidos."""
    
    # Proteína: ESM
    esm_model = self._config.get('esm_model')
    if esm_model and esm_model in ESM_MODELS:
        model_dim = ESM_MODELS[esm_model]['dim']
        self._config['protein_dim'] = model_dim
    
    # Ligante: FM4M
    fm4m_model = self._config.get('fm4m_model')
    if fm4m_model and fm4m_model in FM4M_MODELS:
        model_dim = FM4M_MODELS[fm4m_model]['dim']
        self._config['ligand_dim'] = model_dim
```

## 📝 Exemplo Completo

```python
from build.core import BuildConfig

# Cenário 1: Teste rápido com modelo pequeno
config_test = BuildConfig(
    esm_model='esm2_t6_8M_UR50D',  # 320 dims
    fm4m_model='SMI-TED'            # 768 dims
)
print(f"Teste: {config_test.protein_dim} + {config_test.ligand_dim} = {config_test.protein_dim + config_test.ligand_dim} dims")
# Output: Teste: 320 + 768 = 1088 dims

# Cenário 2: Produção com modelo grande
config_prod = BuildConfig(
    esm_model='esm2_t36_3B_UR50D',  # 2560 dims
    fm4m_model='SMI-TED'             # 768 dims
)
print(f"Produção: {config_prod.protein_dim} + {config_prod.ligand_dim} = {config_prod.protein_dim + config_prod.ligand_dim} dims")
# Output: Produção: 2560 + 768 = 3328 dims

# Cenário 3: Alterar modelo dinamicamente
config_prod.set('esm_model', 'esm2_t33_650M_UR50D')
dims = config_prod.get_model_dimensions()
print(f"Alterado: {dims['protein_dim']} + {dims['ligand_dim']} = {dims['total_dim']} dims")
# Output: Alterado: 1280 + 768 = 2048 dims
```

## 🎯 Benefícios

### 1. Prevenção de Erros
```python
# ❌ ANTES: Possível incompatibilidade
config.esm_model = 'esm2_t6_8M_UR50D'  # 320 dims
config.protein_dim = 2560               # Errado! Shape mismatch!

# ✅ AGORA: Sincronização automática
config.set('esm_model', 'esm2_t6_8M_UR50D')
# protein_dim automaticamente = 320 ✓
```

### 2. Menos Código
```python
# ❌ ANTES: Manual e propenso a erros
config.esm_model = 'esm2_t36_3B_UR50D'
config.protein_dim = ESM_MODELS['esm2_t36_3B_UR50D']['dim']

# ✅ AGORA: Automático
config.set('esm_model', 'esm2_t36_3B_UR50D')
```

### 3. Validação Integrada
```python
# Validação automática após sincronização
config.set('esm_model', 'invalid_model')
# ConfigurationError: Modelo ESM inválido
```

## 🧪 Testes de Validação

```bash
# Executar teste de sincronização
python -c "
import sys
sys.path.insert(0, 'src')
from build.core import BuildConfig

# Teste com todos os modelos ESM
for model_name, info in [
    ('esm2_t6_8M_UR50D', 320),
    ('esm2_t12_35M_UR50D', 480),
    ('esm2_t30_150M_UR50D', 640),
    ('esm2_t33_650M_UR50D', 1280),
    ('esm2_t36_3B_UR50D', 2560),
    ('esm2_t48_15B_UR50D', 5120)
]:
    config = BuildConfig(esm_model=model_name)
    expected_dim = info
    actual_dim = config.protein_dim
    status = '✅' if actual_dim == expected_dim else '❌'
    print(f'{status} {model_name}: {actual_dim} (esperado: {expected_dim})')
"
```

## 🔄 Integração com Pipeline

```python
from build.pipeline import BuildPipeline

# Pipeline usa automaticamente as dimensões corretas
pipeline = BuildPipeline(
    config={
        'esm_model': 'esm2_t36_3B_UR50D',  # 2560 dims
        # protein_dim é automaticamente 2560!
    }
)

# Embeddings terão as dimensões corretas
results = pipeline.run()
```

## 📊 Uso em Visualizações

```python
from build.stratification.visualization import StratificationVisualizer
from build.core import BuildConfig

# Config determina as dimensões
config = BuildConfig(esm_model='esm2_t36_3B_UR50D')
dims = config.get_model_dimensions()

print(f"Cada ponto no gráfico representa:")
print(f"  - Proteína: {dims['protein_dim']} dimensões")
print(f"  - Ligante: {dims['ligand_dim']} dimensões")
print(f"  - Total concatenado: {dims['total_dim']} dimensões")
print(f"  - Reduzido para: 2 dimensões (PCA/t-SNE/UMAP)")
```

## ⚠️ Notas Importantes

1. **Sincronização é automática**: Não precisa definir `protein_dim` ou `ligand_dim` manualmente
2. **Sempre use `set()` ou `update()`**: Para garantir re-sincronização
3. **Validação integrada**: Dimensões são validadas após sincronização
4. **Backward compatible**: Código antigo continua funcionando

## 🎓 Resumo

| Aspecto | Valor |
|---------|-------|
| **Sincronização** | ✅ Automática |
| **Momento** | `__init__`, `set()`, `update()` |
| **Validação** | ✅ Integrada |
| **Performance** | ⚡ Instantânea (lookup em dict) |
| **Backward Compatible** | ✅ Sim |
| **Modelos Suportados** | 6 ESM + 5 FM4M = 11 total |

**Agora as dimensões SEMPRE correspondem ao modelo escolhido!** 🎯
