# Cache de Modelos

Este diretório armazena modelos pré-treinados do **ESM-2** baixados automaticamente via PyTorch Hub.

> **IMPORTANTE**: O código fonte do ESM está incluído localmente em `ESM/` (35 MB). Este diretório armazena apenas os **pesos dos modelos** baixados sob demanda.

## Estrutura

```
models_cache/
└── ESM/              # Modelos ESM-2 (Meta AI) - apenas pesos
    ├── checkpoints/  # Checkpoints .pt baixados automaticamente
    └── hub/          # Cache do PyTorch Hub
```

## Modelos ESM-2 Disponíveis

### 🎯 RECOMENDADO - Produção (Alta Qualidade)
- **esm2_t36_3B_UR50D** ⭐ **(PADRÃO)**
  - **Parâmetros**: 3 bilhões
  - **Tamanho**: ~12 GB
  - **Camadas**: 36
  - **Embedding dim**: 2560
  - **Uso**: Produção com GPU (RTX 4090), máxima qualidade
  - **Status**: Modelo mais recente e robusto do ESM-2

### Alternativas (se necessário)

- **esm2_t33_650M_UR50D**
  - **Parâmetros**: 650 milhões
  - **Tamanho**: ~2.5 GB
  - **Camadas**: 33
  - **Embedding dim**: 1280
  - **Uso**: Produção com recursos moderados, boa qualidade

- **esm2_t6_8M_UR50D** (apenas testes)
  - **Parâmetros**: 8 milhões
  - **Tamanho**: ~30 MB
  - **Camadas**: 6
  - **Embedding dim**: 320
  - **Uso**: Testes rápidos, validação de pipeline

## Configuração Atual

O pipeline está configurado para usar **ESM-2 t36 3B** (definido em `src/build/core/constants.py`):

```python
DEFAULT_ESM_MODEL = 'esm2_t36_3B_UR50D'  # ESM-2 com 3 bilhões de parâmetros
DEFAULT_PROTEIN_DIM = 2560               # Dimensão dos embeddings
```

O cache é configurado automaticamente em `protein_embedding.py` através de:

```python
cache_dir = Path(__file__).parent.parent.parent.parent / "models_cache" / "ESM"
os.environ['TORCH_HOME'] = str(cache_dir)
```

## Download Automático

Na primeira execução, o modelo será baixado automaticamente:

```
Carregando modelo ESM: esm2_t36_3B_UR50D
Cache de modelos: /Users/sulfierry/docktkinase/models_cache/ESM
Downloading: 100%|██████████| 11.8GB/11.8GB [XX:XX<00:00, XXMB/s]
Modelo ESM carregado com sucesso
```

Execuções subsequentes usarão o modelo em cache (instantâneo).

## Limpeza de Cache

Para liberar espaço em disco:

```bash
# Remover todos os modelos baixados
rm -rf models_cache/ESM/checkpoints/*
rm -rf models_cache/ESM/hub/*

# Remover cache completo (modelos serão re-baixados)
rm -rf models_cache/ESM/*
```

**NOTA**: Os modelos são re-baixados automaticamente quando necessário.

## Código Fonte ESM

O código fonte do ESM (35 MB) está incluído em:
- **Localização**: `/Users/sulfierry/docktkinase/ESM/`
- **Versão**: 1.0.3
- **Conteúdo**: Código Python, exemplos, testes
- **Versionado**: ✅ Sim (incluído no Git)

Os **pesos dos modelos** (`.pt`, `.bin`) **NÃO** são versionados:
- **Localização**: `models_cache/ESM/`
- **Tamanho**: 30 MB a 12 GB (dependendo do modelo)
- **Versionado**: ❌ Não (excluídos pelo `.gitignore`)
- **Download**: Automático via PyTorch Hub
