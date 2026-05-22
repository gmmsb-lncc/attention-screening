# 📁 Correção de Portabilidade de Caminhos

**Data:** 21 de Outubro de 2025  
**Branch:** diamante-03  
**Tipo:** Fix - Portabilidade  

---

## 🎯 Problema Identificado

O arquivo de configuração `src/stratification_config.json` continha um **caminho absoluto hardcoded** que funcionava apenas na máquina do desenvolvedor Leon:

```json
"fm4m_config": {
    "model_path": "${HOME}/Desktop/latent_extractor/ibm/FM4M"
}
```

### ❌ Problemas desta Abordagem:

1. **Não portátil**: Não funciona em outras máquinas
2. **Quebra colaboração**: Outros desenvolvedores não conseguem executar
3. **Falha em CI/CD**: Pipelines automatizados não funcionam
4. **Incompatível com Docker**: Containers não encontram o caminho
5. **Dependência de estrutura externa**: Requer path fora do repositório

---

## ✅ Solução Aplicada

Substituição do caminho absoluto por **caminho relativo ao repositório**:

```json
"fm4m_config": {
    "model_path": "../FM4M"
}
```

### ✅ Vantagens desta Abordagem:

1. **Portátil**: Funciona em qualquer máquina que tenha o repositório
2. **Facilita colaboração**: Outros desenvolvedores podem executar imediatamente
3. **CI/CD compatível**: Pipelines automatizados funcionam sem configuração extra
4. **Docker-friendly**: Containers funcionam sem modificações
5. **Auto-contido**: Tudo está dentro do repositório

---

## 📁 Arquivos Modificados

### 1. `src/stratification_config.json`

**Linha modificada:** 53

**Antes:**
```json
"model_path": "${HOME}/Desktop/latent_extractor/ibm/FM4M"
```

**Depois:**
```json
"model_path": "../FM4M"
```

**Contexto:** 
- Arquivo está em: `docktkinase/src/`
- Caminho relativo: `../FM4M` → `docktkinase/FM4M/` ✅

---

### 2. `src/build/example_usage.py`

**Linha modificada:** 40

**Antes:**
```python
"fm4m_config": {
    "model_path": "${HOME}/Desktop/latent_extractor/ibm/FM4M",
    "batch_size": 32,
    "device": "auto"
}
```

**Depois:**
```python
"fm4m_config": {
    "model_path": "../FM4M",
    "batch_size": 32,
    "device": "auto"
}
```

**Contexto:**
- Script de exemplo que **gera configurações automaticamente**
- Garante que novos configs criados já usem caminhos relativos
- Propaga a correção para futuros usos

---

## 📐 Como Funciona o Caminho Relativo

### Estrutura do Repositório:

```
docktkinase/                    ← Raiz do repositório
├── FM4M/                       ← 🎯 Destino do caminho
│   ├── models/
│   │   └── fm4m.py
│   ├── app.py
│   └── README.md
├── src/                        ← 📍 Localização dos arquivos corrigidos
│   ├── stratification_config.json  ← Arquivo 1
│   └── build/
│       └── example_usage.py        ← Arquivo 2
└── docs/
    └── PATH_PORTABILITY_FIX.md     ← Este documento
```

### Resolução do Caminho `../FM4M`:

1. **Posição inicial:** `docktkinase/src/` (onde está o arquivo config)
2. **`../`**: Sobe um nível → `docktkinase/`
3. **`FM4M`**: Entra na pasta → `docktkinase/FM4M/` ✅

### Exemplo em Python:

```python
from pathlib import Path

# Dentro de qualquer script em src/
config_dir = Path(__file__).parent  # docktkinase/src/
fm4m_path = config_dir / "../FM4M"  # docktkinase/FM4M/

# Resolve para caminho absoluto
fm4m_absolute = fm4m_path.resolve()
print(fm4m_absolute)
# Saída: /caminho/completo/para/docktkinase/FM4M
```

---

## 🔍 Outros Caminhos Absolutos Identificados

Durante a análise, foram encontrados outros caminhos absolutos. **Análise de cada caso:**

### ✅ **Mantidos (Propositalmente Não Corrigidos):**

#### 1. Arquivos SQL (`src/database/sql/`)

```sql
-- kinase_humans.sql, kinase_compounds_and_seq.sql, kinase_non_humans.sql
\COPY public.smile_kinase_human_compounds TO '${PROJECT_ROOT}/...' WITH ...
```

**Motivo:** Scripts SQL de backup/export históricos  
**Ação:** Documentados apenas, não usados no pipeline de produção  
**Impacto:** Zero - arquivos de referência histórica  

---

#### 2. Scripts Legacy (`legacy/backup_legacy_scripts/`)

```python
# check_embedding_dim.py, find_missing_sequences.py
protein_embedding_file = "${PROJECT_ROOT}/non_humans/..."
```

**Motivo:** Scripts obsoletos arquivados  
**Ação:** Mantidos para referência histórica  
**Impacto:** Zero - não usados no código ativo  

---

#### 3. Documentação (`docs/HUGGINGFACE_RATE_LIMIT.md`)

```bash
cd ${PROJECT_ROOT}
```

**Motivo:** Exemplos de comandos na documentação  
**Ação:** OK usar caminhos absolutos em exemplos  
**Impacto:** Zero - apenas ilustração  

---

## 🧪 Como Testar

### Teste 1: Verificar Resolução do Caminho

```bash
cd docktkinase/src
python -c "import os; print(os.path.abspath('../FM4M'))"
```

**Saída esperada:** `/caminho/completo/para/docktkinase/FM4M`

---

### Teste 2: Verificar Existência do Diretório

```python
from pathlib import Path

# Simula carregamento do config
config_path = Path("src/stratification_config.json")
fm4m_relative = "../FM4M"

# Resolve caminho
fm4m_full = (config_path.parent / fm4m_relative).resolve()

# Verifica existência
assert fm4m_full.exists(), f"FM4M não encontrado em: {fm4m_full}"
assert fm4m_full.is_dir(), f"FM4M não é um diretório: {fm4m_full}"

print(f"✅ FM4M encontrado em: {fm4m_full}")
```

---

### Teste 3: Testar Importação do Modelo

```python
import sys
from pathlib import Path

# Adicionar FM4M ao path
fm4m_path = Path(__file__).parent.parent / "FM4M"
sys.path.append(str(fm4m_path))

# Tentar importar
try:
    import models.fm4m as fm4m
    print("✅ FM4M importado com sucesso!")
except ImportError as e:
    print(f"❌ Erro ao importar FM4M: {e}")
```

---

## 🚀 Benefícios da Correção

| Aspecto | Antes | Depois |
|---------|-------|--------|
| **Portabilidade** | ❌ Apenas máquina do Leon | ✅ Qualquer máquina |
| **Colaboração** | ❌ Requer ajustes manuais | ✅ Clone e execute |
| **CI/CD** | ❌ Falha em pipelines | ✅ Funciona automaticamente |
| **Docker** | ❌ Requer volume externo | ✅ Auto-contido |
| **Manutenção** | ❌ Dependência externa | ✅ Tudo no repositório |

---

## 📊 Impacto e Alcance

### Componentes Afetados:

- ✅ **Build Pipeline:** Carregamento correto do modelo FM4M
- ✅ **Ligand Embeddings:** Geração de embeddings de ligantes
- ✅ **Example Scripts:** Scripts de exemplo funcionam imediatamente
- ✅ **Configuração:** Templates de config geram caminhos corretos

### Fluxo Corrigido:

```
1. Usuário clona repositório
   └─→ git clone https://github.com/gmmsb-lncc/docktkinase.git

2. Estrutura automática está pronta
   └─→ docktkinase/FM4M/ existe no repositório

3. Configuração funciona sem modificações
   └─→ src/stratification_config.json aponta para ../FM4M

4. Pipeline executa corretamente
   └─→ Modelo FM4M carregado de docktkinase/FM4M/
```

---

## 🔄 Comandos Git

### Verificar Mudanças:

```bash
git status
git diff src/stratification_config.json
git diff src/build/example_usage.py
```

### Commit das Correções:

```bash
git add src/stratification_config.json src/build/example_usage.py
git commit -m "fix: tornar caminhos FM4M relativos ao repositório

- Substituir caminho absoluto ${HOME}/... por ../FM4M
- Aumentar portabilidade entre máquinas diferentes
- Facilitar colaboração e integração CI/CD
- Garantir funcionamento em containers Docker

Arquivos modificados:
- src/stratification_config.json
- src/build/example_usage.py

Resolves: Dependência de path específico do desenvolvedor"
```

---

## 📝 Notas Adicionais

### Compatibilidade com Código Existente:

```python
# Código que resolve caminhos relativos automaticamente
from pathlib import Path
import json

def load_config(config_file: str) -> dict:
    """Carrega config e resolve caminhos relativos."""
    config_path = Path(config_file)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Resolver caminho FM4M relativo ao arquivo de config
    if 'fm4m_config' in config and 'model_path' in config['fm4m_config']:
        fm4m_relative = config['fm4m_config']['model_path']
        fm4m_absolute = (config_path.parent / fm4m_relative).resolve()
        config['fm4m_config']['model_path'] = str(fm4m_absolute)
    
    return config
```

### Retrocompatibilidade:

- ✅ **Caminhos absolutos ainda funcionam** (se existirem)
- ✅ **Caminhos relativos são resolvidos corretamente**
- ✅ **Código detecta automaticamente qual tipo usar**

---

## ✅ Validação Final

### Checklist de Verificação:

- [x] Caminhos absolutos identificados
- [x] Caminhos relativos aplicados nos arquivos de produção
- [x] Arquivos legacy analisados e documentados
- [x] Testes de resolução de caminho executados
- [x] Documentação criada (este arquivo)
- [x] Mudanças prontas para commit

### Status:

**✅ CORREÇÃO COMPLETA E VALIDADA**

---

## 🎯 Conclusão

A substituição de caminhos absolutos por caminhos relativos **elimina a dependência de estruturas de diretório específicas de máquinas individuais**, tornando o projeto:

- **Mais profissional**
- **Mais colaborativo**
- **Mais robusto**
- **Pronto para produção**

Esta é uma **correção crítica de portabilidade** que beneficia todo o ciclo de vida do desenvolvimento do projeto.

---

**Autor:** GitHub Copilot  
**Revisão:** Sulfierry  
**Data:** 21/10/2025  
**Status:** ✅ Implementado
