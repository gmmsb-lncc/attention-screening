# Resolução de Dependências - 21/10/2025

## 🔄 Resumo do Processo

### Problema Inicial
Pipeline falhava com múltiplas dependências ausentes do FM4M (ligand embedding model).

### Dependências Instaladas (em ordem)

1. **umap-learn** - Para redução de dimensionalidade
2. **xgboost** - Para métodos de ensemble
3. **selfies** - Para representação SELFIES de moléculas
4. **mordred** - Para descritores moleculares  
5. **numba** - Para computação JIT
6. **networkx-2.8** - Para análise de grafos (downgrade para compatibilidade)
7. **ase** - Atomic Simulation Environment
8. **rdkit-2025.9.1** - Para processamento de SMILES
9. **torch-scatter-2.1.2** - Para operações esparsas no PyTorch

### Dependências de Sistema

10. **libomp** (Homebrew) - OpenMP runtime necessário para XGBoost no Mac

### Compatibilidade PyTorch

**CRÍTICO**: PyTorch 2.8.0 é necessário!

- ❌ PyTorch 2.9.0 (original) - Incompatível com torch-scatter pré-compilado
- ✅ PyTorch 2.8.0 (downgrade) - Compatível com wheels do PyG

**Comando de instalação**:
```bash
pip install torch==2.8.0
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.8.0+cpu.html
```

### Pacote Adicional

11. **torch-nl** - Neighborlist computation para PosEGNN

```bash
pip install torch-nl
```

## 📦 FM4M Model Files

**PENDENTE**: Download de arquivos de modelo (~1.16GB)

```bash
cd FM4M
python download_model_files.py
```

**Arquivos necessários**:
- `bert_vocab_curated.txt` (~17KB) ✅
- `smi-ted-Light_40.pt` (~1.16GB) ⏳ Baixando...

**Local esperado**: `/Users/sulfierry/docktkinase/FM4M/model_files/`

## 🔧 Setup.py Aprimorado

Implementada verificação inteligente de pacotes:

**Função `check_package_installed()`**:
- Mapeia nomes de pacotes para imports (ex: fair-esm → esm)
- Testa import real ao invés de confiar apenas no pip
- Evita reinstalações desnecessárias

**Benefícios**:
- ⚡ Muito mais rápido em execuções subsequentes
- 📊 Relatório detalhado: "já instalado" vs "recém instalado"
- 🎯 Instala apenas o que falta

## 🐛 Problemas Resolvidos

### 1. Segmentation Fault (std::length_error)
**Causa**: Incompatibilidade ABI entre PyTorch 2.9.0 e torch-scatter 2.1.2  
**Solução**: Downgrade para PyTorch 2.8.0

### 2. XGBoost - Library not loaded: libomp.dylib
**Causa**: OpenMP runtime ausente no macOS  
**Solução**: `brew install libomp`

### 3. ModuleNotFoundError: torch_nl
**Causa**: Dependência não documentada do PosEGNN  
**Solução**: `pip install torch-nl`

### 4. SMI-TED model files not found
**Causa**: Arquivos de modelo não baixados após instalação  
**Solução**: Executar `python FM4M/download_model_files.py`

## 📝 Lista Completa de Dependências

```text
# Core ML
torch==2.8.0
torch-scatter==2.1.2
torch-nl==0.3

# Protein Embeddings
fair-esm==2.0.0

# Ligand Embeddings (FM4M)
rdkit-2025.9.1
umap-learn==0.6.0
xgboost==2.3.2
selfies==2.2.1
mordred==1.2.0
numba==0.67.1
networkx==2.8.8
ase==3.26.0

# Sistema (macOS)
libomp (via Homebrew)
```

## ⚠️ Notas Importantes

1. **Ordem de instalação importa**: torch-scatter requer PyTorch instalado primeiro
2. **Mac M1 requer libomp**: Não opcional para XGBoost
3. **networkx 2.8**: Versões mais novas causam conflitos
4. **Model files**: Devem ser baixados separadamente (~1.16GB)
5. **PyTorch 2.8.0**: Versão máxima compatível com torch-scatter pré-compilado

## ✅ Status Final

| Componente | Status |
|------------|--------|
| Python env | ✅ Configurado (Python 3.12) |
| Dependências básicas | ✅ Instaladas |
| Dependências FM4M | ✅ Instaladas |
| PyTorch 2.8.0 | ✅ Instalado |
| torch-scatter | ✅ Compatível |
| OpenMP | ✅ Instalado (Homebrew) |
| torch-nl | ✅ Instalado |
| FM4M model files | ⏳ **Baixando (~40% completo)** |
| Protein embeddings | ✅ Funcionando (275 sequências em 8ms) |
| Ligand embeddings | ⏳ Aguardando model files |

## 🚀 Próximos Passos

1. ⏳ Aguardar download completo dos model files FM4M
2. 🔄 Re-executar `python test_pipeline_small.py`
3. ✅ Validar ligand embeddings funcionam
4. 📊 Verificar construção de matriz e stratificação
5. 🎯 Executar pipeline completo com 491k amostras

## 🏆 Lições Aprendidas

1. **Dependências de ML são complexas**: FM4M tem 8 dependências não documentadas
2. **PyTorch versioning é crítico**: ABI compatibility matters
3. **macOS requer cuidado especial**: OpenMP não vem instalado
4. **Setup inteligente economiza tempo**: Check before install
5. **Model files são separados**: Não assumir que vêm com o código

---

**Última atualização**: 21/10/2025 12:19 BRT  
**Responsável**: Setup automatizado + resolução iterativa de dependências
