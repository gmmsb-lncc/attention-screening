# 📦 Datasets de Referência

Esta pasta contém os datasets originais utilizados para desenvolvimento e testes do projeto attention-screening.

## 📊 Datasets Disponíveis

### 1. `kinase_all_compounds.tsv` (415 MB)
- **Descrição**: Dataset completo com todos os compostos kinase
- **Origem**: ChEMBL 35
- **Conteúdo**: Compostos humanos + não-humanos
- **Uso**: Testes completos, benchmarks, validação geral

### 2. `kinase_human_compounds.tsv` (404 MB)
- **Descrição**: Dataset apenas com compostos kinase humanos
- **Origem**: Filtrado de kinase_all_compounds.tsv
- **Conteúdo**: Apenas compostos de proteínas humanas
- **Uso**: Testes específicos para aplicações médicas/farmacêuticas

### 3. `kinase_non_human_compounds.tsv` (11 MB)
- **Descrição**: Dataset com compostos kinase não-humanos
- **Origem**: Filtrado de kinase_all_compounds.tsv
- **Conteúdo**: Compostos de outras espécies (modelo, patógenos, etc.)
- **Uso**: Estudos comparativos, pesquisa evolutiva

## 🔍 Formato dos Dados

Todos os arquivos TSV (Tab-Separated Values) contêm as seguintes colunas principais:
- `molregno`: ID único do composto (ChEMBL)
- `canonical_smiles`: Estrutura química em formato SMILES
- `target_kinase`: Nome da proteína kinase alvo
- `seq_id`: ID da sequência proteica
- `seq`: Sequência de aminoácidos da proteína
- `standard_type`: Tipo de medida (IC50, Ki, Kd, etc.)
- `standard_value`: Valor da atividade (nM)
- Outras colunas de metadados

## 📝 Notas Importantes

1. **Tamanho**: Os arquivos são grandes (415 MB total). Certifique-se de ter espaço em disco suficiente.

2. **Git LFS**: Estes arquivos NÃO estão versionados no Git devido ao tamanho. Eles devem ser baixados separadamente ou gerados através dos scripts em `src/database/`.

3. **Processamento**: Para usar estes datasets no pipeline:
   ```python
   from src.build.pipeline.mlp_pipeline import MLPPipeline
   
   pipeline = MLPPipeline(
       input_file="tests/datasets/kinase_all_compounds.tsv",
       output_dir="output"
   )
   pipeline.run_pipeline(num_samples=1000)
   ```

4. **Geração de Novos Datasets**: 
   - Scripts SQL originais: `src/database/sql/`
   - Script de split: `src/database/split_kinase_data.py`
   - Remoção de redundância: `src/database/remove_redundance.py`

## 🚀 Uso Recomendado

### Testes Rápidos (< 5 min)
```bash
python -m src.build.pipeline.mlp_pipeline \
    --input tests/datasets/kinase_all_compounds.tsv \
    --samples 100 \
    --output test_output
```

### Validação Completa (~ 1 hora)
```bash
python -m src.build.pipeline.mlp_pipeline \
    --input tests/datasets/kinase_all_compounds.tsv \
    --samples 10000 \
    --output full_output
```

### Testes Específicos para Humanos
```bash
python -m src.build.pipeline.mlp_pipeline \
    --input tests/datasets/kinase_human_compounds.tsv \
    --samples 1000 \
    --output human_test
```

## 📚 Referências

- **ChEMBL Database**: https://www.ebi.ac.uk/chembl/
- **Versão**: ChEMBL 35 (2024)
- **Documentação do Projeto**: `/docs/USER_GUIDE.md`
- **Pipeline de Processamento**: `/docs/PIPELINE_SUCCESS_REPORT.md`

---

**Última atualização**: 21 de outubro de 2025
**Movido de**: `src/kinase_*/` → `tests/datasets/`
