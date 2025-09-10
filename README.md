# docktkinase

Pipeline para processamento de dados de interação proteína-ligante para quinases não humanas.

## Estrutura do Projeto

```
docktkinase/
├── src/
│   ├── build/           # Scripts principais do pipeline
│   ├── database/       # Dados de entrada (TSV, CSV)
│   └── materials/       # Materiais auxiliares
├── non_humans/          # Diretório de trabalho principal
│   ├── docktkinase.py   # Arquivo de configuração (MODIFIQUE APENAS ESTE)
│   ├── interface.py     # Interface principal (NÃO MODIFIQUE)
│   ├── ligand/          # Arquivos SMILES dos ligantes
│   ├── ligand_embeddings/  # Embeddings dos ligantes
│   ├── protein/         # Sequências proteicas em formato FASTA
│   ├── protein_embeddings/  # Embeddings das proteínas
│   └── concatenated_embeddings/  # Saída final do pipeline
└── env/                 # Ambiente Conda
```

## Configuração

**ATENÇÃO**: Apenas o arquivo `non_humans/docktkinase.py` deve ser modificado pelos usuários.

### Passos para configurar:

1. Edite `non_humans/docktkinase.py`:
   ```python
   # Configure os seguintes parâmetros:
   INPUT_TSV_FILENAME = "seu_arquivo_de_dados.tsv"
   EMBEDDING_MATRIX_FILENAME = "nome_da_matriz_saida.tsv"
   OUTPUT_FOLDER_NAME = "nome_da_pasta_de_saida"
   ```

2. Certifique-se de que seu arquivo TSV está em `src/database/`

## Execução

```bash
cd non_humans
python interface.py
```

## Arquivos de Configuração

### docktkinase.py
Arquivo onde os usuários definem as configurações do pipeline:
- `INPUT_TSV_FILENAME`: Nome do arquivo TSV de entrada (deve estar em src/database/)
- `EMBEDDING_MATRIX_FILENAME`: Nome do arquivo de saída da matriz de embeddings
- `OUTPUT_FOLDER_NAME`: Nome da pasta onde todos os resultados serão salvos
- Configurações automáticas do ambiente Conda

### interface.py
Interface principal do pipeline (NÃO MODIFICAR):
- Coordena a execução de todas as etapas do pipeline
- Importa as configurações do `docktkinase.py`
- Executa os scripts em ordem correta
- Monitora o progresso e trata erros

## Pipeline de Processamento

1. **Geração de Embeddings**:
   - Processa ligantes (SMILES → embeddings ChemBERTa)
   - Processa proteínas (sequências → embeddings ESM)

2. **Construção da Matriz**:
   - Combina embeddings de ligantes e proteínas
   - Gera matrizes concatenadas

3. **Geração de Labels**:
   - Cria labels de interação
   - Gera labels binárias

4. **Validação Final**:
   - Verifica integridade dos dados
   - Confirma alinhamento das matrizes

## Requisitos

- Python 3.10+
- Ambiente Conda configurado
- Dependências listadas em `environment.yml`

## Solução de Problemas

### Erros Comuns

1. **Arquivo de entrada não encontrado**:
   - Verifique se o arquivo TSV está em `src/database/`
   - Confirme o nome do arquivo em `docktkinase.py`

2. **Problemas com ambiente Conda**:
   - Certifique-se de que o ambiente está ativado
   - Verifique se `CONDA_PREFIX` está definido

3. **Erros de permissão**:
   - Certifique-se de ter permissões de escrita no diretório

## Contribuição

Este projeto segue o padrão de separação de configuração e código:
- Arquivos em `src/` e `interface.py` são imutáveis
- Configurações personalizadas vão em `docktkinase.py`