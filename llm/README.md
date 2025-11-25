# LLM - Language Models Directory

Esta pasta contém todos os modelos de linguagem e caches relacionados.

## Estrutura

```
llm/
├── README.md                    # Este arquivo
└── models_cache/                # Cache de modelos baixados
    ├── README.md                # Documentação do cache
    ├── ESM/                     # Modelos ESM-2 (proteínas)
    │   ├── checkpoints/         # Checkpoints baixados
    │   ├── hub/                 # PyTorch Hub cache
    │   └── offload/             # CPU offload para modelos grandes
    ├── ESM3/                    # Modelos ESM-C/ESM-3 (proteínas)
    └── embeddings/              # Cache de embeddings gerados
```

## Modelos Suportados

### ESM-2 (Proteínas)
- `esm2_t6_8M_UR50D` - 8M parâmetros (~31 MB)
- `esm2_t12_35M_UR50D` - 35M parâmetros (~138 MB)
- `esm2_t30_150M_UR50D` - 150M parâmetros (~573 MB)
- `esm2_t33_650M_UR50D` - 650M parâmetros (~2.5 GB)
- `esm2_t36_3B_UR50D` - 3B parâmetros (~11 GB)
- `esm2_t48_15B_UR50D` - 15B parâmetros (~55 GB)

### ESM-C (Proteínas)
- `esmc-300m-2024-12` - 300M parâmetros
- `esmc-600m-2024-12` - 600M parâmetros

### FM4M (Ligantes/Moléculas)
- SMI-TED - Embeddings de SMILES

## Uso

Os modelos são baixados automaticamente na primeira execução.
Para forçar o download antecipado:

```bash
# ESM-2
python -c "import torch; torch.hub.load('facebookresearch/esm:main', 'esm2_t33_650M_UR50D')"

# ESM-C
python scripts/download_esmc_models.py --model esmc-300m-2024-12
```

## Notas

- Esta pasta é ignorada pelo Git (`.gitignore`)
- Os modelos podem ocupar vários GB de espaço
- Use `--esm-model` para selecionar qual modelo usar no pipeline
