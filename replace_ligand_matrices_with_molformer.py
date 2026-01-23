#!/usr/bin/env python3
"""
Script para substituir as matrizes de ligantes existentes pelas matrizes MoLFormer (representações por token).

Uso:
    python replace_ligand_matrices_with_molformer.py [dataset_type]

    dataset_type: "human", "non_human" ou "all" (default: "all")
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch


# Configurações para cada tipo de dataset
DATASET_CONFIGS = {
    "human": {
        "dataset_path": "tests/datasets/kinase_human_compounds.tsv",
        "base_dir": "results/protein_model_benchmark_human_v2",
    },
    "non_human": {
        "dataset_path": "tests/datasets/kinase_non_human_compounds.tsv",
        "base_dir": "results/protein_model_benchmark_non_human_v2",
    }
}


def get_available_protein_models(base_dir: Path) -> list:
    """Detecta automaticamente os modelos de proteína disponíveis no diretório."""
    models = []
    if not base_dir.exists():
        return models

    for item in base_dir.iterdir():
        if item.is_dir() and not item.name.startswith("logs_"):
            # Verificar se tem a estrutura esperada (build/ligand_matrices)
            ligand_dir = item / "build" / "ligand_matrices"
            if ligand_dir.exists():
                models.append(item.name)

    return sorted(models)


def rename_legacy_molformer_files(molformer_dir: Path) -> int:
    """
    Renomeia arquivos com nome antigo (_molformer_matrix.npy) para o novo padrao (_matrix.npy).

    Returns:
        Numero de arquivos renomeados
    """
    renamed = 0
    if not molformer_dir.exists():
        return renamed

    for old_file in molformer_dir.glob("*_molformer_matrix.npy"):
        new_name = old_file.name.replace("_molformer_matrix.npy", "_matrix.npy")
        new_file = molformer_dir / new_name
        if not new_file.exists():
            old_file.rename(new_file)
            renamed += 1

    return renamed


def replace_ligand_matrices_with_molformer(dataset_type: str = "all"):
    """Substitui as matrizes de ligantes existentes pelas matrizes MoLFormer (representações por token)."""

    # Determinar quais datasets processar
    if dataset_type == "all":
        datasets_to_process = ["human", "non_human"]
    elif dataset_type in DATASET_CONFIGS:
        datasets_to_process = [dataset_type]
    else:
        print(f"Tipo de dataset invalido: {dataset_type}")
        print(f"Opcoes validas: human, non_human, all")
        return False

    print(f"Substituindo matrizes de ligantes por matrizes MoLFormer (representacoes por token)...")
    print(f"Datasets a processar: {datasets_to_process}")

    # Carregar modelo MoLFormer uma vez (compartilhado entre datasets)
    print("\nCarregando modelo MoLFormer...")
    try:
        model_path = "llm/models_cache/molformer/model"
        tokenizer_path = "llm/models_cache/molformer/tokenizer"

        if not (Path(model_path).exists() and Path(tokenizer_path).exists()):
            print(f"Modelos MoLFormer nao encontrados em: {model_path} ou {tokenizer_path}")
            return False

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True)
        model.eval()
        print("Modelo MoLFormer carregado")
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        return False

    # Processar cada dataset
    for ds_type in datasets_to_process:
        config = DATASET_CONFIGS[ds_type]
        dataset_path = config["dataset_path"]
        base_dir = Path(config["base_dir"])

        print(f"\n{'='*60}")
        print(f"Processando dataset: {ds_type.upper()}")
        print(f"{'='*60}")

        # Carregar dataset para obter SMILES
        print(f"Carregando dataset: {dataset_path}")

        if not Path(dataset_path).exists():
            print(f"Dataset nao encontrado: {dataset_path}")
            continue

        df = pd.read_csv(dataset_path, sep='\t')
        smiles_map = dict(zip(df['chembl_id'], df['canonical_smiles']))
        print(f"Dataset carregado: {len(df)} entradas, {len(smiles_map)} SMILES unicos")

        # Detectar modelos de proteína disponíveis
        protein_models = get_available_protein_models(base_dir)

        if not protein_models:
            print(f"Nenhum modelo de proteina encontrado em: {base_dir}")
            continue

        print(f"Modelos de proteina encontrados: {protein_models}")
    
        # Processar cada modelo de proteína
        for model_name in protein_models:
            model_dir = base_dir / model_name
            if not model_dir.exists():
                print(f"  Diretorio nao encontrado: {model_dir}")
                continue

            print(f"\n--- Processando {model_name} ---")

            # Diretório de matrizes de ligantes existentes
            ligand_matrix_dir = model_dir / "build" / "ligand_matrices"
            if not ligand_matrix_dir.exists():
                print(f"  Diretorio de matrizes de ligantes nao encontrado: {ligand_matrix_dir}")
                continue

            # Diretório para novas matrizes MoLFormer
            molformer_matrix_dir = model_dir / "build" / "molformer_matrix"
            molformer_matrix_dir.mkdir(exist_ok=True)

            # Renomear arquivos com nome antigo (legacy) se existirem
            renamed = rename_legacy_molformer_files(molformer_matrix_dir)
            if renamed > 0:
                print(f"  Renomeados {renamed} arquivos com nome antigo para padrao _matrix.npy")

            # Obter lista de arquivos de matrizes existentes
            matrix_files = list(ligand_matrix_dir.glob("*_matrix.npy"))
            print(f"  Processando {len(matrix_files)} matrizes existentes...")

            processed = 0
            skipped = 0
            errors = 0

            for matrix_file in matrix_files:
                # Extrair ID do arquivo (ex: CHEMBL1234567_matrix.npy -> CHEMBL1234567)
                chembl_id = matrix_file.name.replace("_matrix.npy", "")

                # Verificar se já existe a matriz MoLFormer (para continuar de onde parou)
                # Nota: usamos _matrix.npy para compatibilidade com MatrixEmbeddingDataset
                new_matrix_file = molformer_matrix_dir / f"{chembl_id}_matrix.npy"
                if new_matrix_file.exists():
                    skipped += 1
                    continue

                if chembl_id in smiles_map:
                    smiles = smiles_map[chembl_id]

                    try:
                        # Tokenizar SMILES
                        inputs = tokenizer(
                            smiles,
                            return_tensors="pt",
                            padding=False,
                            truncation=True,
                            max_length=512
                        )

                        # Gerar matriz por token
                        with torch.no_grad():
                            outputs = model(**inputs, output_hidden_states=True)

                        # Obter representações por token (última camada de estados ocultos)
                        last_hidden_state = outputs.hidden_states[-1][0]  # Shape: [seq_len, hidden_size]
                        matrix = last_hidden_state.cpu().numpy()

                        # Salvar nova matriz MoLFormer
                        np.save(new_matrix_file, matrix)

                        processed += 1

                        # Log de progresso a cada 1000 matrizes
                        if processed % 1000 == 0:
                            print(f"    Progresso: {processed} matrizes processadas...")

                    except Exception as e:
                        errors += 1
                        if errors <= 5:  # Limitar logs de erro
                            print(f"    Erro ao processar {chembl_id}: {e}")
                else:
                    errors += 1

            print(f"  {model_name}: {processed} novas, {skipped} ja existentes, {errors} erros")
            print(f"  Matrizes em: {molformer_matrix_dir}")

    print(f"\nProcessamento concluido!")
    print("As matrizes de ligantes MoLFormer (representacoes por token) foram geradas.")
    print("Cada matriz tem formato [n_tokens, 768] com representacoes por atomo/token.")
    return True

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Gerar matrizes MoLFormer para ligantes (representacoes por token)."
    )
    parser.add_argument(
        "dataset_type",
        nargs="?",
        default="all",
        choices=["human", "non_human", "all"],
        help="Tipo de dataset a processar: 'human', 'non_human' ou 'all' (default: all)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    os.chdir("/media/storage/leon/semantic-screening")

    args = parse_args()
    success = replace_ligand_matrices_with_molformer(args.dataset_type)

    if success:
        print("\nSucesso! Matrizes de ligantes MoLFormer geradas com sucesso.")
        print("\nAs matrizes estao disponiveis em:")
        if args.dataset_type in ["human", "all"]:
            print("  results/protein_model_benchmark_human_v2/[model]/build/molformer_matrix/")
        if args.dataset_type in ["non_human", "all"]:
            print("  results/protein_model_benchmark_non_human_v2/[model]/build/molformer_matrix/")
        print("\nAs matrizes estao prontas para mecanismos de atencao cruzada!")
    else:
        print("\nErro durante o processamento.")
        sys.exit(1)