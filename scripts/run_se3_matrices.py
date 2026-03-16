#!/usr/bin/env python3
"""Generate per-ligand structural vectors for the --se3 pipeline.

This script mirrors the behavior and layout of ``run_chemberta_matrices.py``:
- Reads a benchmark dataset TSV.
- Iterates unique ``chembl_id`` / ``canonical_smiles`` pairs.
- Writes one ``.npy`` vector per ligand to:
  ``results/protein_model_benchmark_{dataset}_v2/{embedding}/build/se3_features/``
- File naming: ``{chembl_id}_se3.npy``

Modes:
- ``auto`` (default): try SE3Transformer inference when dependencies and checkpoint
  are available; otherwise fallback to RDKit 3D structural descriptors.
- ``se3transformer``: require SE3Transformer inference.
- ``rdkit``: force RDKit structural descriptor generation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm


DATASET_PATHS = {
    "human": "tests/datasets/kinase_human_compounds.tsv",
    "non_human": "tests/datasets/kinase_non_human_compounds.tsv",
    "all": "tests/datasets/kinase_all_compounds.tsv",
}

EMBEDDING_BASE = "results/protein_model_benchmark_{dataset}_v2"

SUPPORTED_EMBEDDINGS = {
    "8M": "esm2_t6_8M_UR50D",
    "150M": "esm2_t30_150M_UR50D",
    "650M": "esm2_t33_650M_UR50D",
}


def _rdkit_se3_vector(smiles: str) -> np.ndarray:
    """Build a fixed-size structural vector from 3D conformer geometry.

    This fallback is SE(3)-invariant (distance/statistical descriptors),
    stable, and does not require DGL/SE3Transformer runtime.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

    # 6 global descriptors
    global_desc = np.zeros(6, dtype=np.float32)

    # Atom histogram over common chemistry elements (10 dims)
    atom_vocab = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
    atom_hist = np.zeros(len(atom_vocab), dtype=np.float32)

    # Pairwise distance histogram (40 dims, 0..20A)
    dist_hist = np.zeros(40, dtype=np.float32)

    # Inertia-like 3D summary (3 dims)
    inertia = np.zeros(3, dtype=np.float32)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.concatenate([global_desc, atom_hist, dist_hist, inertia]).astype(np.float32)

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        # Retry with random coords for hard molecules
        params.useRandomCoords = True
        status = AllChem.EmbedMolecule(mol, params)

    if status == 0:
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            pass

    mol_no_h = Chem.RemoveHs(mol)

    # Global descriptors
    try:
        global_desc[0] = float(mol_no_h.GetNumAtoms())
        global_desc[1] = float(mol_no_h.GetNumBonds())
        global_desc[2] = float(Descriptors.MolWt(mol_no_h))
        global_desc[3] = float(rdMolDescriptors.CalcTPSA(mol_no_h))
        global_desc[4] = float(Descriptors.MolLogP(mol_no_h))
        global_desc[5] = float(rdMolDescriptors.CalcNumRings(mol_no_h))
    except Exception:
        pass

    # Atom histogram
    z_to_idx = {z: i for i, z in enumerate(atom_vocab)}
    for atom in mol_no_h.GetAtoms():
        idx = z_to_idx.get(atom.GetAtomicNum())
        if idx is not None:
            atom_hist[idx] += 1.0
    if atom_hist.sum() > 0:
        atom_hist /= atom_hist.sum()

    # 3D descriptors from conformer (if available)
    if mol_no_h.GetNumConformers() > 0:
        conf = mol_no_h.GetConformer()
        n = mol_no_h.GetNumAtoms()

        if n >= 2:
            coords = np.array(
                [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in range(n)],
                dtype=np.float32,
            )
            dists: list[float] = []
            for i in range(n):
                for j in range(i + 1, n):
                    d = float(np.linalg.norm(coords[i] - coords[j]))
                    dists.append(d)

            if dists:
                d_arr = np.asarray(dists, dtype=np.float32)
                hist, _ = np.histogram(d_arr, bins=40, range=(0.0, 20.0), density=False)
                dist_hist = hist.astype(np.float32)
                if dist_hist.sum() > 0:
                    dist_hist /= dist_hist.sum()

                centered = coords - coords.mean(axis=0, keepdims=True)
                cov = centered.T @ centered / max(n - 1, 1)
                eigvals = np.linalg.eigvalsh(cov)
                eigvals = np.sort(np.clip(eigvals, 0.0, None))[::-1]
                inertia = eigvals[:3].astype(np.float32)

    vec = np.concatenate([global_desc, atom_hist, dist_hist, inertia]).astype(np.float32)
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)


def _try_build_se3transformer_encoder(
    se3_repo: str,
    checkpoint_path: str,
    device: str,
    use_layer_norm: bool,
    norm: bool,
) -> tuple[Any, Any, Any, Any] | None:
    """Try building SE3Transformer encoder for pooled structural embeddings.

    Returns tuple ``(model, device_obj, dgl_module, rdkit_module)`` or ``None``.
    """
    import importlib
    import torch

    repo_path = Path(se3_repo)
    if not repo_path.exists():
        print(f"WARNING: SE3 repo not found: {repo_path}")
        return None

    if not Path(checkpoint_path).exists():
        print(f"WARNING: SE3 checkpoint not found: {checkpoint_path}")
        return None

    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    try:
        dgl = importlib.import_module("dgl")
        se3_model_module = importlib.import_module("se3_transformer.model")
        from se3_transformer.model import Fiber, SE3TransformerPooled
    except Exception as exc:
        print(f"WARNING: Could not import SE3Transformer stack ({exc}).")
        return None

    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    model = SE3TransformerPooled(
        fiber_in=Fiber({0: 6}),
        fiber_out=Fiber({0: 128}),
        fiber_edge=Fiber({0: 4}),
        num_degrees=4,
        num_channels=32,
        output_dim=1,
        num_layers=7,
        num_heads=8,
        channels_div=2,
        pooling="max",
        norm=norm,
        use_layer_norm=use_layer_norm,
    ).to(dev)

    checkpoint = torch.load(str(checkpoint_path), map_location=dev)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model, dev, dgl, se3_model_module


def _atom_features_rdkit(mol) -> np.ndarray:
    """Node features with shape [N, 6] to match SE3Transformer input dimensionality."""
    from rdkit.Chem.rdchem import HybridizationType

    feats: list[list[float]] = []
    for atom in mol.GetAtoms():
        hyb = atom.GetHybridization()
        feats.append(
            [
                atom.GetAtomicNum() / 100.0,
                atom.GetTotalDegree() / 6.0,
                atom.GetFormalCharge() / 5.0,
                1.0 if atom.GetIsAromatic() else 0.0,
                1.0 if hyb == HybridizationType.SP else 0.0,
                1.0 if hyb == HybridizationType.SP2 else 0.0,
            ]
        )
    return np.asarray(feats, dtype=np.float32)


def _bond_features_rdkit(mol) -> tuple[np.ndarray, np.ndarray]:
    """Edge index (src,dst) and edge attributes [E,4] from RDKit bonds."""
    from rdkit.Chem.rdchem import BondType

    src: list[int] = []
    dst: list[int] = []
    edge_attr: list[list[float]] = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bt = bond.GetBondType()
        feat = [
            1.0 if bt == BondType.SINGLE else 0.0,
            1.0 if bt == BondType.DOUBLE else 0.0,
            1.0 if bt == BondType.TRIPLE else 0.0,
            1.0 if bond.GetIsAromatic() else 0.0,
        ]

        src.extend([i, j])
        dst.extend([j, i])
        edge_attr.extend([feat, feat])

    return np.asarray(src), np.asarray(dst), np.asarray(edge_attr, dtype=np.float32)


def _se3transformer_vector(smiles: str, model, dev, dgl) -> np.ndarray:
    """Compute pooled SE3Transformer embedding for one molecule.

    Returns [128] vector (fiber_out features for type-0 pooled branch).
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    import torch

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((128,), dtype=np.float32)

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        params.useRandomCoords = True
        status = AllChem.EmbedMolecule(mol, params)
    if status == 0:
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            pass

    mol = Chem.RemoveHs(mol)
    n = mol.GetNumAtoms()
    if n == 0:
        return np.zeros((128,), dtype=np.float32)

    if mol.GetNumConformers() == 0:
        return np.zeros((128,), dtype=np.float32)

    conf = mol.GetConformer()
    pos = np.array(
        [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in range(n)],
        dtype=np.float32,
    )

    node_attr = _atom_features_rdkit(mol)
    src, dst, edge_attr = _bond_features_rdkit(mol)

    if len(src) == 0:
        return np.zeros((128,), dtype=np.float32)

    graph = dgl.graph((src, dst), num_nodes=n)
    graph.ndata["attr"] = torch.from_numpy(node_attr)
    graph.ndata["pos"] = torch.from_numpy(pos)

    rel_pos = graph.ndata["pos"][dst] - graph.ndata["pos"][src]
    graph.edata["rel_pos"] = rel_pos
    graph.edata["edge_attr"] = torch.from_numpy(edge_attr)

    graph = graph.to(dev)
    node_feats = {"0": graph.ndata["attr"][:, :6, None]}
    edge_feats = {"0": graph.edata["edge_attr"][:, :4, None]}

    with torch.no_grad():
        pooled = model.transformer(graph, node_feats, edge_feats).squeeze(-1)
    out = pooled.detach().cpu().numpy().reshape(-1).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def generate_se3_matrices(
    dataset: str,
    embedding_names: list[str],
    mode: str = "auto",
    device: str = "cuda",
    force: bool = False,
    se3_repo: str | None = None,
    checkpoint_path: str | None = None,
    use_layer_norm: bool = True,
    norm: bool = True,
) -> None:
    """Generate per-ligand structural vectors saved as ``*_se3.npy``."""
    dataset_path = Path(DATASET_PATHS[dataset])
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {dataset_path}")
        sys.exit(1)

    df = pd.read_csv(dataset_path, sep="\t")
    unique_smiles = df.drop_duplicates(subset="chembl_id")[["chembl_id", "canonical_smiles"]].copy()
    print(f"Dataset: {dataset} — {len(unique_smiles)} unique compounds")

    encoder = None
    if mode in {"auto", "se3transformer"}:
        if se3_repo and checkpoint_path:
            print("\nTrying SE3Transformer encoder...")
            encoder = _try_build_se3transformer_encoder(
                se3_repo=se3_repo,
                checkpoint_path=checkpoint_path,
                device=device,
                use_layer_norm=use_layer_norm,
                norm=norm,
            )
        elif mode == "se3transformer":
            print("ERROR: --mode se3transformer requires --se3-repo and --checkpoint-path")
            sys.exit(1)

    if encoder is None and mode == "se3transformer":
        print("ERROR: Could not initialize SE3Transformer encoder in strict mode.")
        sys.exit(1)

    if encoder is None:
        print("\nUsing RDKit structural descriptor fallback (SE(3)-invariant geometry vector).")

    for emb_short in embedding_names:
        emb_full = SUPPORTED_EMBEDDINGS.get(emb_short, emb_short)
        base_dir = Path(EMBEDDING_BASE.format(dataset=dataset)) / emb_full / "build"
        output_dir = base_dir / "se3_features"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Embedding: {emb_full}  |  Output: {output_dir}")
        print(f"{'='*60}")

        skipped = 0
        generated = 0

        for _, row in tqdm(unique_smiles.iterrows(), total=len(unique_smiles), desc="SE3"):
            chembl_id = row["chembl_id"]
            smiles = row["canonical_smiles"]
            out_file = output_dir / f"{chembl_id}_se3.npy"

            if out_file.exists() and not force:
                skipped += 1
                continue

            if encoder is not None:
                model, dev, dgl, _ = encoder
                try:
                    vec = _se3transformer_vector(smiles, model, dev, dgl)
                except Exception:
                    vec = _rdkit_se3_vector(smiles)
            else:
                vec = _rdkit_se3_vector(smiles)

            np.save(out_file, vec.astype(np.float32))
            generated += 1

        print(f"  Generated: {generated}, Skipped (exists): {skipped}")

    print("\nDone.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SE3 structural ligand vectors as *_se3.npy"
    )
    parser.add_argument(
        "--dataset",
        "-d",
        required=True,
        choices=list(DATASET_PATHS.keys()),
        help="Dataset to process",
    )
    parser.add_argument(
        "--embedding",
        "-e",
        nargs="+",
        default=["650M"],
        help="Protein embedding shorthand(s) whose build/ dir to use (default: 650M)",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "se3transformer", "rdkit"],
        default="auto",
        help="Generation mode (default: auto)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for SE3Transformer inference (cuda or cpu)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing vectors",
    )
    parser.add_argument(
        "--se3-repo",
        default=None,
        help="Path to NVIDIA SE3Transformer repo root (contains se3_transformer/)",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Path to SE3Transformer checkpoint (.pth) for inference",
    )
    parser.add_argument(
        "--use-layer-norm",
        action="store_true",
        help="Enable layer norm when building SE3Transformer model.",
    )
    parser.add_argument(
        "--norm",
        action="store_true",
        help="Enable equivariant norm when building SE3Transformer model.",
    )

    args = parser.parse_args()

    generate_se3_matrices(
        dataset=args.dataset,
        embedding_names=args.embedding,
        mode=args.mode,
        device=args.device,
        force=args.force,
        se3_repo=args.se3_repo,
        checkpoint_path=args.checkpoint_path,
        use_layer_norm=args.use_layer_norm,
        norm=args.norm,
    )


if __name__ == "__main__":
    main()
