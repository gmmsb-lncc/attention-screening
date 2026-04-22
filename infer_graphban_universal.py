"""
Inferência GraphBAN sobre universal_val.tsv + universal_test.tsv.
Carrega checkpoint best_model, processa universal val + test, salva raw_predictions.npz.

Uso (diamante-02):
    cd <semantic-screening root>
    python infer_graphban_universal.py --corpus all --seeds 42 123 456 789 1024
"""

import argparse, os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

REPO = Path(__file__).parent.resolve()
# GraphBAN upstream coloca módulos em src/case_study/ (inductive mode)
GRAPHBAN_SRC = REPO / "GraphBAN" / "src" / "case_study"
if not GRAPHBAN_SRC.exists():
    GRAPHBAN_SRC = REPO / "GraphBAN" / "src"  # fallback para estrutura plana
sys.path.insert(0, str(GRAPHBAN_SRC))

from dataloader import DTIDataset  # type: ignore
from models import GraphBAN  # type: ignore
from torch.utils.data import DataLoader

import dgl  # fornecido pelo env 'graphban'


def graph_collate_func(batch):
    """Collate para DTIDataset do GraphBAN (case_study). Retorna (v_d, fcfps, v_p, esm)."""
    drugs, fcfps, proteins, esms = zip(*batch)
    drugs_batch = dgl.batch(drugs)
    fcfps_batch = torch.as_tensor(np.stack(fcfps), dtype=torch.float32)
    proteins_batch = torch.as_tensor(np.stack(proteins))
    esms_batch = torch.as_tensor(np.stack(esms), dtype=torch.float32)
    return drugs_batch, fcfps_batch, proteins_batch, esms_batch


UNIVERSAL_TSV = {
    "non_human": ("scaffolds_splits/output/non_human_val.tsv",
                  "scaffolds_splits/output/non_human_test.tsv"),
    "human": ("scaffolds_splits/output/human_val.tsv",
              "scaffolds_splits/output/human_test.tsv"),
    "all": ("scaffolds_splits/output/universal_val.tsv",
            "scaffolds_splits/output/universal_test.tsv"),
}


import pickle


def load_feature_lookups(corpus: str) -> tuple[dict, dict]:
    """Carrega cache GraphBAN train+val+test e constrói lookups por-SMILES (fcfp) e
    por-Protein (esm), sem recomputar ESM-1b."""
    cache_path = REPO / f"GraphBAN/results/{corpus}/feature_cache/features_extracted.pkl"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cache {cache_path} não encontrado. GraphBAN requer features pré-computadas."
        )
    print(f"    carregando cache: {cache_path.name}")
    d = pickle.load(open(cache_path, "rb"))
    fcfp_by_smiles: dict = {}
    esm_by_protein: dict = {}
    for split_name in ("train", "val", "test"):
        if split_name not in d:
            continue
        df = d[split_name]
        for _, row in df.iterrows():
            s = row["SMILES"]
            p = row["Protein"]
            if s not in fcfp_by_smiles:
                fcfp_by_smiles[s] = row["fcfp"]
            if p not in esm_by_protein:
                esm_by_protein[p] = row["esm"]
    print(f"      {len(fcfp_by_smiles)} SMILES únicos com FCFP; {len(esm_by_protein)} proteínas únicas com ESM")
    return fcfp_by_smiles, esm_by_protein


def tsv_to_graphban_df(tsv_path: Path, fcfp_by_smiles: dict, esm_by_protein: dict) -> pd.DataFrame:
    """Monta DataFrame no formato GraphBAN, reutilizando cache ESM/FCFP do treino."""
    df = pd.read_csv(tsv_path, sep="\t")
    smiles_list = df["canonical_smiles"].astype(str).tolist()
    protein_list = df["seq"].astype(str).tolist()
    fcfp_col = []
    esm_col = []
    n_missing_fcfp = 0
    n_missing_esm = 0
    for s, p in zip(smiles_list, protein_list):
        if s in fcfp_by_smiles:
            fcfp_col.append(fcfp_by_smiles[s])
        else:
            fcfp_col.append(np.zeros(384, dtype=np.float64))
            n_missing_fcfp += 1
        if p in esm_by_protein:
            esm_col.append(esm_by_protein[p])
        else:
            esm_col.append(np.zeros(1280, dtype=np.float32))
            n_missing_esm += 1
    if n_missing_fcfp or n_missing_esm:
        print(f"    AVISO: {n_missing_fcfp}/{len(smiles_list)} SMILES sem FCFP em cache; "
              f"{n_missing_esm}/{len(protein_list)} proteínas sem ESM (zero-preenchidos)")
    out = pd.DataFrame({
        "SMILES": smiles_list,
        "Protein": protein_list,
        "Y": df["label"].astype(int),
    })
    out["fcfp"] = fcfp_col
    out["esm"] = esm_col
    return out


def load_model(checkpoint_path: Path, config: dict, device: torch.device):
    model = GraphBAN(**config).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


@torch.no_grad()
def predict(model, loader, y_true_all: np.ndarray, device):
    """Inferência GraphBAN. DTIDataset retorna (v_d, fcfps, v_p, esm);
    Y vem da DataFrame externa por ordenação (shuffle=False)."""
    ps = []
    use_amp = device.type == "cuda"
    for batch in loader:
        v_d, fcfps, v_p, esms = batch
        v_d = v_d.to(device, non_blocking=True)
        v_p = v_p.to(device, non_blocking=True)
        fcfps = fcfps.to(device, non_blocking=True)
        esms = esms.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            # GraphBAN forward signature: model(v_d, v_p, fcfps, esm) — ajustar se falhar
            try:
                out = model(v_d, v_p, fcfps, esms)
            except TypeError:
                out = model(v_d, fcfps, v_p, esms)
        logits = out[-1] if isinstance(out, tuple) else out
        prob = torch.softmax(logits.float(), dim=1)[:, 1]
        ps.append(prob.cpu().numpy())
    return y_true_all, np.concatenate(ps)


def best_checkpoint(seed_dir: Path) -> Path:
    cands = sorted(seed_dir.glob("best_model_epoch_*.pth"))
    if not cands: raise FileNotFoundError(seed_dir)
    return cands[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=["non_human","human","all"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42,123,456,789,1024])
    ap.add_argument("--checkpoint-root", default="GraphBAN/results")
    ap.add_argument("--config", default="GraphBAN/configs/kinase_inductive.yaml")
    ap.add_argument("--output-dir", default="GraphBAN/results_universal")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")

    with open(REPO / args.config) as f:
        config = yaml.safe_load(f)

    val_tsv, test_tsv = UNIVERSAL_TSV[args.corpus]
    print("  montando DataFrame com features cacheadas do treino…")
    fcfp_lookup, esm_lookup = load_feature_lookups(args.corpus)
    val_df = tsv_to_graphban_df(REPO / val_tsv, fcfp_lookup, esm_lookup)
    test_df = tsv_to_graphban_df(REPO / test_tsv, fcfp_lookup, esm_lookup)
    print(f"val n={len(val_df)} pos={val_df['Y'].mean():.3f}")
    print(f"test n={len(test_df)} pos={test_df['Y'].mean():.3f}")

    use_pin = device.type == "cuda"
    val_loader = DataLoader(DTIDataset(val_df.index.values, val_df),
        batch_size=args.batch_size, shuffle=False, collate_fn=graph_collate_func,
        num_workers=args.num_workers, pin_memory=use_pin,
        persistent_workers=(args.num_workers > 0))
    test_loader = DataLoader(DTIDataset(test_df.index.values, test_df),
        batch_size=args.batch_size, shuffle=False, collate_fn=graph_collate_func,
        num_workers=args.num_workers, pin_memory=use_pin,
        persistent_workers=(args.num_workers > 0))

    out_root = REPO / args.output_dir / args.corpus
    out_root.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        sd = REPO / args.checkpoint_root / args.corpus / f"seed_{seed}"
        if not sd.exists():
            print(f"[skip] {sd}"); continue
        o = out_root / f"seed_{seed}"
        out_npz = o / "raw_predictions.npz"
        if out_npz.exists():
            print(f"[seed {seed}] já concluído → {out_npz.relative_to(REPO)} (pulando)")
            continue
        ckpt = best_checkpoint(sd)
        print(f"[seed {seed}] {ckpt.relative_to(REPO)}")
        m = load_model(ckpt, config, device)
        vy, vp = predict(m, val_loader, val_df["Y"].to_numpy(), device)
        ty, tp = predict(m, test_loader, test_df["Y"].to_numpy(), device)
        o.mkdir(exist_ok=True)
        np.savez(out_npz,
            val_y_true=vy, val_y_prob=vp, test_y_true=ty, test_y_prob=tp)
        print(f"  saved → {out_npz.relative_to(REPO)}")


if __name__ == "__main__":
    main()
