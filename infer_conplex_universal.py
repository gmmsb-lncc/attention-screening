"""
Inferência ConPLex sobre universal_val.tsv + universal_test.tsv.
Carrega checkpoint, emite similaridade cosseno como probabilidade,
salva raw_predictions.npz alinhado ao universal split.

Uso (diamante-02):
    python infer_conplex_universal.py --corpus all --replicates 0 1 2 3 4
"""

import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

REPO = Path(__file__).parent.resolve()
CONPLEX_ROOT = REPO / "ConPLex"
CONPLEX_SRC = CONPLEX_ROOT / "src"
sys.path.insert(0, str(CONPLEX_ROOT))  # permite 'from src.X import Y'

from src.featurizers.molecule import MorganFeaturizer  # type: ignore
from src.featurizers.protein import ProtBertFeaturizer  # type: ignore
from src.architectures import SimpleCoembedding  # type: ignore


UNIVERSAL_TSV = {
    "non_human": ("scaffolds_splits/output/non_human_val.tsv",
                  "scaffolds_splits/output/non_human_test.tsv"),
    "human": ("scaffolds_splits/output/human_val.tsv",
              "scaffolds_splits/output/human_test.tsv"),
    "all": ("scaffolds_splits/output/universal_val.tsv",
            "scaffolds_splits/output/universal_test.tsv"),
}


def tsv_to_triples(tsv_path: Path):
    df = pd.read_csv(tsv_path, sep="\t")
    return df["canonical_smiles"].tolist(), df["seq"].tolist(), df["label"].astype(int).to_numpy()


def _featurize_all(featurizer, items, label: str):
    """Pre-computa todos os embeddings uma vez (CPU bottleneck) e cacheia em tensor."""
    embs = []
    unique = {}
    for it in items:
        if it in unique:
            embs.append(unique[it])
            continue
        e = torch.as_tensor(featurizer(it), dtype=torch.float32)
        unique[it] = e
        embs.append(e)
    print(f"    {label}: {len(items)} pares, {len(unique)} únicos featurizados")
    return torch.stack(embs)


@torch.no_grad()
def predict(model, d_all, p_all, device, batch_size=1024):
    """Inferência GPU-batched sobre embeddings pré-computados."""
    use_amp = device.type == "cuda"
    probs = []
    n = d_all.shape[0]
    for i in range(0, n, batch_size):
        d_b = d_all[i:i+batch_size].to(device, non_blocking=True)
        p_b = p_all[i:i+batch_size].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            sim = model(d_b, p_b)
        probs.append(sim.float().cpu().numpy())
    return np.concatenate(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=["non_human","human","all"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42,123,456,789,1024],
                    help="Canonical seeds (replicate IDs) — deve bater com DrugBAN/GraphBAN")
    ap.add_argument("--checkpoint-root", default="ConPLex/best_models")
    ap.add_argument("--output-dir", default="ConPLex/results_universal")
    ap.add_argument("--batch-size", type=int, default=1024)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")
    drug_feat = MorganFeaturizer(save_dir=Path.cwd())
    prot_feat = ProtBertFeaturizer(save_dir=Path.cwd())
    # Tentar mover ProtBert para GPU se featurizer suportar
    if hasattr(prot_feat, "to") and device.type == "cuda":
        try:
            prot_feat.to(device)
            print(f"  ProtBertFeaturizer movido para {device}")
        except Exception as e:
            print(f"  ProtBertFeaturizer permanece em CPU: {e}")

    val_tsv, test_tsv = UNIVERSAL_TSV[args.corpus]
    val_smi, val_seq, val_y = tsv_to_triples(REPO / val_tsv)
    test_smi, test_seq, test_y = tsv_to_triples(REPO / test_tsv)
    print(f"val n={len(val_y)}  test n={len(test_y)}")
    print("  pré-featurizando val/test (uma vez por corpus)…")
    val_drug_emb = _featurize_all(drug_feat, val_smi, "val drug")
    val_prot_emb = _featurize_all(prot_feat, val_seq, "val prot")
    test_drug_emb = _featurize_all(drug_feat, test_smi, "test drug")
    test_prot_emb = _featurize_all(prot_feat, test_seq, "test prot")

    out_root = REPO / args.output_dir / args.corpus
    out_root.mkdir(parents=True, exist_ok=True)

    # Mapear seed canônica para índice de réplica (0..len(seeds)-1)
    # checkpoints em diamante-02 seguem padrão trained_<corpus>_rep<idx>
    seed_to_rep = {s: i for i, s in enumerate(sorted(set(args.seeds)))}
    for seed in args.seeds:
        rep_idx = seed_to_rep[seed]
        candidates = [
            f"trained_{args.corpus}_rep{rep_idx}",
            f"trained_{args.corpus}_rep{seed}",
            f"rep_{seed}", f"seed_{seed}",
        ]
        ckpt_dir = None
        for cand in candidates:
            p = REPO / args.checkpoint_root / cand
            if p.exists():
                ckpt_dir = p
                break
        if ckpt_dir is None:
            print(f"[skip] no dir for seed={seed} (searched: {candidates})"); continue
        o = out_root / f"seed_{seed}"
        out_npz = o / "raw_predictions.npz"
        if out_npz.exists():
            print(f"[seed {seed}] já concluído → {out_npz.relative_to(REPO)} (pulando)")
            continue
        ckpts = list(ckpt_dir.glob("*.pt"))
        if not ckpts: print(f"[skip] no .pt in {ckpt_dir}"); continue
        ckpt = ckpts[0]
        print(f"[seed {seed}] {ckpt.relative_to(REPO)}")
        model = SimpleCoembedding(drug_feat.shape, prot_feat.shape, latent_dimension=1024).to(device)
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state, strict=False)
        model.eval()
        vp = predict(model, val_drug_emb, val_prot_emb, device, args.batch_size)
        tp = predict(model, test_drug_emb, test_prot_emb, device, args.batch_size)
        o.mkdir(exist_ok=True)
        np.savez(out_npz,
            val_y_true=val_y, val_y_prob=vp, test_y_true=test_y, test_y_prob=tp)
        print(f"  saved → {out_npz.relative_to(REPO)}")


if __name__ == "__main__":
    main()
