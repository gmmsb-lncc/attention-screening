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


@torch.no_grad()
def predict(model, drug_feat, prot_feat, smiles_list, seq_list, device, batch_size=256):
    probs = []
    for i in range(0, len(smiles_list), batch_size):
        smi_b = smiles_list[i:i+batch_size]
        seq_b = seq_list[i:i+batch_size]
        d_emb = torch.stack([torch.tensor(drug_feat(s), dtype=torch.float32) for s in smi_b]).to(device)
        p_emb = torch.stack([torch.tensor(prot_feat(s), dtype=torch.float32) for s in seq_b]).to(device)
        sim = model(d_emb, p_emb)  # cosine similarity scalar per pair
        probs.append(sim.cpu().numpy())
    return np.concatenate(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=["non_human","human","all"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42,123,456,789,1024],
                    help="Canonical seeds (replicate IDs) — deve bater com DrugBAN/GraphBAN")
    ap.add_argument("--checkpoint-root", default="ConPLex/best_models")
    ap.add_argument("--output-dir", default="ConPLex/results_universal")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    drug_feat = MorganFeaturizer(save_dir=Path.cwd())
    prot_feat = ProtBertFeaturizer(save_dir=Path.cwd())

    val_tsv, test_tsv = UNIVERSAL_TSV[args.corpus]
    val_smi, val_seq, val_y = tsv_to_triples(REPO / val_tsv)
    test_smi, test_seq, test_y = tsv_to_triples(REPO / test_tsv)
    print(f"val n={len(val_y)}  test n={len(test_y)}")

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
        ckpts = list(ckpt_dir.glob("*.pt"))
        if not ckpts: print(f"[skip] no .pt in {ckpt_dir}"); continue
        ckpt = ckpts[0]
        print(f"[seed {seed}] {ckpt.relative_to(REPO)}")
        model = SimpleCoembedding(drug_feat.shape, prot_feat.shape, latent_dim=1024).to(device)
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state, strict=False)
        model.eval()
        vp = predict(model, drug_feat, prot_feat, val_smi, val_seq, device, args.batch_size)
        tp = predict(model, drug_feat, prot_feat, test_smi, test_seq, device, args.batch_size)
        o = out_root / f"seed_{seed}"; o.mkdir(exist_ok=True)
        np.savez(o / "raw_predictions.npz",
            val_y_true=val_y, val_y_prob=vp, test_y_true=test_y, test_y_prob=tp)
        print(f"  saved → {o/'raw_predictions.npz'}")


if __name__ == "__main__":
    main()
