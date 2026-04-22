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

try:
    from dataloader import graph_collate_func  # type: ignore
except ImportError:
    import dgl  # fornecido pelo env 'graphban'

    def graph_collate_func(batch):  # type: ignore[override]
        drugs, proteins, labels = zip(*batch)
        drugs_batch = dgl.batch(drugs)
        proteins_batch = torch.as_tensor(np.array(proteins))
        labels_batch = torch.as_tensor(np.array(labels))
        return drugs_batch, proteins_batch, labels_batch


UNIVERSAL_TSV = {
    "non_human": ("scaffolds_splits/output/non_human_val.tsv",
                  "scaffolds_splits/output/non_human_test.tsv"),
    "human": ("scaffolds_splits/output/human_val.tsv",
              "scaffolds_splits/output/human_test.tsv"),
    "all": ("scaffolds_splits/output/universal_val.tsv",
            "scaffolds_splits/output/universal_test.tsv"),
}


def tsv_to_graphban_df(tsv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t")
    return pd.DataFrame({
        "SMILES": df["canonical_smiles"].astype(str),
        "Protein": df["seq"].astype(str),
        "Y": df["label"].astype(int),
    })


def load_model(checkpoint_path: Path, config: dict, device: torch.device):
    model = GraphBAN(**config).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


@torch.no_grad()
def predict(model, loader, device):
    ys, ps = [], []
    use_amp = device.type == "cuda"
    for batch in loader:
        v_d, v_p, y = batch
        v_d = v_d.to(device, non_blocking=True); v_p = v_p.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            out = model(v_d, v_p)
        logits = out[-1] if isinstance(out, tuple) else out
        prob = torch.softmax(logits.float(), dim=1)[:, 1]
        ys.append(y.numpy()); ps.append(prob.cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


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
    val_df = tsv_to_graphban_df(REPO / val_tsv)
    test_df = tsv_to_graphban_df(REPO / test_tsv)
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
        vy, vp = predict(m, val_loader, device)
        ty, tp = predict(m, test_loader, device)
        o.mkdir(exist_ok=True)
        np.savez(out_npz,
            val_y_true=vy, val_y_prob=vp, test_y_true=ty, test_y_prob=tp)
        print(f"  saved → {out_npz.relative_to(REPO)}")


if __name__ == "__main__":
    main()
