"""
Inferência DrugBAN sobre universal_val.tsv + universal_test.tsv.

Carrega checkpoint existente, processa universal split (val + test),
produz raw_predictions.npz alinhado a (n_val=69.318, n_test=41.441).

Uso:
    cd /Users/sulfierry/semantic-screening
    source env/bin/activate
    python infer_drugban_universal.py \\
        --corpus all \\
        --seeds 42 123 456 789 1024 \\
        --output-dir DrugBAN/results_universal
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).parent.resolve()
DRUGBAN_SRC = REPO_ROOT / "DrugBAN" / "src"
if not (DRUGBAN_SRC / "dataloader.py").exists():
    sys.stderr.write(
        f"[fatal] DrugBAN upstream not installed at {DRUGBAN_SRC}.\n"
        f"        DrugBAN/src/ is gitignored (upstream clone).\n"
        f"        Install via: bash DrugBAN/setup_env.sh\n"
        f"        Or clone upstream:\n"
        f"          git clone https://github.com/peizhenbai/DrugBAN.git {DRUGBAN_SRC}\n"
    )
    sys.exit(2)
sys.path.insert(0, str(DRUGBAN_SRC))

from dataloader import DTIDataset  # type: ignore
from models import DrugBAN  # type: ignore
from torch.utils.data import DataLoader

# graph_collate_func não exportado em DrugBAN/src/dataloader.py (padrão upstream).
# Define local baseado no padrão DTIDataset.__getitem__ → (dgl_graph, np_array, label).
try:
    from dataloader import graph_collate_func  # type: ignore
except ImportError:
    import dgl  # fornecido pelo env 'drugban'

    def graph_collate_func(batch):  # type: ignore[override]
        drugs, proteins, labels = zip(*batch)
        drugs_batch = dgl.batch(drugs)
        proteins_batch = torch.as_tensor(np.array(proteins))
        labels_batch = torch.as_tensor(np.array(labels))
        return drugs_batch, proteins_batch, labels_batch


UNIVERSAL_TSV = {
    "non_human": (
        "scaffolds_splits/output/non_human_val.tsv",
        "scaffolds_splits/output/non_human_test.tsv",
    ),
    "human": (
        "scaffolds_splits/output/human_val.tsv",
        "scaffolds_splits/output/human_test.tsv",
    ),
    "all": (
        "scaffolds_splits/output/universal_val.tsv",
        "scaffolds_splits/output/universal_test.tsv",
    ),
}


def tsv_to_drugban_df(tsv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t")
    out = pd.DataFrame(
        {
            "SMILES": df["canonical_smiles"].astype(str),
            "Protein": df["seq"].astype(str),
            "Y": df["label"].astype(int),
        }
    )
    return out


def load_model(checkpoint_path: Path, config: dict, device: torch.device) -> DrugBAN:
    # DrugBAN upstream: aceita yaml completo como kwargs (config["DRUG"], ["PROTEIN"], ["BCN"], ["DECODER"])
    model = DrugBAN(**config).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


@torch.no_grad()
def predict(model: DrugBAN, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    # fp32 inference (no autocast). AMP fp16 saturates DrugBAN on cross-
    # dataset shifts (e.g. human→non_human) → logits go NaN → all 30
    # off-diagonal cells return NaN predictions. Pure fp32 is correct
    # given weights were trained fp32; throughput cost is negligible
    # for inference-only pass.
    ys, ps = [], []
    for batch in loader:
        v_d, v_p, y = batch
        v_d = v_d.to(device, non_blocking=True)
        v_p = v_p.to(device, non_blocking=True)
        _, _, _, f = model(v_d, v_p)
        # Final softmax in float64 for numerical safety on extreme logits
        # (has no effect when logits are well-behaved).
        f = f.float()
        prob = torch.softmax(f, dim=1)[:, 1]
        # Defensive: if model still emits NaN (e.g. graph build failure
        # for malformed SMILES), clamp to 0.5 here instead of letting it
        # propagate to aggregate.py.
        if not torch.isfinite(prob).all():
            n_bad = int((~torch.isfinite(prob)).sum())
            print(f"    [warn] {n_bad}/{prob.size(0)} non-finite in batch → clamp to 0.5")
            prob = torch.nan_to_num(prob, nan=0.5, posinf=1.0, neginf=0.0)
        ys.append(y.numpy())
        ps.append(prob.cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def best_checkpoint(seed_dir: Path) -> Path | None:
    """Return first best_model_epoch_*.pth or None if absent.

    Cross-dataset matrix runs across all (corpus × seed) combos but checkpoints
    may be incomplete on some hosts (e.g. diamante-02 may only have NH).
    Caller must handle None → skip seed.
    """
    candidates = sorted(seed_dir.glob("best_model_epoch_*.pth"))
    return candidates[0] if candidates else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=["non_human", "human", "all"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1024])
    ap.add_argument("--checkpoint-root", default="DrugBAN/results")
    ap.add_argument("--config", default="DrugBAN/configs/kinase.yaml")
    ap.add_argument("--output-dir", default="DrugBAN/results_universal")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--val-tsv", default=None,
                    help="Override val TSV (used to calibrate threshold). "
                         "Defaults to the training corpus val TSV.")
    ap.add_argument("--test-tsv", default=None,
                    help="Override test TSV. Defaults to the training corpus test TSV.")
    args = ap.parse_args()

    repo = REPO_ROOT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")

    with open(repo / args.config) as f:
        config = yaml.safe_load(f)

    default_val, default_test = UNIVERSAL_TSV[args.corpus]
    val_tsv = args.val_tsv if args.val_tsv is not None else str(repo / default_val)
    test_tsv = args.test_tsv if args.test_tsv is not None else str(repo / default_test)
    print(f"Loading universal val: {val_tsv}")
    val_df = tsv_to_drugban_df(Path(val_tsv))
    print(f"  rows: {len(val_df)}  pos_rate: {val_df['Y'].mean():.3f}")
    print(f"Loading universal test: {test_tsv}")
    test_df = tsv_to_drugban_df(Path(test_tsv))
    print(f"  rows: {len(test_df)}  pos_rate: {test_df['Y'].mean():.3f}")

    val_ds = DTIDataset(val_df.index.values, val_df)
    test_ds = DTIDataset(test_df.index.values, test_df)

    use_pin = device.type == "cuda"
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=graph_collate_func,
        pin_memory=use_pin, persistent_workers=(args.num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=graph_collate_func,
        pin_memory=use_pin, persistent_workers=(args.num_workers > 0),
    )

    # Cross-dataset mode (--test-tsv override): flat layout output_dir/seed_{s}/.
    # Legacy diagonal mode keeps the output_dir/{corpus}/seed_{s}/ nesting.
    if args.test_tsv is not None:
        out_root = repo / args.output_dir
    else:
        out_root = repo / args.output_dir / args.corpus
    out_root.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        seed_dir = repo / args.checkpoint_root / args.corpus / f"seed_{seed}"
        if not seed_dir.exists():
            print(f"[skip] no dir {seed_dir}")
            continue
        out_seed = out_root / f"seed_{seed}"
        out_npz = out_seed / "raw_predictions.npz"
        if out_npz.exists():
            print(f"[seed {seed}] já concluído → {out_npz.relative_to(repo)} (pulando)")
            continue
        ckpt = best_checkpoint(seed_dir)
        if ckpt is None:
            print(f"[skip seed={seed}] no best_model_epoch_*.pth in {seed_dir.relative_to(repo)}")
            continue
        print(f"[seed {seed}] loading {ckpt.relative_to(repo)}")
        model = load_model(ckpt, config, device)

        print(f"  inferring val…")
        val_y, val_p = predict(model, val_loader, device)
        print(f"  inferring test…")
        test_y, test_p = predict(model, test_loader, device)

        out_seed.mkdir(exist_ok=True)
        np.savez(
            out_npz,
            val_y_true=val_y, val_y_prob=val_p,
            test_y_true=test_y, test_y_prob=test_p,
        )
        print(f"  saved → {out_npz.relative_to(repo)}")

    print("\nDone. Recalibrate via:")
    print(f"  python recalibrate_baselines_f1val.py --results-root {args.output_dir.replace('/results_universal', '')} --models DrugBAN --output RECALIB_F1VAL_DRUGBAN_UNIVERSAL.json")


if __name__ == "__main__":
    main()
