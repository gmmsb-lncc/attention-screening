"""DT-Kinase-LEGACY scoring adapter for the committee orchestrator.

Reads pairs.tsv (uniprot, sequence, chembl_id, smiles, source) and emits
scores_dtkinase.csv (uniprot, chembl_id, prob, pred, threshold).

Pipeline:
    1. Load v7+F LEGACY checkpoint (ADAPTER_LEGACY=1 hardcoded via env).
    2. Encode unique proteins via ESM-2 8M (cached by uniprot).
    3. Encode unique SMILES via MoLFormer (cached by sha1 hash).
    4. For each pair: forward InteractionMapCNN, sigmoid → raw probability.
    5. Apply Platt calibration (a, b) + threshold from sidecar JSON.
    6. Emit CSV.

Calibration sidecar JSON format (next to checkpoint, named
``level4_cnn_calibration.json``):
    {
      "platt_a": float, "platt_b": float, "threshold": float,
      "corpus": "human"|"non_human"|"all", "n_val": int,
      "source_run": str
    }

If sidecar missing, falls back to threshold=0.5 with a warning. Generate
the sidecar via scripts/thesis_followups/eval_checkpoint_on_dataset.py
(reads its JSON output and writes the abbreviated sidecar).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

# --- ADAPTER_LEGACY=1 must be set BEFORE importing level4_cnn so the
# EmbeddingAdapter constructor reads the LEGACY default. --------------
os.environ.setdefault("BENCHMARK_LEVEL4CNN_ADAPTER_LEGACY", "1")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from benchmark.levels.level4_cnn import InteractionMapCNN, MOLFORMER_DIM  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "scripts" / "inference"))
from encoders import (  # noqa: E402
    load_esm2_8m, encode_proteins,
    load_molformer, encode_ligands,
    ESM2_DIM,
)


CANONICAL_CONFIG = REPO_ROOT / "configs" / "v7.yaml"
# Canonical vanilla v7 ckpts per corpus (PR #206 Platt-on-val protocol,
# matches thesis Capítulo 5 main baselines table). All five canonical seeds
# are versioned in the repository; the committee aggregates over them.
CANONICAL_SEEDS = (42, 123, 456, 789, 1024)
CANONICAL_CKPT_DIR_BY_CORPUS = {
    "human":     REPO_ROOT / "results" / "benchmark_human_8M_13_05_2026"
                 / "test" / "level4_cnn_8M" / "human",
    "non_human": REPO_ROOT / "results" / "benchmark_non_human_8M_13_05_2026_v3"
                 / "test" / "level4_cnn_8M" / "non_human",
    "all":       REPO_ROOT / "results" / "all" / "benchmark_all_8M_13_04_2026"
                 / "test" / "level4_cnn_8M" / "all",
}


def resolve_ckpt(corpus: str, seed: int) -> Path:
    """Return the canonical ckpt path for a (corpus, seed) cell."""
    return CANONICAL_CKPT_DIR_BY_CORPUS[corpus] / f"seed_{seed}" / "level4_cnn_model.pt"


# Backwards-compat: keep the legacy single-seed map (used by older callers
# that index by corpus directly).
CANONICAL_CKPT_BY_CORPUS = {
    c: resolve_ckpt(c, 42) for c in CANONICAL_CKPT_DIR_BY_CORPUS
}


# ======================================================================
# Model loading
# ======================================================================

def build_model(config_path: Path, device: torch.device) -> InteractionMapCNN:
    """Rebuild InteractionMapCNN matching v7+F config (LEGACY adapter)."""
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    lc = cfg["level4_cnn"]
    adapter = lc.get("adapter", {})
    contrastive_weight = float(lc.get("contrastive_weight", 0.0))
    contrastive_dim = int(lc.get("contrastive_dim", 128)) if contrastive_weight > 0 else 0

    model = InteractionMapCNN(
        protein_dim=ESM2_DIM,
        ligand_dim=MOLFORMER_DIM,
        num_heads=int(lc.get("num_heads", 16)),
        head_dim=int(lc.get("head_dim", 64)),
        cnn_channels=int(lc.get("channels", 64)),
        dropout=float(lc.get("dropout", 0.35)),
        variant=lc.get("variant", "v7"),
        mlp_head=bool(lc.get("mlp_head", True)),
        cosine_sim=bool(lc.get("cosine_sim", False)),
        use_adapter=bool(adapter.get("enabled", True)),
        adapter_bottleneck_prot=int(adapter.get("prot_dim", 512)),
        adapter_bottleneck_lig=int(adapter.get("lig_dim", 1024)),
        adapter_layers=int(adapter.get("layers", 1)),
        adapter_self_attn=bool(adapter.get("self_attn", True)),
        contrastive_dim=contrastive_dim,
        cosine_feat=bool(lc.get("cosine_feat", True)),
        pool_num_heads=int(lc.get("pool_num_heads", 1)),
    )
    return model.to(device).eval()


def load_checkpoint(model: InteractionMapCNN, ckpt_path: Path,
                    device: torch.device) -> None:
    """Load bare state_dict, applying backward-compatibility renames.

    Backward-compat:
      - Strips torch.compile `_orig_mod.` prefix (compiled-model checkpoint).
      - Renames `pool.{lig,prot}_pool.query` → `queries` for ckpts saved
        before the multi-head pool refactor (single-head shape is identical;
        pool_num_heads=1 in the canonical config).
    """
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    cleaned = {}
    for k, v in state.items():
        k = k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k
        if k.endswith(".query"):
            k = k[:-len(".query")] + ".queries"
        cleaned[k] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=True)


def load_calibration(ckpt_path: Path) -> dict:
    """Load Platt + threshold sidecar JSON; default to identity if missing."""
    sidecar = ckpt_path.parent / "level4_cnn_calibration.json"
    if sidecar.exists():
        with open(sidecar) as fh:
            data = json.load(fh)
        return {
            "platt_a":   float(data.get("platt_a", 1.0)),
            "platt_b":   float(data.get("platt_b", 0.0)),
            "threshold": float(data.get("threshold", 0.5)),
            "source":    str(sidecar),
        }
    print(
        f"  warn: no calibration sidecar at {sidecar}; using identity Platt + thr=0.5\n"
        f"        generate sidecar via scripts/inference/build_calibration.py",
        file=sys.stderr,
    )
    return {"platt_a": 1.0, "platt_b": 0.0, "threshold": 0.5, "source": None}


def apply_platt(logits: np.ndarray, a: float, b: float) -> np.ndarray:
    """Platt scaling: probability = sigmoid(a * logit + b)."""
    z = a * logits + b
    return 1.0 / (1.0 + np.exp(-z))


# ======================================================================
# Forward per pair
# ======================================================================

@torch.inference_mode()
def score_pair(
    model: InteractionMapCNN,
    prot_mat: np.ndarray, lig_mat: np.ndarray,
    device: torch.device,
) -> float:
    """Run a single forward; return raw logit (pre-Platt)."""
    p = torch.from_numpy(prot_mat).unsqueeze(0).to(device)         # [1, sp, 320]
    l = torch.from_numpy(lig_mat).unsqueeze(0).to(device)          # [1, sl, 768]
    pm = torch.ones(1, p.shape[1], dtype=torch.bool, device=device)
    lm = torch.ones(1, l.shape[1], dtype=torch.bool, device=device)
    return float(model(p, l, pm, lm).squeeze().item())


# ======================================================================
# CLI
# ======================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score pairs.tsv with DT-Kinase-LEGACY (v7+F)."
    )
    ap.add_argument("--pairs",   type=Path, required=True,
                    help="input pairs.tsv (uniprot, sequence, chembl_id, smiles, source)")
    ap.add_argument("--out",     type=Path, required=True,
                    help="output scores_dtkinase.csv")
    ap.add_argument("--corpus",  choices=["human", "non_human", "all"], default="all",
                    help="which checkpoint corpus to load (default: all → broadest)")
    ap.add_argument("--ckpt",    type=Path, default=None,
                    help="explicit checkpoint path (overrides --corpus auto-resolve, "
                         "single-seed mode; ignores --seeds)")
    ap.add_argument("--seeds",   type=str, default="42,123,456,789,1024",
                    help="comma-separated seeds for the 5-seed ensemble "
                         "(default: all five canonical seeds; pass --seeds 42 for "
                         "legacy single-seed behaviour)")
    ap.add_argument("--config",  type=Path, default=CANONICAL_CONFIG)
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="root for embedding cache (default: data/reference/cache)")
    ap.add_argument("--batch-size-lig", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve the ensemble of (ckpt, calibration) pairs to score.
    if args.ckpt is not None:
        ckpts = [args.ckpt]
    else:
        seed_list = [int(s) for s in args.seeds.split(",") if s.strip()]
        ckpts = [resolve_ckpt(args.corpus, s) for s in seed_list]
        missing_ckpts = [str(c) for c in ckpts if not c.exists()]
        if missing_ckpts:
            sys.exit(
                "checkpoint(s) not found:\n  " + "\n  ".join(missing_ckpts) +
                f"\nverify the seeds {seed_list} are populated for corpus={args.corpus}, "
                "or pass --ckpt for single-seed mode."
            )

    pairs = pd.read_csv(args.pairs, sep="\t")
    required = {"uniprot", "sequence", "chembl_id", "smiles"}
    missing = required - set(pairs.columns)
    if missing:
        sys.exit(f"pairs.tsv missing columns: {missing}")
    print(f"  loaded {len(pairs)} pairs from {args.pairs}", file=sys.stderr)
    print(f"  ensemble: {len(ckpts)} seed(s) (corpus={args.corpus})",
          file=sys.stderr)

    # Encode unique proteins + ligands once
    unique_prot = (
        pairs[["uniprot", "sequence"]].drop_duplicates(subset=["uniprot"])
        .itertuples(index=False, name=None)
    )
    unique_lig = (
        pairs[["chembl_id", "smiles"]].drop_duplicates(subset=["chembl_id"])
        .itertuples(index=False, name=None)
    )

    print("  loading ESM-2 8M...", file=sys.stderr)
    esm_model, esm_alpha = load_esm2_8m(device)
    prot_mats = encode_proteins(esm_model, esm_alpha, unique_prot, device)
    del esm_model, esm_alpha
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("  loading MoLFormer...", file=sys.stderr)
    mol_model, mol_tok = load_molformer(device)
    lig_mats = encode_ligands(mol_model, mol_tok, unique_lig, device,
                              batch_size=args.batch_size_lig)
    del mol_model, mol_tok
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Score every pair across the ensemble of seeds.
    # Each seed contributes a calibrated probability vector; the ensemble
    # output is the mean over seeds (probability) and the mean of the
    # per-seed thresholds (operational threshold).
    n = len(pairs)
    probs_per_seed = np.empty((len(ckpts), n), dtype=np.float64)
    thresholds_per_seed = np.empty(len(ckpts), dtype=np.float64)

    model = build_model(args.config, device)
    for k, ckpt in enumerate(ckpts):
        load_checkpoint(model, ckpt, device)
        calib = load_calibration(ckpt)
        print(
            f"  [seed {k+1}/{len(ckpts)}] ckpt={ckpt.parent.name} "
            f"a={calib['platt_a']:.4f} b={calib['platt_b']:.4f} thr={calib['threshold']:.3f}",
            file=sys.stderr,
        )
        logits = np.empty(n, dtype=np.float64)
        for i, row in enumerate(pairs.itertuples(index=False)):
            try:
                pmat = prot_mats[row.uniprot]
                lmat = lig_mats[row.chembl_id]
            except KeyError as e:
                sys.exit(f"missing embedding for {e!r} (row {i})")
            logits[i] = score_pair(model, pmat, lmat, device)
        probs_per_seed[k] = apply_platt(logits, calib["platt_a"], calib["platt_b"])
        thresholds_per_seed[k] = calib["threshold"]

    # Ensemble aggregation: mean probability across seeds, mean threshold.
    probs = probs_per_seed.mean(axis=0)
    threshold = float(thresholds_per_seed.mean())
    preds = (probs >= threshold).astype(int)

    out_df = pd.DataFrame({
        "uniprot":   pairs["uniprot"].to_numpy(),
        "chembl_id": pairs["chembl_id"].to_numpy(),
        "prob":      probs,
        "pred":      preds,
        "threshold": threshold,
        "n_seeds":   len(ckpts),
        "prob_std":  probs_per_seed.std(axis=0, ddof=0),
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(
        f"  wrote {len(out_df)} rows → {args.out} "
        f"(ensemble of {len(ckpts)} seeds, mean threshold={threshold:.3f})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
