"""Explain DT-Kinase v7+F predictions via attention/HierPool weights.

For a (protein, ligand) pair (or a query + library scoring), extracts:
  - per-head interaction maps M_k ∈ R^{sp × sl}
  - mean-over-heads aggregated map M ∈ R^{sp × sl}
  - HierPool attention weights (lig axis, then prot axis)
  - per-residue importance score
  - per-ligand-token importance score

Two modes:
  pair   — single (protein, ligand). Returns full interpretation.
  topk   — query (protein OR ligand) scored against a library. Returns
           top-K candidates each with attention extraction.

Currently uses pre-cached embeddings (.npy files under
results/protein_model_benchmark_<corpus>_v2/<embed>/build/) for speed.
On-the-fly encoding from raw FASTA/SMILES is a future addition.

Usage:
  # Single pair from cached IDs (fast)
  python3 scripts/inference/explain.py pair \
      --checkpoint <path/to/level4_cnn_model.pt> \
      --config configs/v7_plus_F.yaml \
      --seq-id <SEQ_ID> --chembl-id <CHEMBL_ID> \
      --corpus non_human \
      --output explanation.json

  # Top-K ligands for a protein query
  python3 scripts/inference/explain.py topk \
      --checkpoint ... --config configs/v7_plus_F.yaml \
      --seq-id <SEQ_ID> \
      --library scaffolds_splits/output/non_human_test.tsv \
      --top-k 10 \
      --corpus non_human \
      --output explanations_top10.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

# --- Repo imports (same conventions as run_from_config.py) -----------------
import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from benchmark.levels.level4_cnn import InteractionMapCNN, MOLFORMER_DIM


# ===========================================================================
# Model loading
# ===========================================================================

def load_model_from_checkpoint(
    checkpoint_path: Path,
    config_path: Path,
    device: torch.device,
) -> InteractionMapCNN:
    """Rebuild InteractionMapCNN matching config and load state_dict."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    lc = cfg["level4_cnn"]
    adapter_cfg = lc.get("adapter", {})

    model = InteractionMapCNN(
        protein_dim=320,  # ESM-2 8M
        ligand_dim=MOLFORMER_DIM,
        num_heads=lc.get("num_heads", 8),
        head_dim=lc.get("head_dim", 32),
        cnn_channels=lc.get("channels", 64),
        dropout=lc.get("dropout", 0.3),
        variant=lc.get("variant", "v7"),
        mlp_head=lc.get("mlp_head", False),
        cosine_sim=lc.get("cosine_sim", False),
        use_adapter=adapter_cfg.get("enabled", False),
        adapter_bottleneck_prot=adapter_cfg.get("prot_dim", 256),
        adapter_bottleneck_lig=adapter_cfg.get("lig_dim", 512),
        adapter_layers=adapter_cfg.get("layers", 1),
        adapter_self_attn=adapter_cfg.get("self_attn", False),
        contrastive_dim=lc.get("contrastive_dim", 128) if lc.get("contrastive_weight", 0) > 0 else 0,
        cosine_feat=lc.get("cosine_feat", False),
        pool_num_heads=lc.get("pool_num_heads", 1),
        use_ban_residual=os.getenv("BENCHMARK_LEVEL4CNN_BAN_RESIDUAL", "0") == "1",
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[explain] WARN: missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[explain] WARN: unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    model.to(device).eval()
    return model


# ===========================================================================
# Embedding cache lookup
# ===========================================================================

def find_matrix(cache_dirs: list[Path], identifier: str, modality: str) -> np.ndarray:
    """Search cache directories for {identifier}_matrix.npy or _molformer_matrix.npy."""
    for d in cache_dirs:
        p = d / f"{identifier}_matrix.npy"
        if p.exists():
            return np.load(p).astype(np.float32)
        p_alt = d / f"{identifier}_molformer_matrix.npy"
        if p_alt.exists():
            return np.load(p_alt).astype(np.float32)
    raise FileNotFoundError(
        f"{modality} matrix not found for '{identifier}' in: {cache_dirs}")


def cache_dirs_for(corpus: str, embedding: str = "8M") -> tuple[list[Path], list[Path]]:
    """Resolve cache dirs for a corpus (mirrors matrix_utils._resolve_matrix_dirs)."""
    base = REPO_ROOT / "results" / f"protein_model_benchmark_{corpus}_v2" / embedding / "build"
    return (
        [base / "protein_matrices"],
        [base / "ligand_matrices", base / "molformer_matrix"],
    )


# ===========================================================================
# Hooks for attention extraction
# ===========================================================================

class ExplainHooks:
    """Captures interaction maps + HierPool attention via forward hooks."""

    def __init__(self, model: InteractionMapCNN):
        self.model = model
        self.interaction_maps: torch.Tensor | None = None  # [B, K, sp, sl]
        self.lig_pool_attn: torch.Tensor | None = None     # [B, H_pool, sp, sl_valid]
        self.prot_pool_attn: torch.Tensor | None = None    # [B, H_pool, sp_valid]
        self._handles: list = []

    def __enter__(self):
        # Hook on the v7 forward path: capture interaction tensor right
        # after stack but before CNN. Since v7 builds it inline, we monkey-
        # patch the model's forward to store self._captured_interaction.
        # Cleaner approach: use a hook on the CNN's first layer's input.
        cnn_first = self.model.cnn[0] if hasattr(self.model, "cnn") else None
        if cnn_first is not None:
            def cap_cnn(_mod, inputs, _out):
                # input is the stacked interaction tensor [B, K, sp, sl]
                self.interaction_maps = inputs[0].detach().cpu()
            self._handles.append(cnn_first.register_forward_hook(cap_cnn))

        # HierPool: hook on each _AxisAttentionPool to capture attn weights.
        # Attribute is `self.pool` in InteractionMapCNN.
        if hasattr(self.model, "pool"):
            hp = self.model.pool
            def cap_lig(_mod, inputs, _out):
                # Reproduce the attention computation to extract weights.
                x = inputs[0]                          # [B, sp, C]  (per-prot vectors)
                pad_mask = inputs[1] if len(inputs) > 1 else None
                self.lig_pool_attn = self._compute_axis_attn(_mod, x, pad_mask)
            def cap_prot(_mod, inputs, _out):
                x = inputs[0]
                pad_mask = inputs[1] if len(inputs) > 1 else None
                self.prot_pool_attn = self._compute_axis_attn(_mod, x, pad_mask)
            self._handles.append(hp.lig_pool.register_forward_hook(cap_lig))
            self._handles.append(hp.prot_pool.register_forward_hook(cap_prot))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @staticmethod
    @torch.inference_mode()
    def _compute_axis_attn(pool_mod, x: torch.Tensor, pad_mask):
        """Replay _AxisAttentionPool internals to expose softmax weights."""
        B = x.size(0)
        q = pool_mod.queries.expand(B, -1, -1)
        scores = torch.bmm(q, x.transpose(1, 2)) * pool_mod.scale
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask.unsqueeze(1), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        return attn.detach().cpu()


# ===========================================================================
# Forward + interpretation
# ===========================================================================

@torch.inference_mode()
def explain_pair(
    model: InteractionMapCNN,
    prot_mat: np.ndarray,
    lig_mat: np.ndarray,
    device: torch.device,
) -> dict:
    """Run forward and extract attention weights for a single pair."""
    p = torch.from_numpy(prot_mat).unsqueeze(0).to(device)  # [1, sp, 320]
    l = torch.from_numpy(lig_mat).unsqueeze(0).to(device)   # [1, sl, 768]
    pm = torch.ones(1, p.shape[1], dtype=torch.bool, device=device)
    lm = torch.ones(1, l.shape[1], dtype=torch.bool, device=device)

    with ExplainHooks(model) as hooks:
        logit = model(p, l, pm, lm).squeeze().item()

    prob = float(torch.sigmoid(torch.tensor(logit)))

    out = {
        "logit": logit,
        "prob": prob,
        "seq_len_protein": int(p.shape[1]),
        "seq_len_ligand": int(l.shape[1]),
    }

    if hooks.interaction_maps is not None:
        M = hooks.interaction_maps[0]  # [K, sp, sl]
        # Aggregate over heads and per-axis sums for per-residue / per-token importance.
        M_mean = M.mean(0)               # [sp, sl]
        prot_imp = M_mean.sum(-1).numpy().tolist()  # per-residue
        lig_imp = M_mean.sum(0).numpy().tolist()    # per-ligand-token
        out["interaction_maps_shape"] = list(M.shape)
        out["prot_residue_attention"] = prot_imp
        out["lig_token_attention"] = lig_imp
        out["per_head_intensity"] = M.abs().mean(dim=(-2, -1)).numpy().tolist()

    if hooks.prot_pool_attn is not None:
        # Shape: [B, num_pool_heads, sp]. For B=1, mean over heads → [sp].
        out["hierpool_prot_weights"] = (
            hooks.prot_pool_attn.mean(dim=(0, 1)).numpy().tolist()
        )
    if hooks.lig_pool_attn is not None:
        # Shape: [B*sp, num_pool_heads, sl] (lig_pool runs once per prot pos).
        # Aggregate over (B*sp) to get per-token importance averaged across
        # protein positions. mean(dim=(0,1)) → [sl].
        out["hierpool_lig_weights"] = (
            hooks.lig_pool_attn.mean(dim=(0, 1)).numpy().tolist()
        )

    return out


# ===========================================================================
# Library scoring (topk mode)
# ===========================================================================

def score_library(
    model: InteractionMapCNN,
    fixed_mat: np.ndarray,
    fixed_role: str,             # "protein" or "ligand"
    library_df: pd.DataFrame,
    cache_protein: list[Path],
    cache_ligand: list[Path],
    device: torch.device,
) -> pd.DataFrame:
    """Score every (fixed, candidate) pair. Returns DataFrame with logit/prob."""
    rows = []
    fixed_t = torch.from_numpy(fixed_mat).unsqueeze(0).to(device)
    fixed_mask = torch.ones(1, fixed_t.shape[1], dtype=torch.bool, device=device)

    for _, row in library_df.iterrows():
        try:
            if fixed_role == "protein":
                lig_mat = find_matrix(cache_ligand, row["chembl_id"], "ligand")
                p, l = fixed_t, torch.from_numpy(lig_mat).unsqueeze(0).to(device)
                pm, lm = fixed_mask, torch.ones(1, l.shape[1], dtype=torch.bool, device=device)
                cand_id = row["chembl_id"]
            else:
                prot_mat = find_matrix(cache_protein, row["seq_id"], "protein")
                p, l = torch.from_numpy(prot_mat).unsqueeze(0).to(device), fixed_t
                pm = torch.ones(1, p.shape[1], dtype=torch.bool, device=device)
                lm = fixed_mask
                cand_id = row["seq_id"]
        except FileNotFoundError:
            continue

        with torch.inference_mode():
            logit = model(p, l, pm, lm).squeeze().item()
        rows.append({
            "candidate_id": cand_id,
            "logit": logit,
            "prob": float(torch.sigmoid(torch.tensor(logit))),
        })
    return pd.DataFrame(rows).sort_values("prob", ascending=False).reset_index(drop=True)


# ===========================================================================
# CLI
# ===========================================================================

def _lookup_metadata(splits_dir: Path, corpus: str, seq_id: str | None,
                     chembl_id: str | None) -> dict:
    """Look up SMILES + protein sequence from split TSVs (best-effort)."""
    meta = {}
    for split in ("train", "val", "test"):
        for prefix in (corpus, "universal"):
            tsv = splits_dir / f"{prefix}_{split}.tsv"
            if not tsv.exists():
                continue
            df = pd.read_csv(tsv, sep="\t")
            if seq_id and "seq_id" in df.columns:
                m = df[df["seq_id"] == seq_id]
                if len(m) > 0 and "sequence" in m.columns:
                    meta["protein_sequence"] = str(m.iloc[0]["sequence"])
            if chembl_id and "chembl_id" in df.columns:
                m = df[df["chembl_id"] == chembl_id]
                if len(m) > 0 and "smiles" in m.columns:
                    meta["ligand_smiles"] = str(m.iloc[0]["smiles"])
            if (seq_id is None or "protein_sequence" in meta) and \
               (chembl_id is None or "ligand_smiles" in meta):
                return meta
    return meta


def cmd_pair(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(args.checkpoint, args.config, device)
    cache_p, cache_l = cache_dirs_for(args.corpus, args.embedding)

    prot_mat = find_matrix(cache_p, args.seq_id, "protein")
    lig_mat = find_matrix(cache_l, args.chembl_id, "ligand")
    expl = explain_pair(model, prot_mat, lig_mat, device)
    meta = _lookup_metadata(args.splits_dir, args.corpus, args.seq_id, args.chembl_id)
    expl.update({
        "seq_id": args.seq_id,
        "chembl_id": args.chembl_id,
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "corpus": args.corpus,
        **meta,
    })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(expl, f, indent=2)
    print(f"[explain.pair] {args.seq_id} × {args.chembl_id}")
    print(f"  logit={expl['logit']:.4f}, prob={expl['prob']:.4f}")
    print(f"  protein len={expl['seq_len_protein']}, ligand len={expl['seq_len_ligand']}")
    if "per_head_intensity" in expl:
        print(f"  per-head intensities: {[f'{x:.3f}' for x in expl['per_head_intensity']]}")
    print(f"  → {args.output}")


def cmd_topk(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(args.checkpoint, args.config, device)
    cache_p, cache_l = cache_dirs_for(args.corpus, args.embedding)

    library = pd.read_csv(args.library, sep="\t")
    if args.seq_id:
        # Score against ligand library — dedupe by chembl_id.
        library = library.drop_duplicates(subset=["chembl_id"])
    elif args.chembl_id:
        library = library.drop_duplicates(subset=["seq_id"])
    else:
        raise SystemExit("Provide --seq-id OR --chembl-id for topk mode.")

    if args.seq_id:
        fixed = find_matrix(cache_p, args.seq_id, "protein")
        scored = score_library(model, fixed, "protein", library, cache_p, cache_l, device)
        query_descr = {"seq_id": args.seq_id}
    elif args.chembl_id:
        fixed = find_matrix(cache_l, args.chembl_id, "ligand")
        scored = score_library(model, fixed, "ligand", library, cache_p, cache_l, device)
        query_descr = {"chembl_id": args.chembl_id}
    else:
        raise SystemExit("Provide --seq-id OR --chembl-id for topk mode.")

    top = scored.head(args.top_k)
    print(f"[explain.topk] query={query_descr}, top {args.top_k}/{len(scored)}:")
    for _, row in top.iterrows():
        print(f"  {row['candidate_id']:30s}  logit={row['logit']:+.4f}  prob={row['prob']:.4f}")

    # Explain each top-K
    explanations = []
    for _, row in top.iterrows():
        if args.seq_id:
            try:
                lig_mat = find_matrix(cache_l, row["candidate_id"], "ligand")
            except FileNotFoundError:
                continue
            expl = explain_pair(model, fixed, lig_mat, device)
            expl["candidate_id"] = row["candidate_id"]
        else:
            try:
                prot_mat = find_matrix(cache_p, row["candidate_id"], "protein")
            except FileNotFoundError:
                continue
            expl = explain_pair(model, prot_mat, fixed, device)
            expl["candidate_id"] = row["candidate_id"]
        explanations.append(expl)

    output = {
        "query": query_descr,
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "corpus": args.corpus,
        "top_k": args.top_k,
        "n_library_total": len(scored),
        "results": explanations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  → {args.output}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--checkpoint", type=Path, required=True)
    common.add_argument("--config", type=Path, default=Path("configs/v7_plus_F.yaml"))
    common.add_argument("--corpus", choices=["human", "non_human", "all"], default="non_human")
    common.add_argument("--embedding", default="8M")
    common.add_argument("--output", type=Path, required=True)
    common.add_argument("--splits-dir", type=Path,
                        default=Path("scaffolds_splits/output"),
                        help="Directory with split TSVs for SMILES/sequence lookup.")

    sp_pair = sub.add_parser("pair", parents=[common], help="Single pair explanation")
    sp_pair.add_argument("--seq-id", required=True)
    sp_pair.add_argument("--chembl-id", required=True)
    sp_pair.set_defaults(func=cmd_pair)

    sp_topk = sub.add_parser("topk", parents=[common], help="Library scoring + top-K")
    sp_topk.add_argument("--seq-id")
    sp_topk.add_argument("--chembl-id")
    sp_topk.add_argument("--library", type=Path, required=True,
                         help="TSV with seq_id/chembl_id columns")
    sp_topk.add_argument("--top-k", type=int, default=10)
    sp_topk.set_defaults(func=cmd_topk)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
