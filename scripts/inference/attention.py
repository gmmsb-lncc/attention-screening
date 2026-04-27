"""Extract attention maps for STRONG/LIKELY committee hits.

Reads consensus.csv + pairs.tsv, selects the top-K rows in the requested
tiers, runs DT-Kinase-LEGACY forward with hooks, and saves per-pair NPZ
attention bundles plus heatmap PDFs.

Three attention sources from DT-Kinase:
    1. M_k pre-CNN raw           — shape [K=16, sp, sl]
    2. HierPool stage-1 lig-axis — shape [sp, sl_attn]  (per-prot-pos lig weights)
    3. HierPool stage-2 prot-axis — shape [sp_attn]      (overall prot importance)

DrugBAN / GraphBAN BAN attention extraction is delegated to per-baseline
adapters (TODO: scripts/inference/models/{drugban,graphban}_attention.py).
ConPLex has no native attention.

Usage:
    python scripts/inference/attention.py \\
        --consensus results/inference/run_001/consensus.csv \\
        --pairs results/inference/run_001/pairs.tsv \\
        --out-dir results/inference/run_001/attention \\
        --top-k 20 --tier STRONG,LIKELY \\
        --corpus all
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Ensure LEGACY adapter
os.environ.setdefault("BENCHMARK_LEVEL4CNN_ADAPTER_LEGACY", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "inference"))

from benchmark.levels.level4_cnn import InteractionMapCNN  # noqa: E402

from encoders import load_esm2_8m, encode_proteins, load_molformer, encode_ligands  # noqa: E402
from models.dtkinase_score import build_model, load_checkpoint, CANONICAL_CONFIG, CANONICAL_CKPT_BY_CORPUS  # noqa: E402


# ======================================================================
# Hooks (re-uses the design from scripts/inference/explain.py)
# ======================================================================

class AttentionHooks:
    """Capture interaction maps + HierPool attention via forward hooks."""

    def __init__(self, model: InteractionMapCNN):
        self.model = model
        self.interaction_maps: torch.Tensor | None = None
        self.lig_pool_attn: torch.Tensor | None = None
        self.prot_pool_attn: torch.Tensor | None = None
        self._handles: list = []

    def __enter__(self):
        if hasattr(self.model, "cnn"):
            def cap_cnn(_mod, inputs, _out):
                self.interaction_maps = inputs[0].detach().cpu()
            self._handles.append(self.model.cnn[0].register_forward_hook(cap_cnn))

        if hasattr(self.model, "pool"):
            hp = self.model.pool

            def cap_lig(_mod, inputs, _out):
                x = inputs[0]
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
        B = x.size(0)
        q = pool_mod.queries.expand(B, -1, -1)
        scores = torch.bmm(q, x.transpose(1, 2)) * pool_mod.scale
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask.unsqueeze(1), float("-inf"))
        return torch.softmax(scores, dim=-1).detach().cpu()


@torch.inference_mode()
def extract_pair(
    model: InteractionMapCNN, prot_mat: np.ndarray, lig_mat: np.ndarray,
    device: torch.device,
) -> dict:
    """Run forward + capture attention. Returns dict of np arrays."""
    p = torch.from_numpy(prot_mat).unsqueeze(0).to(device)
    l = torch.from_numpy(lig_mat).unsqueeze(0).to(device)
    pm = torch.ones(1, p.shape[1], dtype=torch.bool, device=device)
    lm = torch.ones(1, l.shape[1], dtype=torch.bool, device=device)

    with AttentionHooks(model) as hooks:
        logit = float(model(p, l, pm, lm).squeeze().item())

    out: dict[str, np.ndarray | float] = {"logit": logit}
    if hooks.interaction_maps is not None:
        Mk = hooks.interaction_maps[0].numpy()           # [K, sp, sl]
        out["Mk_raw"]      = Mk
        out["Mk_mean"]     = Mk.mean(0)                  # [sp, sl]
        out["per_head"]    = np.abs(Mk).mean(axis=(-2, -1))  # [K]
        out["prot_imp"]    = out["Mk_mean"].sum(-1)      # [sp]
        out["lig_imp"]     = out["Mk_mean"].sum(0)       # [sl]

    if hooks.prot_pool_attn is not None:
        # [B=1, H_pool, sp] → average over pool heads → [sp]
        out["hierpool_prot"] = hooks.prot_pool_attn.mean(dim=(0, 1)).numpy()
    if hooks.lig_pool_attn is not None:
        # [B*sp, H_pool, sl] → average over (B*sp, H_pool) → [sl]
        out["hierpool_lig"] = hooks.lig_pool_attn.mean(dim=(0, 1)).numpy()
    return out


# ======================================================================
# Plot
# ======================================================================

def plot_consensus_heatmap(att: dict, pair_id: str, out_path: Path) -> None:
    """Render a 2x2 PDF: M_k mean heatmap + per-residue + per-token + per-head."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  matplotlib unavailable; skipping plot for {pair_id}",
              file=sys.stderr)
        return

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(f"DT-Kinase attention: {pair_id} (logit={att['logit']:+.2f})")

    if "Mk_mean" in att:
        ax = axes[0, 0]
        im = ax.imshow(att["Mk_mean"], aspect="auto", cmap="viridis")
        ax.set_title("M̄ = mean over 16 heads")
        ax.set_xlabel("ligand token")
        ax.set_ylabel("protein residue")
        plt.colorbar(im, ax=ax)

    if "hierpool_prot" in att:
        ax = axes[0, 1]
        ax.bar(range(len(att["hierpool_prot"])), att["hierpool_prot"])
        ax.set_title("HierPool protein weights (stage 2)")
        ax.set_xlabel("residue index")
        ax.set_ylabel("attention weight")

    if "hierpool_lig" in att:
        ax = axes[1, 0]
        ax.bar(range(len(att["hierpool_lig"])), att["hierpool_lig"])
        ax.set_title("HierPool ligand weights (stage 1, mean over prot pos)")
        ax.set_xlabel("ligand token")
        ax.set_ylabel("attention weight")

    if "per_head" in att:
        ax = axes[1, 1]
        ax.bar(range(len(att["per_head"])), att["per_head"])
        ax.set_title("Per-head intensity (|M_k|.mean)")
        ax.set_xlabel("head k")
        ax.set_ylabel("|M_k|")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


# ======================================================================
# CLI
# ======================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Extract DT-Kinase attention for top consensus hits.")
    ap.add_argument("--consensus", type=Path, required=True)
    ap.add_argument("--pairs",     type=Path, required=True)
    ap.add_argument("--out-dir",   type=Path, required=True)
    ap.add_argument("--top-k",     type=int, default=20)
    ap.add_argument("--tier",      type=str, default="STRONG,LIKELY",
                    help="comma-separated subset of {STRONG,LIKELY,UNCERTAIN,UNLIKELY}")
    ap.add_argument("--corpus",    choices=["human", "non_human", "all"], default="all")
    ap.add_argument("--ckpt",      type=Path, default=None)
    ap.add_argument("--config",    type=Path, default=CANONICAL_CONFIG)
    ap.add_argument("--no-plot",   action="store_true")
    args = ap.parse_args()

    consensus = pd.read_csv(args.consensus)
    pairs     = pd.read_csv(args.pairs, sep="\t")

    tiers_keep = {t.strip().upper() for t in args.tier.split(",") if t.strip()}
    sel = consensus[consensus["tier"].isin(tiers_keep)].head(args.top_k)
    if len(sel) == 0:
        print("no rows in selected tiers; nothing to do", file=sys.stderr)
        return
    print(f"  extracting attention for {len(sel)} pairs (tiers={tiers_keep})",
          file=sys.stderr)

    # Join consensus with pairs to recover sequences/SMILES
    merged = sel.merge(
        pairs[["uniprot", "sequence", "chembl_id", "smiles"]],
        on=["uniprot", "chembl_id"], how="left",
    )

    # Build model + ckpt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = args.ckpt or CANONICAL_CKPT_BY_CORPUS[args.corpus]
    if not ckpt.exists():
        sys.exit(f"checkpoint not found: {ckpt}")
    model = build_model(args.config, device)
    load_checkpoint(model, ckpt, device)

    # Encode unique only
    unique_prot = merged[["uniprot", "sequence"]].drop_duplicates(["uniprot"])
    unique_lig  = merged[["chembl_id", "smiles"]].drop_duplicates(["chembl_id"])

    print("  encoding proteins...", file=sys.stderr)
    esm_m, esm_a = load_esm2_8m(device)
    prot_mats = encode_proteins(esm_m, esm_a,
                                unique_prot.itertuples(index=False, name=None),
                                device)
    del esm_m, esm_a
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("  encoding ligands...", file=sys.stderr)
    mol_m, mol_t = load_molformer(device)
    lig_mats = encode_ligands(mol_m, mol_t,
                              unique_lig.itertuples(index=False, name=None),
                              device)
    del mol_m, mol_t
    if device.type == "cuda":
        torch.cuda.empty_cache()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for _, row in merged.iterrows():
        pair_id = f"{row['uniprot']}__{row['chembl_id']}"
        try:
            pmat = prot_mats[row["uniprot"]]
            lmat = lig_mats[row["chembl_id"]]
        except KeyError:
            continue
        att = extract_pair(model, pmat, lmat, device)

        sub = args.out_dir / pair_id
        sub.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            sub / "dtkinase_Mk.npz",
            Mk_raw=att.get("Mk_raw"),
            Mk_mean=att.get("Mk_mean"),
            per_head=att.get("per_head"),
            prot_imp=att.get("prot_imp"),
            lig_imp=att.get("lig_imp"),
        )
        if "hierpool_prot" in att or "hierpool_lig" in att:
            np.savez_compressed(
                sub / "dtkinase_hierpool.npz",
                prot_weights=att.get("hierpool_prot"),
                lig_weights=att.get("hierpool_lig"),
            )
        if not args.no_plot:
            plot_consensus_heatmap(att, pair_id, sub / "consensus_heatmap.pdf")
        print(f"  {pair_id}: logit={att['logit']:+.3f}", file=sys.stderr)


if __name__ == "__main__":
    main()
