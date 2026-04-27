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

def _topk_indices(arr: np.ndarray, k: int = 10) -> np.ndarray:
    """Return indices of the top-k values in arr, sorted by descending value."""
    if arr is None or len(arr) == 0:
        return np.array([], dtype=int)
    k = min(k, len(arr))
    idx = np.argpartition(arr, -k)[-k:]
    return idx[np.argsort(-arr[idx])]


# ======================================================================
# JSON export — structured graph data for downstream visualization
# ======================================================================

def _smiles_char_to_atom_idx(smi: str) -> dict:
    """Map SMILES character position → RDKit atom index.

    Walks the SMILES string and assigns each character that's part of
    an atom token (single-letter, two-letter halogens, or bracket atom
    `[...]`) to a sequential atom index. Bond, ring-closure, and
    parenthesis characters are not mapped. The output is suitable for
    aggregating per-character attention into per-atom attention.
    """
    mapping = {}
    i = 0
    atom_idx = 0
    n = len(smi)
    while i < n:
        c = smi[i]
        if c == "[":
            j = smi.find("]", i)
            if j == -1:
                i += 1; continue
            for k in range(i, j + 1):
                mapping[k] = atom_idx
            atom_idx += 1
            i = j + 1
        elif c in "Cc" and i + 1 < n and smi[i + 1] == "l":
            mapping[i] = atom_idx
            mapping[i + 1] = atom_idx
            atom_idx += 1
            i += 2
        elif c in "Bb" and i + 1 < n and smi[i + 1] == "r":
            mapping[i] = atom_idx
            mapping[i + 1] = atom_idx
            atom_idx += 1
            i += 2
        elif c in "BCNOSPFIbcnosp":
            mapping[i] = atom_idx
            atom_idx += 1
            i += 1
        else:
            i += 1
    return mapping


def _build_ligand_graph(smiles: str, token_attention: np.ndarray) -> dict:
    """Build a JSON-serializable ligand graph annotated with per-atom attention.

    Output schema (Cytoscape / D3 / NetworkX compatible):
      {
        "smiles": str,
        "canonical_smiles": str,
        "molblock": <RDKit 2D MOL block, or "" if RDKit unavailable>,
        "n_atoms": int,
        "atoms": [
          {"idx": int, "element": str, "x": float, "y": float,
           "attention": float, "rank": int|null, "is_top": bool},
          ...
        ],
        "bonds": [
          {"source": int, "target": int, "order": int, "aromatic": bool},
          ...
        ]
      }

    Per-atom attention is the mean of per-SMILES-char attention values
    over the chars belonging to that atom (single-letter, halogen pair,
    or bracket-atom contents). Atoms with no chars in the (truncated)
    token window get attention = 0.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        return {"smiles": smiles, "canonical_smiles": smiles,
                "molblock": "", "n_atoms": 0, "atoms": [], "bonds": []}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"smiles": smiles, "canonical_smiles": smiles,
                "molblock": "", "n_atoms": 0, "atoms": [], "bonds": []}

    canon = Chem.MolToSmiles(mol, canonical=True)
    n_atoms = mol.GetNumAtoms()

    # Aggregate token attention → atom attention via SMILES char walk.
    char_to_atom = _smiles_char_to_atom_idx(canon)
    atom_attn = np.zeros(n_atoms, dtype=float)
    atom_n    = np.zeros(n_atoms, dtype=float)
    if token_attention is not None:
        n_chars = min(len(canon), len(token_attention))
        for char_pos in range(n_chars):
            a = char_to_atom.get(char_pos)
            if a is not None and a < n_atoms:
                atom_attn[a] += float(token_attention[char_pos])
                atom_n[a] += 1
    atom_attn = atom_attn / np.clip(atom_n, 1.0, None)

    # 2D coordinates for graph visualization.
    try:
        mol_2d = Chem.Mol(mol)
        AllChem.Compute2DCoords(mol_2d)
        conf = mol_2d.GetConformer()
        coords = [(conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y)
                  for i in range(n_atoms)]
        molblock = Chem.MolToMolBlock(mol_2d)
    except Exception:
        coords = [(0.0, 0.0)] * n_atoms
        molblock = ""

    # Top atoms by attention (rank 1 = strongest).
    top_idx = list(np.argsort(-atom_attn))
    rank_of = {int(idx): r + 1 for r, idx in enumerate(top_idx)}
    top_set = set(int(idx) for idx in top_idx[:5])

    atoms = []
    for i in range(n_atoms):
        a = mol.GetAtomWithIdx(i)
        atoms.append({
            "idx":       i,
            "element":   a.GetSymbol(),
            "aromatic":  a.GetIsAromatic(),
            "charge":    a.GetFormalCharge(),
            "x":         float(coords[i][0]),
            "y":         float(coords[i][1]),
            "attention": float(atom_attn[i]),
            "rank":      rank_of.get(i),
            "is_top":    i in top_set,
        })
    bonds = [{
        "source":   b.GetBeginAtomIdx(),
        "target":   b.GetEndAtomIdx(),
        "order":    int(b.GetBondTypeAsDouble()),
        "aromatic": b.GetIsAromatic(),
    } for b in mol.GetBonds()]

    return {
        "smiles":           smiles,
        "canonical_smiles": canon,
        "molblock":         molblock,
        "n_atoms":          n_atoms,
        "atoms":            atoms,
        "bonds":            bonds,
    }


def plot_sequence_track(att: dict, pair_id: str, out_dir: Path,
                         sequence: str | None,
                         top_residues: int = 12) -> Path | None:
    """Render the protein sequence as a horizontal attention track.

    Output: <pair_id>_sequence_track.png

    Designed for the "show me which region of the sequence the model
    looked at" question. The plot is a single horizontal band where
    each cell is a residue colored by attention intensity (inferno
    colormap). The top-K residues are annotated above the band with
    their position + AA letter, and a sliding-window mean is drawn
    underneath to highlight contiguous high-attention regions.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import colors as mcolors
    except ImportError:
        return None

    res_attn = att.get("hierpool_prot")
    if res_attn is None:
        res_attn = att.get("prot_imp")
    if res_attn is None or len(res_attn) == 0:
        return None
    res_attn = np.asarray(res_attn, dtype=float)
    n = len(res_attn)
    seq_eff = (sequence[:n] if sequence else "X" * n)

    # Sliding-window mean to highlight contiguous regions
    win = max(5, n // 50)
    pad = np.pad(res_attn, win // 2, mode="edge")
    smooth = np.convolve(pad, np.ones(win) / win, mode="valid")[:n]

    fig = plt.figure(figsize=(min(20, max(10, n * 0.020)), 4.2))
    gs  = fig.add_gridspec(2, 1, height_ratios=[1.6, 1.0], hspace=0.35)
    ax_track  = fig.add_subplot(gs[0])
    ax_smooth = fig.add_subplot(gs[1], sharex=ax_track)

    # Top track: per-residue heatmap (1 row × n cols)
    norm = mcolors.Normalize(vmin=res_attn.min(), vmax=res_attn.max())
    band = res_attn[np.newaxis, :]
    im = ax_track.imshow(band, aspect="auto", cmap="inferno", norm=norm,
                         interpolation="nearest", extent=[0, n, 0, 1])
    ax_track.set_yticks([])
    ax_track.set_ylim(0, 1)
    ax_track.set_xlim(0, n)
    ax_track.set_title(f"DT-Kinase attention along protein sequence — {pair_id}\n"
                       f"(n = {n} residues; logit = {att.get('logit', 0.0):+.2f})",
                       fontsize=11)
    cbar = plt.colorbar(im, ax=ax_track, fraction=0.04, pad=0.02,
                        orientation="vertical")
    cbar.set_label("attention weight", fontsize=8)

    # Annotate top-K residues
    top = _topk_indices(res_attn, top_residues)
    for r in top:
        r = int(r)
        aa = seq_eff[r] if r < len(seq_eff) else "X"
        ax_track.annotate(f"{r+1}·{aa}",
                          xy=(r + 0.5, 1.0),
                          xytext=(0, 4), textcoords="offset points",
                          ha="center", fontsize=7,
                          color="#A04500", weight="bold",
                          rotation=0)

    # Bottom: smoothed signal showing contiguous regions
    ax_smooth.plot(np.arange(n), smooth, color="#2E86AB", linewidth=1.6)
    ax_smooth.fill_between(np.arange(n), smooth, alpha=0.25, color="#2E86AB")
    ax_smooth.axhline(np.percentile(res_attn, 90), color="#E8630A",
                      linestyle="--", linewidth=0.9, label="P90 threshold")
    ax_smooth.set_xlabel("residue position (N → C)", fontsize=9)
    ax_smooth.set_ylabel(f"sliding mean (w={win})", fontsize=8)
    ax_smooth.set_xlim(0, n)
    ax_smooth.legend(loc="upper right", fontsize=7, framealpha=0.9)

    fig.tight_layout()
    out_path = out_dir / f"{pair_id}_sequence_track.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def plot_ligand_2d(att: dict, pair_id: str, out_dir: Path,
                    smiles: str | None) -> Path | None:
    """Render the ligand as a 2D molecule with atoms colored by attention.

    Output: <pair_id>_ligand_2d.png

    Uses RDKit's drawing engine to render the canonical SMILES with
    a per-atom color overlay derived from the per-SMILES-character
    HierPool weights. Atoms in the top-K by attention are circled and
    labeled; bond order and aromaticity are rendered standard.
    """
    if not smiles:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from rdkit import Chem
        from rdkit.Chem import AllChem, Draw
        from rdkit.Chem.Draw import rdMolDraw2D
    except ImportError:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    canon = Chem.MolToSmiles(mol, canonical=True)
    mol_canon = Chem.MolFromSmiles(canon)
    AllChem.Compute2DCoords(mol_canon)
    n_atoms = mol_canon.GetNumAtoms()

    # Aggregate token attention → per-atom attention
    char_to_atom = _smiles_char_to_atom_idx(canon)
    hp_lig = att.get("hierpool_lig")
    if hp_lig is None or len(hp_lig) == 0:
        return None

    atom_attn = np.zeros(n_atoms, dtype=float)
    atom_n    = np.zeros(n_atoms, dtype=float)
    n_chars = min(len(canon), len(hp_lig))
    for char_pos in range(n_chars):
        a = char_to_atom.get(char_pos)
        if a is not None and a < n_atoms:
            atom_attn[a] += float(hp_lig[char_pos])
            atom_n[a] += 1
    atom_attn = atom_attn / np.clip(atom_n, 1.0, None)
    if atom_attn.max() > 0:
        atom_attn_norm = (atom_attn - atom_attn.min()) / (atom_attn.max() - atom_attn.min() + 1e-12)
    else:
        atom_attn_norm = atom_attn

    # Map normalized attention → inferno color
    cmap = plt.get_cmap("inferno")
    atom_colors = {i: cmap(0.15 + 0.7 * atom_attn_norm[i])[:3]
                   for i in range(n_atoms)}
    top = _topk_indices(atom_attn, min(5, n_atoms))
    highlight_atoms = [int(i) for i in top]

    # Render mol via rdMolDraw2D (cairo PNG)
    drawer = rdMolDraw2D.MolDraw2DCairo(900, 700)
    opts = drawer.drawOptions()
    opts.padding = 0.10
    opts.bondLineWidth = 2
    opts.fixedFontSize = 18
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol_canon,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colors,
    )
    drawer.FinishDrawing()
    png_bytes = drawer.GetDrawingText()

    # Compose final PNG: molecule + colorbar legend on the side
    import io
    from PIL import Image
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    mol_img = Image.open(io.BytesIO(png_bytes))
    fig = plt.figure(figsize=(11.0, 7.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.045], wspace=0.02)
    ax_mol = fig.add_subplot(gs[0, 0])
    ax_mol.imshow(mol_img)
    ax_mol.axis("off")

    # Title + caption
    a_min = float(atom_attn.min())
    a_max = float(atom_attn.max())
    n_top = len(top)
    ax_mol.set_title(
        f"Ligand 2D · atoms colored by attention "
        f"(top {n_top} highlighted)\n"
        f"raw range: [{a_min:.4g}, {a_max:.4g}]   ·   colorbar normalized 0–1",
        fontsize=11,
    )

    # Colorbar built from the SAME truncated colormap used on atoms
    # (cairo render maps atom_attn_norm ∈ [0,1] → cmap(0.15 + 0.7·v))
    from matplotlib.colors import LinearSegmentedColormap
    sample = np.linspace(0.15, 0.85, 256)
    truncated = LinearSegmentedColormap.from_list(
        "inferno_truncated",
        [cmap(s) for s in sample],
    )
    cax = fig.add_subplot(gs[0, 1])
    norm = Normalize(vmin=0.0, vmax=1.0)
    sm = ScalarMappable(norm=norm, cmap=truncated)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="vertical")
    cb.set_label("per-atom attention (norm. 0–1)", fontsize=10)
    cb.ax.tick_params(labelsize=8)

    out_path = out_dir / f"{pair_id}_ligand_2d.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def save_attention_json(att: dict, pair_id: str, out_dir: Path,
                         sequence: str | None,
                         smiles: str | None,
                         uniprot: str | None = None) -> Path:
    """Save a structured JSON with ligand graph + protein residues + attention.

    File layout (out_dir/<pair_id>_attention.json):
      {
        "pair_id": str, "uniprot": str, "logit": float,
        "ligand":   {graph from _build_ligand_graph},
        "protein":  {sequence + per-residue attention + top-K residues},
        "interaction": {
            "shape": [sp, sl],
            "per_head": [...],
            "top_cells": [{"residue": i, "aa": "F", "token": j, "char": "c",
                           "attention": float}, ...]
        },
        "metadata": {model name, num heads, generation timestamp}
      }
    Designed for downstream graph viz (Cytoscape / D3 / Mol*) — atoms
    carry pre-computed 2D coords + attention so a viewer just needs to
    bind the value to color/size.
    """
    import json
    import time

    Mk_mean  = att.get("Mk_mean")
    per_head = att.get("per_head")
    hp_prot  = att.get("hierpool_prot")
    hp_lig   = att.get("hierpool_lig")

    # Ligand graph (with per-atom attention from token-level signal).
    ligand_block = _build_ligand_graph(smiles or "", hp_lig)

    # Per-residue annotation.
    if sequence is None:
        seq_eff = ""
    elif Mk_mean is not None:
        sp = Mk_mean.shape[0]
        seq_eff = sequence[:sp]
    else:
        seq_eff = sequence

    res_attn = hp_prot if hp_prot is not None else att.get("prot_imp")
    residues = []
    if res_attn is not None:
        n = min(len(res_attn), len(seq_eff)) if seq_eff else len(res_attn)
        rank_of = {int(idx): r + 1 for r, idx in enumerate(np.argsort(-res_attn))}
        top_set = set(int(i) for i in np.argsort(-res_attn)[:10])
        for i in range(n):
            aa = seq_eff[i] if i < len(seq_eff) else "X"
            residues.append({
                "position":  i + 1,                # 1-based for biology
                "aa":        aa,
                "attention": float(res_attn[i]),
                "rank":      rank_of.get(i),
                "is_top":    i in top_set,
            })

    # Top interaction cells (residue × token).
    top_cells = []
    if Mk_mean is not None:
        flat = Mk_mean.ravel()
        top_flat = np.argsort(-flat)[:20]
        sp, sl = Mk_mean.shape
        smi_tokens = list(smiles or "")
        for f in top_flat:
            r, t = int(f // sl), int(f % sl)
            top_cells.append({
                "residue":     r + 1,
                "aa":          seq_eff[r] if r < len(seq_eff) else "X",
                "token":       t,
                "smiles_char": smi_tokens[t] if t < len(smi_tokens) else "?",
                "attention":   float(Mk_mean[r, t]),
            })

    # Hints for downstream 3D structural visualization. The JSON itself
    # carries 1D per-residue attention; viewers that want to render the
    # signal on a 3D structure (PyMOL/ChimeraX/Mol*) can pull the
    # corresponding model from AlphaFold (when uniprot is a real ID),
    # then bind residues[].attention to the B-factor / pLDDT field.
    pdb_hint = None
    if uniprot and not uniprot.startswith("USER_") and uniprot.isalpha() is False:
        # Heuristic: assume it's a UniProt ID (e.g. P00519). The
        # AlphaFold URL pattern is stable; viewers do the actual fetch.
        # Skip if uniprot is just a numeric seq_id (e.g. "173").
        try:
            int(uniprot)
            pdb_hint = None  # numeric seq_id, no public PDB mapping
        except ValueError:
            pdb_hint = {
                "uniprot_id":     uniprot,
                "alphafold_pdb":  f"https://alphafold.ebi.ac.uk/files/AF-{uniprot}-F1-model_v4.pdb",
                "alphafold_html": f"https://alphafold.ebi.ac.uk/entry/{uniprot}",
                "rcsb_search":    f"https://www.rcsb.org/search?q={uniprot}",
                "binding_field":  "B-factor (write residue.attention into atom B-factor column)",
            }

    payload = {
        "pair_id":   pair_id,
        "uniprot":   uniprot,
        "logit":     float(att.get("logit", 0.0)),
        "ligand":    ligand_block,
        "protein": {
            "sequence_length": len(seq_eff),
            "residues":        residues,
            "pdb_hint":        pdb_hint,
        },
        "interaction": {
            "shape":      list(Mk_mean.shape) if Mk_mean is not None else None,
            "per_head":   per_head.tolist() if per_head is not None else None,
            "top_cells":  top_cells,
        },
        "metadata": {
            "model":            "DT-Kinase v7",
            "n_heads":          int(Mk_mean.shape[0]) if Mk_mean is not None and Mk_mean.ndim > 2 else None,
            "generated_utc":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "schema_version":   "1.0",
        },
    }

    out_path = out_dir / f"{pair_id}_attention.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return out_path


def plot_consensus_heatmap(att: dict, pair_id: str, out_dir: Path,
                            sequence: str | None = None,
                            smiles_tokens: list[str] | None = None,
                            top_residues: int = 10,
                            top_tokens: int = 10) -> None:
    """Render attention overview AND a focused 'hotspot' map for the pair.

    Produces three companion files inside ``out_dir``:

      - <pair_id>_attention.png         (high-DPI overview, 4 panels)
      - <pair_id>_attention.pdf         (vector overview for thesis/paper)
      - <pair_id>_hotspots.png          (single-panel focused heatmap with
                                         top residue + token annotations)

    The focused hotspot panel uses an inferno colormap (high contrast,
    perceptually uniform) and overlays:
      - row labels with the AA letter at the top-K most attended positions
      - column labels with the SMILES token at the top-K most attended cols
      - white tick marks at those positions

    so that the user can immediately read off which protein region and
    which ligand substructure the model focused on.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import colors as mcolors
    except ImportError:
        print(f"  matplotlib unavailable; skipping plot for {pair_id}",
              file=sys.stderr)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    Mk_mean       = att.get("Mk_mean")            # [sp, sl]
    prot_imp      = att.get("prot_imp")           # [sp]
    lig_imp       = att.get("lig_imp")            # [sl]
    per_head      = att.get("per_head")           # [K]
    hp_prot       = att.get("hierpool_prot")      # [sp]
    hp_lig        = att.get("hierpool_lig")       # [sl]

    # Pick the strongest residue/token signal: prefer HierPool weights when
    # available (these are explicitly the model's attention output), else
    # fall back to Mk_mean per-axis sums.
    prot_signal = hp_prot if hp_prot is not None else prot_imp
    lig_signal  = hp_lig  if hp_lig  is not None else lig_imp

    # ------------------------------------------------------------------
    # FILE 1: focused hotspot heatmap
    # ------------------------------------------------------------------
    if Mk_mean is not None:
        sp, sl = Mk_mean.shape
        fig, ax = plt.subplots(figsize=(max(8, sl * 0.18),
                                        max(6, sp * 0.04 + 2)))
        # Use inferno: dark = low attention, bright = high attention.
        im = ax.imshow(Mk_mean, aspect="auto", cmap="inferno",
                       interpolation="nearest")
        ax.set_title(f"DT-Kinase attention hotspot — {pair_id}\n"
                     f"(logit = {att['logit']:+.2f}; mean over {Mk_mean.shape[0]}×{Mk_mean.shape[1]} cells)",
                     fontsize=11)
        ax.set_xlabel("ligand SMILES token  →")
        ax.set_ylabel("protein residue (N → C)")
        cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label("attention intensity (M̄ over 16 heads)", fontsize=9)

        # Annotate top-K residues + tokens
        top_res = _topk_indices(prot_signal, top_residues) if prot_signal is not None else []
        top_tok = _topk_indices(lig_signal,  top_tokens)   if lig_signal  is not None else []

        # Sparse y-tick labels with AA letter at top residues
        if len(top_res) and sequence is not None:
            seq = sequence[:sp]
            yticks_idx = sorted(int(i) for i in top_res)
            yticks_lbl = [f"{i+1}·{seq[i]}" if i < len(seq) else str(i+1)
                          for i in yticks_idx]
            ax.set_yticks(yticks_idx)
            ax.set_yticklabels(yticks_lbl, fontsize=8)
        elif sp > 40:
            step = max(1, sp // 25)
            ax.set_yticks(range(0, sp, step))
            ax.set_yticklabels(range(1, sp + 1, step), fontsize=7)

        if len(top_tok) and smiles_tokens is not None:
            xticks_idx = sorted(int(i) for i in top_tok)
            xticks_lbl = [f"{i}·{smiles_tokens[i]}" if i < len(smiles_tokens) else str(i)
                          for i in xticks_idx]
            ax.set_xticks(xticks_idx)
            ax.set_xticklabels(xticks_lbl, fontsize=8, rotation=45, ha="right")
        elif sl > 30:
            step = max(1, sl // 20)
            ax.set_xticks(range(0, sl, step))
            ax.set_xticklabels(range(0, sl, step), fontsize=7)

        # White overlay rectangles around top-3 residue × top-3 token cells
        for r in top_res[:3]:
            for t in top_tok[:3]:
                ax.add_patch(plt.Rectangle((t - 0.5, r - 0.5), 1, 1,
                                           linewidth=1.5, edgecolor="white",
                                           facecolor="none"))

        fig.tight_layout()
        fig.savefig(out_dir / f"{pair_id}_hotspots.png", dpi=180,
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)

    # ------------------------------------------------------------------
    # FILE 2: 4-panel overview (PNG)
    # ------------------------------------------------------------------
    # Wider figure + extra wspace so each bar plot has room on its right
    # for the top-10 highlighted-values list outside the data area.
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 10.5))
    plt.subplots_adjust(wspace=0.55, hspace=0.40)
    fig.suptitle(f"DT-Kinase attention overview — {pair_id}  "
                 f"(logit = {att['logit']:+.2f})", fontsize=12, weight="bold")

    if Mk_mean is not None:
        ax = axes[0, 0]
        # Row-centered map exposes per-token deviation from each residue's
        # mean — without this, residue-dominant signal (rows nearly uniform)
        # collapses the visible structure to horizontal stripes.
        row_mean = Mk_mean.mean(axis=1, keepdims=True)
        Mk_centered = Mk_mean - row_mean
        v = float(np.abs(Mk_centered).max()) + 1e-12
        im = ax.imshow(
            Mk_centered, aspect="auto", cmap="RdBu_r",
            vmin=-v, vmax=+v, interpolation="nearest",
        )
        ax.set_title(
            "M̄ row-centered: per-residue deviation across ligand tokens\n"
            f"(raw range Mk: [{float(Mk_mean.min()):+.3f}, {float(Mk_mean.max()):+.3f}])",
            fontsize=9,
        )
        ax.set_xlabel("ligand token")
        ax.set_ylabel("protein residue (N → C)")
        cb = plt.colorbar(im, ax=ax, fraction=0.04)
        cb.set_label("M̄ − row-mean", fontsize=8)
        cb.ax.tick_params(labelsize=7)

    def _normalized_bar(ax, signal, top_k_idx, labels, value_label,
                        title, xlabel, list_title):
        """Bar plot with signal min-max normalized to [0, 1].

        Top-K bars are highlighted in orange; their identifiers + values
        are listed in a right-margin text block (outside the data area)
        so they never collide with bar tops, which solves the overlap
        seen when many top hits cluster in a narrow X range.
        """
        signal = np.asarray(signal, dtype=float)
        raw_min = float(signal.min())
        raw_max = float(signal.max())
        denom = max(raw_max - raw_min, 1e-12)
        norm = (signal - raw_min) / denom
        n = len(norm)

        bars = ax.bar(range(n), norm, color="#2E86AB",
                      edgecolor="none", width=1.0)
        for i in top_k_idx:
            bars[int(i)].set_color("#E8630A")
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(0, 1.05)
        ax.set_title(
            f"{title}\nraw range: [{raw_min:.4g}, {raw_max:.4g}]",
            fontsize=9,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("attention weight (norm. 0–1)")

        # Right-side label list — opposite the y-axis label
        ax.text(
            1.04, 1.00, list_title,
            transform=ax.transAxes, va="top", fontsize=8.5,
            fontweight="bold", color="#A04500",
        )
        line_step = 0.085
        for k, idx in enumerate(top_k_idx):
            idx = int(idx)
            if labels is not None and idx < len(labels):
                line = f"{value_label(idx)}·{labels[idx]} = {norm[idx]:.2f}"
            else:
                line = f"{value_label(idx)} = {norm[idx]:.2f}"
            ax.text(
                1.04, 0.93 - k * line_step, line,
                transform=ax.transAxes, va="top", fontsize=7.8,
                color="#A04500", family="monospace",
            )

    if prot_signal is not None:
        top_res = _topk_indices(prot_signal, top_residues)
        seq_letters = sequence[:len(prot_signal)] if sequence is not None else None
        _normalized_bar(
            axes[0, 1], prot_signal, top_res,
            labels=seq_letters,
            value_label=lambda r: f"{r + 1}",
            title=f"Protein attention per residue  (top {top_residues} highlighted)",
            xlabel="residue position (N → C)",
            list_title=f"Top-{top_residues} residues",
        )

    if lig_signal is not None:
        top_tok = _topk_indices(lig_signal, top_tokens)
        _normalized_bar(
            axes[1, 0], lig_signal, top_tok,
            labels=None,
            value_label=lambda r: f"{r}",
            title=f"Ligand attention per token  (top {top_tokens} highlighted)",
            xlabel="ligand token position",
            list_title=f"Top-{top_tokens} tokens",
        )

    if per_head is not None:
        ax = axes[1, 1]
        ph = np.asarray(per_head, dtype=float)
        ph_max = float(ph.max())
        # Normalize by max only (preserve relative magnitudes — head with
        # lowest intensity stays visible instead of collapsing to zero)
        ph_norm = ph / max(ph_max, 1e-12)
        bars = ax.bar(range(len(ph_norm)), ph_norm, color="#1B813E",
                      edgecolor="white", linewidth=0.5)
        for k, v in enumerate(ph_norm):
            ax.text(k, v + 0.02, f"{v:.2f}",
                    ha="center", fontsize=7.5, color="#0a3a1a")
        ax.set_title(
            f"Per-head intensity (|M_k|.mean across heads)\n"
            f"raw range: [{float(ph.min()):.3g}, {ph_max:.3g}]",
            fontsize=9,
        )
        ax.set_xlabel("interaction-map head index k")
        ax.set_ylabel("|M_k| mean (norm. by max)")
        ax.set_ylim(0, 1.15)

    fig.savefig(out_dir / f"{pair_id}_attention.png", dpi=180,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ======================================================================
# CLI
# ======================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Extract DT-Kinase attention for top hits.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--consensus", type=Path,
                     help="committee consensus.csv (multi-model mode); "
                          "selects rows by --tier and --top-k.")
    src.add_argument("--scores",    type=Path,
                     help="single-model scores csv (cols: uniprot, chembl_id, "
                          "prob, [pred, threshold]); ranks by prob descending.")
    ap.add_argument("--pairs",     type=Path, required=True)
    ap.add_argument("--out-dir",   type=Path, required=True)
    ap.add_argument("--top-k",     type=int, default=20)
    ap.add_argument("--tier",      type=str, default="STRONG,LIKELY",
                    help="(--consensus only) comma-separated subset of "
                         "{STRONG,LIKELY,UNCERTAIN,UNLIKELY}")
    ap.add_argument("--min-prob",  type=float, default=None,
                    help="(--scores only) minimum probability to keep "
                         "(default: keep top-k regardless of prob).")
    ap.add_argument("--corpus",    choices=["human", "non_human", "all"], default="all")
    ap.add_argument("--ckpt",      type=Path, default=None)
    ap.add_argument("--config",    type=Path, default=CANONICAL_CONFIG)
    ap.add_argument("--no-plot",   action="store_true")
    args = ap.parse_args()

    pairs = pd.read_csv(args.pairs, sep="\t")

    if args.consensus is not None:
        consensus = pd.read_csv(args.consensus)
        tiers_keep = {t.strip().upper() for t in args.tier.split(",") if t.strip()}
        sel = consensus[consensus["tier"].isin(tiers_keep)].head(args.top_k)
        if len(sel) == 0:
            print("no rows in selected tiers; nothing to do", file=sys.stderr)
            return
        print(f"  extracting attention for {len(sel)} pairs (tiers={tiers_keep})",
              file=sys.stderr)
    else:
        scores = pd.read_csv(args.scores)
        if "prob" not in scores.columns:
            sys.exit("--scores csv must have a 'prob' column")
        sel = scores.sort_values("prob", ascending=False)
        if args.min_prob is not None:
            sel = sel[sel["prob"] >= args.min_prob]
        sel = sel.head(args.top_k).copy()
        if len(sel) == 0:
            print("no rows passed --min-prob filter; nothing to do",
                  file=sys.stderr)
            return
        print(f"  extracting attention for {len(sel)} pairs "
              f"(top-{args.top_k} by prob, single-model mode)",
              file=sys.stderr)

    # Join consensus with pairs to recover sequences/SMILES
    # Cast keys to str on both sides to avoid type mismatch from CSV sniffing.
    sel["uniprot"]   = sel["uniprot"].astype(str)
    sel["chembl_id"] = sel["chembl_id"].astype(str)
    pairs["uniprot"]   = pairs["uniprot"].astype(str)
    pairs["chembl_id"] = pairs["chembl_id"].astype(str)
    merged = sel.merge(
        pairs[["uniprot", "sequence", "chembl_id", "smiles"]],
        on=["uniprot", "chembl_id"], how="left",
    )

    # Filter rows where sequence/smiles couldn't be recovered (NaN merge result):
    # this happens if pairs.tsv chembl_id space doesn't intersect consensus
    # chembl_id space (e.g., lookup-mode demo where pairs.tsv was generated
    # for one query SMILES but consensus rows came from a different chembl_id
    # via raw_predictions.npz indexing).
    n_total = len(merged)
    merged = merged.dropna(subset=["sequence", "smiles"])
    n_dropped = n_total - len(merged)
    if n_dropped:
        print(f"  WARN: {n_dropped}/{n_total} pairs lack sequence/SMILES in pairs.tsv "
              f"(chembl_id mismatch between consensus and pairs); skipping these.",
              file=sys.stderr)
    if len(merged) == 0:
        print("  no pairs left after merge — re-run with a pairs.tsv that "
              "shares chembl_id with the consensus output.", file=sys.stderr)
        return

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
        seq = str(row.get("sequence", "")) or None
        smi = str(row.get("smiles", "")) or None
        smiles_tokens = list(smi) if smi else None

        if not args.no_plot:
            # Pass sequence + SMILES tokens so the plot annotates top
            # residues with their AA letter and top tokens with the SMILES
            # character at that position.
            plot_consensus_heatmap(att, pair_id, sub,
                                   sequence=seq,
                                   smiles_tokens=smiles_tokens)
            # Sequence track + 2D ligand structure (the two viz the user
            # cares about for "where in the sequence/molecule does the
            # model focus").
            plot_sequence_track(att, pair_id, sub, sequence=seq)
            plot_ligand_2d(att, pair_id, sub, smiles=smi)

        # Structured JSON: ligand atomic graph + per-atom attention +
        # per-residue attention + top interaction cells. Consumable by
        # Cytoscape, D3.js, NetworkX, or a Mol* viewer for visualization
        # of the strongest attention regions on the molecular graph.
        try:
            save_attention_json(att, pair_id, sub,
                                sequence=seq, smiles=smi,
                                uniprot=str(row.get("uniprot", "")))
        except Exception as e:
            print(f"  {pair_id}: JSON export failed: {e}", file=sys.stderr)

        print(f"  {pair_id}: logit={att['logit']:+.3f}", file=sys.stderr)


if __name__ == "__main__":
    main()
