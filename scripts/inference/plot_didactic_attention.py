"""Render didactic attention figure for thesis Anexo B.

Inputs:
  - dtkinase_Mk.npz (Mk_mean, residue × token interaction map)
  - <pair_id>_attention.json (sequence + ligand graph)
Outputs:
  - <out>.pdf (4-panel composite: sequence track, molecule, token bars, heatmap zoom)

Usage:
  env/bin/python scripts/inference/plot_didactic_attention.py \
      --pair-dir results/inference/imatinib_demo/attention/173__CHEMBL941 \
      --pair-name "Imatinibe ↔ ABL1" \
      --out figures/attention_didactic_imatinib_abl.pdf
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from PIL import Image

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D


def smiles_char_to_atom_idx(smi: str) -> dict:
    mapping = {}
    i = 0
    atom_idx = 0
    n = len(smi)
    while i < n:
        c = smi[i]
        if c == "[":
            j = smi.find("]", i)
            if j == -1:
                i += 1
                continue
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


def render_mol_image(mol, atom_attn, size=(620, 480)):
    cmap = plt.get_cmap("plasma")
    norm = Normalize(vmin=float(atom_attn.min()), vmax=float(atom_attn.max()) + 1e-12)
    n_atoms = mol.GetNumAtoms()
    hl_colors = {i: tuple(cmap(norm(float(atom_attn[i])))[:3]) for i in range(n_atoms)}
    hl_radii = {i: 0.42 for i in range(n_atoms)}

    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    opts = drawer.drawOptions()
    opts.fillHighlights = True
    opts.bondLineWidth = 2
    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(range(n_atoms)),
        highlightAtomColors=hl_colors,
        highlightAtomRadii=hl_radii,
    )
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair-dir", required=True, type=Path)
    p.add_argument("--pair-name", default="(par)")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--top-residues", type=int, default=8)
    p.add_argument("--heatmap-residues", type=int, default=30)
    args = p.parse_args()

    mk_path = args.pair_dir / "dtkinase_Mk.npz"
    json_files = list(args.pair_dir.glob("*_attention.json"))
    if not json_files:
        # try parent
        json_files = list(args.pair_dir.parent.glob(f"{args.pair_dir.name}_attention.json"))
    if not json_files:
        raise SystemExit(f"no *_attention.json found in {args.pair_dir} or parent")
    json_path = json_files[0]

    npz = np.load(mk_path)
    Mk_mean = npz["Mk_mean"]
    per_head = npz["per_head"] if "per_head" in npz.files else None
    with open(json_path) as f:
        meta = json.load(f)

    sp, sl = Mk_mean.shape
    # Per-residue: max over tokens (find binding-site hotspots)
    prot_score = Mk_mean.max(axis=1)
    # Per-token: average over the top-attended residues only (filters
    # out residues that broadcast uniformly across all tokens)
    n_top_res = max(20, sp // 50)
    top_res = np.argsort(prot_score)[-n_top_res:]
    tok_score = Mk_mean[top_res, :].mean(axis=0)
    tok_score = np.clip(tok_score, 0, None)  # only positive contributions

    canon = meta["ligand"]["canonical_smiles"]
    mol = Chem.MolFromMolBlock(meta["ligand"]["molblock"]) if meta["ligand"]["molblock"] else Chem.MolFromSmiles(canon)
    n_atoms = mol.GetNumAtoms()

    char_to_atom = smiles_char_to_atom_idx(canon)
    atom_attn = np.zeros(n_atoms)
    atom_n = np.zeros(n_atoms)
    n_chars = min(len(canon), len(tok_score))
    for cp in range(n_chars):
        a = char_to_atom.get(cp)
        if a is not None and a < n_atoms:
            atom_attn[a] += float(tok_score[cp])
            atom_n[a] += 1
    atom_attn = atom_attn / np.clip(atom_n, 1.0, None)

    residues = meta["protein"]["residues"]

    fig = plt.figure(figsize=(11.5, 12.0))
    gs = fig.add_gridspec(
        3, 2,
        height_ratios=[1.05, 1.55, 1.55],
        width_ratios=[1.25, 1.0],
        hspace=0.85,
        wspace=0.30,
    )

    # --- Panel A: protein sequence track ---
    ax_a = fig.add_subplot(gs[0, :])
    positions = np.arange(1, sp + 1)
    ax_a.fill_between(positions, prot_score, color="#5b8def", alpha=0.45, linewidth=0)
    ax_a.plot(positions, prot_score, color="#1f4ec3", linewidth=0.6, alpha=0.8)
    top_idx = np.argsort(prot_score)[-args.top_residues:][::-1]
    # de-conflict overlapping labels by sorting by position and staggering
    sorted_for_label = sorted(top_idx, key=lambda i: i)
    last_x = -1e9
    min_dx = sp // 40
    y_max = float(prot_score.max())
    label_levels = [1.06, 1.14, 1.22]
    level = 0
    for i in sorted_for_label:
        pos = i + 1
        aa = residues[i]["aa"]
        ax_a.scatter([pos], [prot_score[i]], color="#d62728", s=22, zorder=5)
        if pos - last_x < min_dx:
            level = (level + 1) % len(label_levels)
        else:
            level = 0
        ax_a.annotate(
            f"{aa}{pos}",
            xy=(pos, prot_score[i]),
            xytext=(pos, y_max * label_levels[level]),
            textcoords="data",
            fontsize=7.5,
            color="#7d2020",
            ha="center",
            fontweight="bold",
            arrowprops=dict(arrowstyle="-", color="#bbbbbb", lw=0.5),
        )
        last_x = pos
    ax_a.set_ylim(0, y_max * 1.32)
    ax_a.set_xlim(0, sp + 1)
    ax_a.set_xlabel("Posição na sequência protéica (resíduo)", fontsize=10)
    ax_a.set_ylabel("Atenção máx. sobre tokens", fontsize=10)
    ax_a.set_title(
        f"(a) Faixa de atenção sobre a sequência protéica · {args.pair_name}",
        fontsize=11, loc="left",
    )
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.tick_params(labelsize=8)

    # --- Panel B-left: molecule with atoms colored by attention ---
    ax_b = fig.add_subplot(gs[1, 0])
    img = render_mol_image(mol, atom_attn, size=(720, 540))
    ax_b.imshow(img)
    ax_b.axis("off")
    ax_b.set_title(
        "(b) Estrutura 2D do ligante · átomos colorizados por atenção",
        fontsize=11, loc="left",
    )
    cbar_ax = fig.add_axes([0.085, 0.385, 0.30, 0.011])
    cmap = plt.get_cmap("plasma")
    norm = Normalize(vmin=float(atom_attn.min()), vmax=float(atom_attn.max()))
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax, orientation="horizontal",
    )
    cb.set_label("Atenção por átomo", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # --- Panel B-right: per-head intensity ---
    ax_b2 = fig.add_subplot(gs[1, 1])
    if per_head is not None:
        K = len(per_head)
        bars = ax_b2.bar(np.arange(K), per_head, color="#2ca02c", alpha=0.85, width=0.7)
        for k, v in enumerate(per_head):
            ax_b2.text(k, v + per_head.max() * 0.02, f"{v:.2f}",
                       ha="center", fontsize=7.5)
        ax_b2.set_xticks(np.arange(K))
        ax_b2.set_xticklabels([f"$k_{{{i}}}$" for i in range(K)], fontsize=9)
        ax_b2.set_xlabel("Cabeça de atenção (head)", fontsize=10)
        ax_b2.set_ylabel("Intensidade média $|\\overline{M}_k|$", fontsize=10)
        ax_b2.set_title("(c) Intensidade por cabeça (K=8)", fontsize=11, loc="left")
        ax_b2.set_ylim(0, per_head.max() * 1.18)
    else:
        ax_b2.bar(np.arange(sl), tok_score, color="#2ca02c", alpha=0.85, width=0.8)
        ax_b2.set_xlabel("Posição BPE (token SMILES)", fontsize=10)
        ax_b2.set_ylabel("Atenção sobre resíduos", fontsize=10)
        ax_b2.set_title("(c) Atenção por token SMILES", fontsize=11, loc="left")
    ax_b2.spines["top"].set_visible(False)
    ax_b2.spines["right"].set_visible(False)
    ax_b2.tick_params(labelsize=8)

    # --- Panel C: 2D heatmap zoom on top residues × all tokens ---
    ax_c = fig.add_subplot(gs[2, :])
    top_n = args.heatmap_residues
    top_idx_full = np.argsort(prot_score)[-top_n:][::-1]  # descending
    sub = Mk_mean[top_idx_full, :]
    im = ax_c.imshow(sub, aspect="auto", cmap="plasma", interpolation="nearest")
    ax_c.set_yticks(np.arange(len(top_idx_full)))
    yticklabels = [f"{residues[i]['aa']}{i + 1}" for i in top_idx_full]
    ax_c.set_yticklabels(yticklabels, fontsize=6.5)
    ax_c.set_xlabel("Posição do token SMILES (BPE)", fontsize=10)
    ax_c.set_ylabel(f"Top {top_n} resíduos protéicos (por atenção máx.)", fontsize=10)
    ax_c.set_title(
        "(d) Mapa de atenção bimodal $\\overline{M}$ (média sobre cabeças) · zoom em "
        + f"{top_n} resíduos mais atendidos",
        fontsize=11, loc="left",
    )
    ax_c.tick_params(labelsize=8)
    cbar2 = fig.colorbar(im, ax=ax_c, fraction=0.025, pad=0.012)
    cbar2.set_label("Intensidade $\\overline{M}_{ij}$", fontsize=8)
    cbar2.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"Atenção bimodal DT-Kinase · {args.pair_name}",
        fontsize=13.5,
        fontweight="bold",
        y=0.995,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
