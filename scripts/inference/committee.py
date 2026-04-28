"""Committee orchestrator: run all 4 models on a pair list, aggregate consensus.

Pipeline:
    1. expand_pairs.py        — input → pairs.tsv
    2. parallel subprocess    — each model scores pairs.tsv in its own conda env
    3. aggregate.py           — merge 4 CSVs → consensus.csv
    4. attention.py (optional) — extract 3-source attention for top-K hits

Usage:
    python committee.py --smiles "CCO..." --organism human --out results/run_001
    python committee.py --fasta seq.fa    --out results/run_002
    python committee.py --smiles "CCO..." --fasta seq.fa --out results/run_003
    python committee.py --pairs batch.tsv --out results/run_004 --top-k 50

Outputs in <out>/:
    pairs.tsv
    scores_dtkinase.csv, scores_drugban.csv, scores_graphban.csv, scores_conplex.csv
    consensus.csv
    consensus.top.csv (if --top-k > 0)
    attention/<pair_id>/
        <pair_id>_attention.png       overview 4-panel (PNG, ~250 KB)
        <pair_id>_attention.pdf       vector overview (PDF)
        <pair_id>_hotspots.png        focused single-panel heatmap with
                                      top residues + tokens annotated
        dtkinase_Mk.npz               raw [K, sp, sl] tensor + agg
        dtkinase_hierpool.npz         stage 1 + stage 2 attention weights
        drugban_BAN.npz, graphban_BAN.npz   (when adapters available)
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


REPO_ROOT = Path(__file__).resolve().parents[2]
INFER_DIR = Path(__file__).resolve().parent

# Each baseline lives in its own conda env per CLAUDE.md. The model scoring
# scripts are thin adapters that load the canonical checkpoint, score the
# pairs.tsv, and emit a uniform CSV. They live in scripts/inference/models/.
MODEL_RUNNERS = {
    "dtkinase": {
        "env":    "attention-screening",
        "script": INFER_DIR / "models" / "dtkinase_score.py",
    },
    "drugban": {
        "env":    "drugban",
        "script": INFER_DIR / "models" / "drugban_score.py",
    },
    "graphban": {
        "env":    "graphban",
        "script": INFER_DIR / "models" / "graphban_score.py",
    },
    "conplex": {
        "env":    "conplex",
        "script": INFER_DIR / "models" / "conplex_score.py",
    },
}


def run_expand(args: argparse.Namespace, out_dir: Path) -> Path:
    """Invoke expand_pairs.py to produce pairs.tsv."""
    pairs_path = out_dir / "pairs.tsv"
    cmd = [
        sys.executable, str(INFER_DIR / "expand_pairs.py"),
        "--out", str(pairs_path),
        "--organism", args.organism,
    ]
    if args.smiles: cmd += ["--smiles", args.smiles]
    if args.fasta:  cmd += ["--fasta",  args.fasta]
    if args.pairs:  cmd += ["--pairs",  args.pairs]
    if args.family: cmd += ["--family", args.family]
    subprocess.run(cmd, check=True)
    return pairs_path


def run_model(model: str, pairs_path: Path, out_dir: Path,
              ckpt_corpus: str, single_env: str | None = None,
              dry_run: bool = False) -> tuple[str, Path]:
    """Invoke a single baseline scoring script.

    Two execution modes:
      - Per-model conda envs (default): uses ``conda run -n {env}`` to switch
        between {attention-screening, drugban, graphban, conplex}. This handles
        version conflicts between baseline upstream repos.
      - Unified env (``single_env`` set, e.g. ``baseline``): uses ``conda run
        -n {single_env}`` for ALL models. Requires a single env that satisfies
        the union of dependencies (see scripts/inference/setup_baseline_env.sh).
        Eliminates per-model env switching overhead and disk usage.

    Each adapter (scripts/inference/models/<model>_score.py) reads pairs.tsv
    and emits a uniform CSV (uniprot, chembl_id, prob, pred, threshold) that
    aggregate.py merges via outer-join.
    """
    cfg = MODEL_RUNNERS[model]
    target_env = single_env if single_env else cfg["env"]
    out_path = out_dir / f"scores_{model}.csv"
    cmd = [
        "conda", "run", "-n", target_env, "--no-capture-output",
        "python", str(cfg["script"]),
        "--pairs",  str(pairs_path),
        "--out",    str(out_path),
        "--corpus", ckpt_corpus,
    ]
    if dry_run:
        print(f"[dry-run] {' '.join(cmd)}")
        return model, out_path
    subprocess.run(cmd, check=True)
    if not out_path.exists():
        raise RuntimeError(f"{model} did not produce {out_path}")
    return model, out_path


def run_aggregate(out_dir: Path, top_k: int) -> Path:
    """Invoke aggregate.py to produce consensus.csv."""
    consensus_path = out_dir / "consensus.csv"
    cmd = [
        sys.executable, str(INFER_DIR / "aggregate.py"),
        "--scores-dir", str(out_dir),
        "--out",        str(consensus_path),
        "--top-k",      str(top_k),
    ]
    subprocess.run(cmd, check=True)
    return consensus_path


def run_attention(out_dir: Path, consensus_path: Path, pairs_path: Path,
                  top_k: int, ckpt_corpus: str = "all") -> None:
    """Extract attention for top-K STRONG/LIKELY pairs.

    Generates per-pair PNG + PDF visualizations under
    <out_dir>/attention/<pair_id>/:
      <pair_id>_attention.png    overview 4-panel
      <pair_id>_attention.pdf    vector overview
      <pair_id>_hotspots.png     focused single-panel heatmap with top
                                 residues + tokens annotated
      dtkinase_Mk.npz            raw [K, sp, sl] tensor + agg stats
      dtkinase_hierpool.npz      stage 1 + stage 2 weights
    """
    att_script = INFER_DIR / "attention.py"
    if not att_script.exists():
        print("attention.py not yet implemented — skipping attention extraction")
        return
    cmd = [
        sys.executable, str(att_script),
        "--consensus", str(consensus_path),
        "--pairs",     str(pairs_path),
        "--out-dir",   str(out_dir / "attention"),
        "--top-k",     str(top_k),
        "--tier",      "STRONG,LIKELY",
        "--corpus",    ckpt_corpus,
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="DT-Kinase committee inference orchestrator.")
    grp = ap.add_argument_group("input (pick one combination)")
    grp.add_argument("--smiles", type=str, help="single SMILES string")
    grp.add_argument("--fasta",  type=str, help="single-sequence FASTA file")
    grp.add_argument("--pairs",  type=str, help="batch TSV (uniprot, sequence, chembl_id, smiles)")

    ap.add_argument("--organism", choices=["human", "non_human", "all"], default="human",
                    help="kinome subset for SMILES-only mode")
    ap.add_argument("--family", type=str, default=None,
                    help="optional kinase family filter (TK, CMGC, AGC, ...)")
    ap.add_argument("--ckpt-corpus", choices=["human", "non_human", "all"], default="all",
                    help="which training-corpus checkpoint each model loads "
                         "(canonical mode: same corpus for all 4 models)")
    ap.add_argument("--hybrid", action="store_true",
                    help="hybrid dispatch: DT-Kinase uses --ckpt-corpus, baselines "
                         "(DrugBAN/GraphBAN/ConPLex) override to 'all' regardless of "
                         "--ckpt-corpus. Empirically optimal for non_human eval "
                         "(committee MCC +0.036) and marginal for human eval (+0.008). "
                         "See Anexo B §B.3.1 of the thesis for the ablation study.")
    ap.add_argument("--out",     type=Path, required=True,
                    help="output directory for this run")
    ap.add_argument("--top-k",   type=int, default=20,
                    help="emit attention + ranked subset for top-K pairs")
    ap.add_argument("--models",  type=str, default="dtkinase,drugban,graphban,conplex",
                    help="comma-separated subset of models to run")
    ap.add_argument("--parallel", action="store_true",
                    help="run model scoring in parallel (requires 4× GPU memory)")
    ap.add_argument("--single-env", type=str, default=None, metavar="ENV_NAME",
                    help="run all models in a single conda env (e.g. 'baseline'); "
                         "see scripts/inference/setup_baseline_env.sh")
    ap.add_argument("--dry-run", action="store_true",
                    help="print model commands without executing")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    selected = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in selected if m not in MODEL_RUNNERS]
    if unknown:
        sys.exit(f"unknown model(s): {unknown}; valid: {list(MODEL_RUNNERS)}")

    print(f"[1/4] expanding pairs → {args.out}/pairs.tsv")
    pairs_path = run_expand(args, args.out)

    # Hybrid dispatch: DT-K keeps --ckpt-corpus; baselines pinned to 'all'.
    def _ckpt_corpus_for(model: str) -> str:
        if args.hybrid and model != "dtkinase":
            return "all"
        return args.ckpt_corpus

    env_mode = f"single env '{args.single_env}'" if args.single_env else "per-model envs"
    mode_tag = "HYBRID" if args.hybrid else "canonical"
    print(f"[2/4] scoring with models: {selected} ({env_mode}, {mode_tag})")
    if args.hybrid:
        for m in selected:
            print(f"      {m:9s} ← ckpt_corpus={_ckpt_corpus_for(m)}")
    if args.parallel:
        with ThreadPoolExecutor(max_workers=len(selected)) as ex:
            futures = [ex.submit(run_model, m, pairs_path, args.out,
                                 _ckpt_corpus_for(m), args.single_env, args.dry_run)
                       for m in selected]
            for fut in as_completed(futures):
                model, out_path = fut.result()
                print(f"  ✓ {model} → {out_path}")
    else:
        for m in selected:
            model, out_path = run_model(m, pairs_path, args.out,
                                        _ckpt_corpus_for(m),
                                        args.single_env, args.dry_run)
            print(f"  ✓ {model} → {out_path}")

    if args.dry_run:
        print("[dry-run] skipping aggregate + attention")
        return

    print(f"[3/4] aggregating → {args.out}/consensus.csv")
    consensus_path = run_aggregate(args.out, args.top_k)

    print(f"[4/4] attention extraction (top-{args.top_k} STRONG/LIKELY)")
    run_attention(args.out, consensus_path, pairs_path, args.top_k,
                  ckpt_corpus=args.ckpt_corpus)

    print(f"done. consensus → {consensus_path}")


if __name__ == "__main__":
    main()
