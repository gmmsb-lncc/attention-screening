"""TOST equivalence sensitivity analysis (D3 of statistical protocol).

For each model pair (A, B) and corpus, computes the paired bootstrap
IC95 of Delta_metric and declares equivalence (IC95 contained in
[-delta_eq, +delta_eq]) under multiple delta_eq bands:

- Absolute SESOI bands: 0.03, 0.05 (primary), 0.07.
- Cohen-anchored bands (Lakens 2017, SESOI-via-d): 0.2, 0.5, 0.8 times
  sigma_pooled (the pooled standard deviation of paired per-seed deltas
  across all model pairs in the corpus).

The primary band 0.05 MCC is preserved as the SESOI operational anchor
declared in docs/01-methodology/statistical_protocol.md (D3); the other
bands are sensitivity layers required by Ash/Wognum 2025 G3.

Reuses scripts.thesis_followups.bootstrap_ci.paired_delta for the
IC95 computation; n_boot defaults to 10**4.

CLI:
    python -m scripts.statistical_analysis.tost_sensitivity \\
        --corpus non_human --metric mcc \\
        --out results/statistical/non_human/tost.json
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)

from . import (
    DEFAULT_BOOTSTRAP_B, MODELS, PRIMARY_METRICS, PRIMARY_TOST_BAND, SEEDS,
    data_loader,
)
from scripts.thesis_followups.bootstrap_ci import paired_delta as _paired_delta


def _to_bootstrap_dict(seed_data: dict) -> dict:
    """Adapt data_loader output to bootstrap_ci.paired_delta input shape."""
    return {
        "logits": seed_data["logits"],
        "labels": seed_data["y_true"],
        "threshold": seed_data["threshold"],
    }


def _per_seed_metric(seed_data: dict, metric: str) -> float:
    y = seed_data["y_true"]
    p = seed_data["y_prob"]
    pred = (p >= seed_data["threshold"]).astype(np.int64)
    if metric == "mcc":
        return float(matthews_corrcoef(y, pred)) if y.std() > 0 else 0.0
    if metric == "auroc":
        return float(roc_auc_score(y, p)) if len(set(y)) == 2 else 0.5
    if metric == "auprc":
        return float(average_precision_score(y, p)) if len(set(y)) == 2 else float(np.mean(y))
    if metric == "f1":
        return float(f1_score(y, pred, zero_division=0))
    if metric == "accuracy":
        return float(accuracy_score(y, pred))
    if metric == "precision":
        return float(precision_score(y, pred, zero_division=0))
    if metric == "recall":
        return float(recall_score(y, pred, zero_division=0))
    raise ValueError(metric)


def _sigma_pooled(panel: dict, metric: str) -> float:
    """Pooled SD of paired per-seed deltas across all 6 unordered pairs."""
    models = list(panel.keys())
    per_seed_metrics = {
        m: np.array([_per_seed_metric(s, metric) for s in panel[m]],
                    dtype=np.float64)
        for m in models
    }
    deltas = []
    for a, b in itertools.combinations(models, 2):
        d = per_seed_metrics[a] - per_seed_metrics[b]
        deltas.append(d)
    stacked = np.concatenate(deltas)
    if len(stacked) < 2:
        return 0.0
    return float(np.std(stacked, ddof=1))


def _resolve_bands(spec: list[str], sigma_pooled: float) -> list[dict]:
    """Resolve band specifications into numeric (label, value, kind) tuples."""
    out = []
    for s in spec:
        s = s.strip().lower()
        if s.endswith("sigma"):
            mult = float(s[:-5])
            value = mult * sigma_pooled
            label = f"{mult:.1f}sigma"
            kind = "cohen_d"
        else:
            value = float(s)
            label = f"{value:.3f}"
            kind = "absolute"
        out.append({"label": label, "value": value, "kind": kind,
                    "primary": (kind == "absolute"
                                and abs(value - PRIMARY_TOST_BAND) < 1e-9)})
    return out


def tost_sensitivity_panel(corpus: str, metric: str,
                          bands_spec: list[str],
                          n_bootstrap: int = DEFAULT_BOOTSTRAP_B,
                          rng_seed: int = 0) -> dict:
    panel = data_loader.load_panel(corpus, MODELS, SEEDS)
    rng = np.random.default_rng(rng_seed)

    # Convert to bootstrap_ci.paired_delta input.
    boot_panel = {m: [_to_bootstrap_dict(s) for s in panel[m]] for m in MODELS}

    # Compute paired deltas + IC95 for every ordered pair (only need
    # unordered, but ordered makes downstream tables symmetric).
    pair_results = {}
    for a, b in itertools.permutations(MODELS, 2):
        cmp = _paired_delta(boot_panel[a], boot_panel[b],
                            n_boot=n_bootstrap, rng=rng)
        m = cmp[metric]
        pair_results[f"{a}__vs__{b}"] = {
            "model_a": a,
            "model_b": b,
            "median_delta": float(m["median_delta"]),
            "ci_lo": float(m["ci_95"][0]),
            "ci_hi": float(m["ci_95"][1]),
            "wilcoxon_p": float(m["wilcoxon_p"]),
            "per_seed_deltas": list(map(float, m["per_seed"])),
        }

    sigma_pooled = _sigma_pooled(panel, metric)
    bands = _resolve_bands(bands_spec, sigma_pooled)

    by_band = {}
    counts = {}
    for band in bands:
        delta_eq = band["value"]
        equiv_pairs = {}
        equiv_count = 0
        # Use unordered pairs for the count (6 unique pairs out of 4 models).
        seen = set()
        for key, p in pair_results.items():
            unordered_key = "__".join(sorted([p["model_a"], p["model_b"]]))
            if unordered_key in seen:
                continue
            seen.add(unordered_key)
            equiv = (p["ci_lo"] >= -delta_eq) and (p["ci_hi"] <= delta_eq)
            equiv_pairs[unordered_key] = {
                "model_a": p["model_a"],
                "model_b": p["model_b"],
                "ci_lo": p["ci_lo"],
                "ci_hi": p["ci_hi"],
                "median_delta": p["median_delta"],
                "equivalent": bool(equiv),
            }
            if equiv:
                equiv_count += 1
        by_band[band["label"]] = {
            "delta_eq": float(delta_eq),
            "kind": band["kind"],
            "primary": bool(band["primary"]),
            "n_equivalent": equiv_count,
            "n_total": len(equiv_pairs),
            "pairs": equiv_pairs,
        }
        counts[band["label"]] = equiv_count

    return {
        "corpus": corpus,
        "metric": metric,
        "n_bootstrap": int(n_bootstrap),
        "sigma_pooled": float(sigma_pooled),
        "primary_band": PRIMARY_TOST_BAND,
        "bands": bands,
        "by_band": by_band,
        "summary_counts": counts,
        "ordered_pair_results": pair_results,
        "note": (
            "Primary band 0.05 MCC is the SESOI-anchored equivalence "
            "threshold declared in statistical_protocol.md (D3). Cohen-"
            "anchored bands (Lakens 2017) reported as sensitivity only."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True, choices=("human", "non_human", "all"))
    ap.add_argument("--metric", default="mcc", choices=PRIMARY_METRICS)
    ap.add_argument("--bands", nargs="+",
                    default=["0.03", "0.05", "0.07", "0.2sigma", "0.5sigma", "0.8sigma"])
    ap.add_argument("--n-bootstrap", type=int, default=DEFAULT_BOOTSTRAP_B)
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    result = tost_sensitivity_panel(args.corpus, args.metric, args.bands,
                                   args.n_bootstrap, args.rng_seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        json.dump(result, fh, indent=2)

    print(f"[tost] corpus={args.corpus} metric={args.metric} "
          f"sigma_pooled={result['sigma_pooled']:.4f}")
    print(f"  Equivalent pairs (of 6) per band:")
    for label, count in result["summary_counts"].items():
        primary = " (PRIMARY)" if result["by_band"][label]["primary"] else ""
        delta = result["by_band"][label]["delta_eq"]
        print(f"    delta_eq={label} (={delta:+.4f}){primary}: "
              f"{count}/6 equivalent")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
