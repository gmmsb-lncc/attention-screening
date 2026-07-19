#!/usr/bin/env python3
"""Teste descorrelação arquitetural vs. mera média (cap5 L808 'não conduzido').

Compara, em cada corpus:
  - ganho de um ENSEMBLE DE SEMENTES de um único modelo (DT-Kinase): PoE sobre
    4 sementes vs. MCC médio de semente única (puro efeito de averaging, mesmo
    paradigma/calibração);
  - ganho do COMITÊ 4-MODELOS (PoE sobre os 4 modelos seed-averaged) sobre o
    melhor modelo individual (efeito de diversidade arquitetural).

Se ganho_comitê > ganho_sementes, a diversidade arquitetural adiciona valor além
da redução de variância por média. CPU, sobre predições salvas (sem dedup).
"""
import sys
import numpy as np

HERE = "/Users/sulfierry/attention-screening/scripts/inference/experiments"
sys.path.insert(0, HERE)
from committee_vs_individual import load_5seed, _mcc_fast, model_paths  # noqa: E402

SEEDS = [42, 123, 456, 789, 1024]
CORPORA = ["non_human", "human", "all"]
MODELS = ["dtkinase", "drugban", "graphban", "conplex"]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def poe_score(probs_list):
    """Log-odds do Produto-de-Especialistas (monotônico com p_PoE)."""
    return np.sum([logit(p) for p in probs_list], axis=0)


def mcc_opt(score, y):
    cand = np.unique(score)
    if len(cand) > 1500:
        cand = np.quantile(cand, np.linspace(0, 1, 1500))
    best = -2.0
    for t in cand:
        m = _mcc_fast(y, (score >= t).astype(int))
        if m > best:
            best = m
    return best


def load_per_seed(model, corpus):
    out, y = [], None
    for s in SEEDS:
        npz, _ = model_paths(model, corpus, s)
        d = np.load(npz)
        pk = "test_y_prob" if "test_y_prob" in d.files else "y_prob"
        tk = "test_y_true" if "test_y_true" in d.files else "y_true"
        out.append(d[pk].astype(np.float64))
        yy = d[tk].astype(int)
        y = yy if y is None else y
    return out, y


print("# Teste: diversidade arquitetural vs. média (ensemble de sementes)\n")
for corpus in CORPORA:
    # ensemble de sementes (DT-Kinase): 4 sementes
    seed_probs, y = load_per_seed("dtkinase", corpus)
    single = [mcc_opt(logit(p), y) for p in seed_probs]
    mean_single = float(np.mean(single))
    ens4 = mcc_opt(poe_score(seed_probs[:4]), y)        # n=4 sementes (L808)
    ens5 = mcc_opt(poe_score(seed_probs), y)            # n=5 (completude)
    gain_seed = ens4 - mean_single

    # comitê 4-modelos (cada modelo seed-averaged)
    avg = {m: load_5seed(m, corpus)[0] for m in MODELS}
    ind = {m: mcc_opt(logit(avg[m]), y) for m in MODELS}
    comm = mcc_opt(poe_score([avg[m] for m in MODELS]), y)
    best = max(ind.values())
    gain_comm = comm - best

    print(f"## {corpus}  (n={len(y)}, sem dedup)")
    print(f"  semente única (MCC-opt, média 5): {mean_single:.3f}")
    print(f"  ensemble 4 sementes DT-Kinase:    {ens4:.3f}  (ganho {gain_seed:+.3f})")
    print(f"  ensemble 5 sementes DT-Kinase:    {ens5:.3f}")
    print(f"  melhor modelo individual:         {best:.3f}")
    print(f"  comitê 4-modelos (PoE):           {comm:.3f}  (ganho {gain_comm:+.3f})")
    verdict = ("DIVERSIDADE > média" if gain_comm > gain_seed + 0.005
               else "comparável a média" if abs(gain_comm - gain_seed) <= 0.005
               else "média > diversidade")
    print(f"  -> ganho_comitê {gain_comm:+.3f} vs ganho_sementes {gain_seed:+.3f}: {verdict}\n")
