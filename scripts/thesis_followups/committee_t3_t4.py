#!/usr/bin/env python3
"""T3 (correlação de erros entre os 4 modelos) + T4 (sensibilidade MCC-optimal
para todos) — pós-processamento CPU sobre predições de teste já salvas.

Reusa os helpers canônicos do comitê (alinhamento por posição validado por
y_true, dedup por (seq_id, chembl_id)). Sem GPU, sem re-treino.

T3: vetores de erro e_m = (pred_m != y_true) ao limiar nativo de cada modelo;
    matriz de correlação (phi) pareada + diversidade (par DrugBAN x GraphBAN,
    que compartilham o núcleo BAN, deve ser o mais correlacionado).

T4: cada baseline (DrugBAN/GraphBAN, nativamente F1-optimal) re-calibrado sob
    MCC-optimal derivado NA VALIDAÇÃO (sem leakage de teste) e aplicado ao teste;
    DT-Kinase e ConPLex já são MCC-optimal. Mostra se os gaps DT-vs-BAN mudam
    quando todos são julgados sob MCC-optimal.
"""
import sys, json
from pathlib import Path
import numpy as np

HERE = Path("/Users/sulfierry/attention-screening/scripts/inference/experiments")
sys.path.insert(0, str(HERE))
from committee_vs_individual import (  # noqa: E402
    load_5seed, dedupe_predictions, load_test_keys, _mcc_fast, model_paths,
)

SEEDS = [42, 123, 456, 789, 1024]
CORPORA = ["non_human", "human", "all"]
MODELS = ["dtkinase", "drugban", "graphban", "conplex"]
MCC_OPT_NATIVE = {"dtkinase", "conplex"}  # já calibrados por MCC-optimal


def mcc_opt_threshold(prob, y):
    """Limiar que maximiza MCC sobre (prob, y); grade de 1001 pontos."""
    cand = np.linspace(0.0, 1.0, 1001)
    best_t, best_m = 0.5, -2.0
    for t in cand:
        m = _mcc_fast(y, (prob >= t).astype(int))
        if m > best_m:
            best_m, best_t = m, t
    return best_t


def load_val_5seed(model, corpus):
    """(prob_mean_val, y_true_val) para baselines (npz com val_y_*)."""
    probs, yref = [], None
    for s in SEEDS:
        npz, _ = model_paths(model, corpus, s)
        d = np.load(npz)
        if "val_y_prob" not in d.files:
            return None, None
        y = d["val_y_true"].astype(int)
        probs.append(d["val_y_prob"].astype(np.float64))
        yref = y if yref is None else yref
        assert np.array_equal(yref, y)
    return np.mean(np.stack(probs), axis=0), yref


def phi(a, b):
    """Correlação phi (Pearson) entre dois vetores binários de erro."""
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


print("# T3 (correlação de erros) + T4 (MCC-optimal para todos)\n")
for corpus in CORPORA:
    # ---- carregar 4 modelos alinhados + dedup (espelha run_corpus) ----
    probs, thr_native, y = {}, {}, None
    for m in MODELS:
        p, yt, t = load_5seed(m, corpus)
        probs[m], thr_native[m] = p, t
        y = yt if y is None else y
        assert np.array_equal(y, yt)
    dedup_ok = True
    try:
        keys, seq_ids = load_test_keys(corpus)
        yd = None
        for m in MODELS:
            pd_, yd_, _ = dedupe_predictions(probs[m], y, keys)
            probs[m] = pd_
            yd = yd_ if yd is None else yd
            assert np.array_equal(yd, yd_)
        y = yd
    except FileNotFoundError:
        dedup_ok = False  # TSV de chaves ausente local -> usa teste alinhado bruto
    n, pos = len(y), int(y.sum())

    tag = "dedup" if dedup_ok else "SEM dedup (TSV chaves ausente)"
    print(f"## Corpus {corpus}  (n={n} {tag}, {pos} pos / {n-pos} neg)\n")

    # ---- T3: erros ao limiar nativo + correlação ----
    err = {m: (((probs[m] >= thr_native[m]).astype(int)) != y).astype(float)
           for m in MODELS}
    print("### T3 — matriz de correlação de erros (phi, limiar nativo)")
    print("model     " + "  ".join(f"{m[:7]:>7}" for m in MODELS))
    offdiag = []
    for a in MODELS:
        row = []
        for b in MODELS:
            r = 1.0 if a == b else phi(err[a], err[b])
            row.append(r)
            if a < b:
                offdiag.append((a, b, r))
        print(f"{a:9s} " + "  ".join(f"{v:7.3f}" for v in row))
    er = {m: err[m].mean() for m in MODELS}
    print("taxa erro: " + "  ".join(f"{m[:7]}={er[m]:.3f}" for m in MODELS))
    offdiag.sort(key=lambda x: -x[2])
    print("pares + correlacionados: " +
          "; ".join(f"{a}-{b}={r:.3f}" for a, b, r in offdiag[:3]))
    print(f"correlação média off-diagonal: {np.mean([r for *_ , r in offdiag]):.3f}\n")

    # ---- T4: MCC nativo vs MCC sob MCC-optimal (val) para todos ----
    print("### T4 — MCC nativo vs MCC-optimal-para-todos (limiar derivado na validação)")
    print("model      MCC_nativo  MCC_optimal  Δ      criterio_nativo")
    mcc_opt = {}
    for m in MODELS:
        mcc_nat = _mcc_fast(y, (probs[m] >= thr_native[m]).astype(int))
        if m in MCC_OPT_NATIVE:
            t_opt, crit = thr_native[m], "MCC-opt (nativo)"
        else:
            pv, yv = load_val_5seed(m, corpus)
            if pv is None:
                t_opt, crit = thr_native[m], "F1-opt (sem val)"
            else:
                t_opt, crit = mcc_opt_threshold(pv, yv), "F1-opt->MCC-opt(val)"
        mcc_o = _mcc_fast(y, (probs[m] >= t_opt).astype(int))
        mcc_opt[m] = mcc_o
        print(f"{m:9s}  {mcc_nat:9.3f}  {mcc_o:10.3f}  {mcc_o-mcc_nat:+.3f}  {crit}")
    # gaps DT-Kinase vs BAN sob MCC-optimal-para-todos
    dtk = mcc_opt["dtkinase"]
    print("gaps DT-Kinase vs baselines sob MCC-optimal-para-todos: " +
          "; ".join(f"{b}={dtk-mcc_opt[b]:+.3f}" for b in ["drugban", "graphban", "conplex"]))
    print()
