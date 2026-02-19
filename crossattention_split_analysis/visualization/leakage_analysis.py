"""Legacy leakage diagnostics and plots (01-05) for baseline analysis."""

import os
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Plot style (kept consistent with the baseline scripts)
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12

_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def create_stratified_split(
    df: pd.DataFrame,
    seed: int,
    test_fraction: float,
    val_fraction: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create train/val/test indices with stratification when possible."""
    indices = np.arange(len(df), dtype=np.int64)
    labels = np.asarray(df["label"].values)

    if val_fraction <= 0.0:
        try:
            train_idx, test_idx = train_test_split(
                indices, test_size=test_fraction, stratify=labels, random_state=seed
            )
        except ValueError:
            train_idx, test_idx = train_test_split(
                indices, test_size=test_fraction, random_state=seed
            )
        return train_idx, np.array([], dtype=np.int64), test_idx

    holdout_fraction = test_fraction + val_fraction
    try:
        train_idx, holdout_idx = train_test_split(
            indices, test_size=holdout_fraction, stratify=labels, random_state=seed
        )
    except ValueError:
        train_idx, holdout_idx = train_test_split(
            indices, test_size=holdout_fraction, random_state=seed
        )

    holdout_labels = labels[holdout_idx]
    relative_test = test_fraction / holdout_fraction
    try:
        val_idx, test_idx = train_test_split(
            holdout_idx,
            test_size=relative_test,
            stratify=holdout_labels,
            random_state=seed + 1,
        )
    except ValueError:
        val_idx, test_idx = train_test_split(
            holdout_idx, test_size=relative_test, random_state=seed + 1
        )

    return train_idx, val_idx, test_idx


def analyze_compound_leakage(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> Dict:
    """Analyze leakage between train/test by compound and exact pair duplication."""
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    train_compounds = set(train_df["chembl_id"].unique())
    test_compounds = set(test_df["chembl_id"].unique())
    leaked_compounds = train_compounds & test_compounds
    new_compounds = test_compounds - train_compounds

    test_leaked_mask = test_df["chembl_id"].isin(leaked_compounds)
    n_test_leaked = int(test_leaked_mask.sum())

    train_pairs = set(zip(train_df["chembl_id"], train_df["target_kinase"]))
    test_pairs = set(zip(test_df["chembl_id"], test_df["target_kinase"]))
    exact_duplicates = train_pairs & test_pairs
    test_exact_mask = test_df.apply(
        lambda row: (row["chembl_id"], row["target_kinase"]) in exact_duplicates, axis=1
    )
    n_exact = int(test_exact_mask.sum())

    return {
        "train_compounds": train_compounds,
        "test_compounds": test_compounds,
        "leaked_compounds": leaked_compounds,
        "new_compounds": new_compounds,
        "n_test_leaked_lines": n_test_leaked,
        "n_test_total": int(len(test_df)),
        "exact_duplicates": exact_duplicates,
        "n_exact_lines": n_exact,
    }


def plot_leakage_analysis(leakage_info: Dict, output_dir: str, prefix: str = "") -> str:
    """Generate pie charts for leakage diagnostics (01)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1 = axes[0]
    n_leaked = leakage_info["n_test_leaked_lines"]
    n_total = leakage_info["n_test_total"]
    n_new = n_total - n_leaked

    sizes = [n_leaked, n_new]
    labels = [
        f"Composto Vazado\n{n_leaked} ({100*n_leaked/n_total:.1f}%)",
        f"Composto Novo\n{n_new} ({100*n_new/n_total:.1f}%)",
    ]
    ax1.pie(
        sizes,
        labels=labels,
        colors=["#ff6b6b", "#4ecdc4"],
        explode=(0.05, 0.0),
        autopct="",
        startangle=90,
        textprops={"fontsize": 11},
    )
    ax1.set_title("Vazamento de Compostos nas Linhas de Teste", fontsize=14, fontweight="bold")

    ax2 = axes[1]
    n_exact = leakage_info["n_exact_lines"]
    n_not_exact = n_total - n_exact
    sizes2 = [n_exact, n_not_exact]
    labels2 = [
        f"Duplicata Exata\n{n_exact} ({100*n_exact/n_total:.1f}%)",
        f"Nao Duplicata\n{n_not_exact} ({100*n_not_exact/n_total:.1f}%)",
    ]
    ax2.pie(
        sizes2,
        labels=labels2,
        colors=["#ff8c42", "#6c5ce7"],
        explode=(0.05, 0.0),
        autopct="",
        startangle=90,
        textprops={"fontsize": 11},
    )
    ax2.set_title("Duplicatas Exatas (Composto + Quinase)", fontsize=14, fontweight="bold")

    plt.tight_layout()
    filename = f"{prefix}01_leakage_analysis.png" if prefix else "01_leakage_analysis.png"
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def _compute_morgan_fingerprints(smiles_list: list) -> Tuple[np.ndarray, list]:
    """Compute Morgan fingerprints and return valid source indices."""
    fps = []
    valid_idx = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fps.append(_MORGAN_GEN.GetFingerprintAsNumPy(mol).astype(np.float32))
            valid_idx.append(i)
    if not fps:
        return np.empty((0, 2048), dtype=np.float32), []
    return np.stack(fps), valid_idx


def _lookup_baseline_compound(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    compound_majority = train_df.groupby("chembl_id")["label"].agg(
        lambda x: 1 if x.mean() >= 0.5 else 0
    ).to_dict()
    global_majority = 1 if train_df["label"].mean() >= 0.5 else 0
    return np.array([compound_majority.get(c, global_majority) for c in test_df["chembl_id"]], dtype=np.int64)


def _lookup_baseline_kinase(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    kinase_majority = train_df.groupby("target_kinase")["label"].agg(
        lambda x: 1 if x.mean() >= 0.5 else 0
    ).to_dict()
    global_majority = 1 if train_df["label"].mean() >= 0.5 else 0
    return np.array([kinase_majority.get(k, global_majority) for k in test_df["target_kinase"]], dtype=np.int64)


def _lookup_baseline_compound_kinase(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    pair_class = train_df.groupby(["chembl_id", "target_kinase"])["label"].agg(
        lambda x: 1 if x.mean() >= 0.5 else 0
    ).to_dict()
    compound_majority = train_df.groupby("chembl_id")["label"].agg(
        lambda x: 1 if x.mean() >= 0.5 else 0
    ).to_dict()
    kinase_majority = train_df.groupby("target_kinase")["label"].agg(
        lambda x: 1 if x.mean() >= 0.5 else 0
    ).to_dict()
    global_majority = 1 if train_df["label"].mean() >= 0.5 else 0

    preds = []
    for _, row in test_df.iterrows():
        pair = (row["chembl_id"], row["target_kinase"])
        if pair in pair_class:
            preds.append(pair_class[pair])
        elif row["chembl_id"] in compound_majority:
            preds.append(compound_majority[row["chembl_id"]])
        elif row["target_kinase"] in kinase_majority:
            preds.append(kinase_majority[row["target_kinase"]])
        else:
            preds.append(global_majority)
    return np.array(preds, dtype=np.int64)


def _train_knn_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    train_fp, train_valid = _compute_morgan_fingerprints(train_df["canonical_smiles"].tolist())
    test_fp, test_valid = _compute_morgan_fingerprints(test_df["canonical_smiles"].tolist())
    if len(train_fp) == 0 or len(test_fp) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    all_kinases = list(set(train_df["target_kinase"].unique()) | set(test_df["target_kinase"].unique()))
    kinase_to_idx = {k_name: i for i, k_name in enumerate(all_kinases)}

    train_kin_oh = np.zeros((len(train_valid), len(all_kinases)), dtype=np.float32)
    test_kin_oh = np.zeros((len(test_valid), len(all_kinases)), dtype=np.float32)

    for i, idx in enumerate(train_valid):
        train_kin_oh[i, kinase_to_idx[train_df.iloc[idx]["target_kinase"]]] = 1.0
    for i, idx in enumerate(test_valid):
        test_kin_oh[i, kinase_to_idx[test_df.iloc[idx]["target_kinase"]]] = 1.0

    X_train = np.hstack([train_fp, train_kin_oh])
    X_test = np.hstack([test_fp, test_kin_oh])
    y_train = train_df.iloc[train_valid]["label"].values
    y_test = test_df.iloc[test_valid]["label"].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=k, weights="distance", metric="cosine", n_jobs=-1)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    return pred, y_test


def evaluate_lookup_baselines(
    df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, knn_k: int = 5
) -> Dict:
    """Evaluate lookup baselines and original KNN baseline (02)."""
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    y_true = test_df["label"].values

    results = {}
    pred_comp = _lookup_baseline_compound(train_df, test_df)
    results["Lookup: Composto"] = {
        "accuracy": float(accuracy_score(y_true, pred_comp)),
        "mcc": float(matthews_corrcoef(y_true, pred_comp)),
    }

    pred_kin = _lookup_baseline_kinase(train_df, test_df)
    results["Lookup: Quinase"] = {
        "accuracy": float(accuracy_score(y_true, pred_kin)),
        "mcc": float(matthews_corrcoef(y_true, pred_kin)),
    }

    pred_both = _lookup_baseline_compound_kinase(train_df, test_df)
    results["Lookup: Comp+Quin"] = {
        "accuracy": float(accuracy_score(y_true, pred_both)),
        "mcc": float(matthews_corrcoef(y_true, pred_both)),
    }

    pred_knn, y_knn = _train_knn_baseline(train_df, test_df, k=knn_k)
    if len(y_knn) > 0:
        results["KNN (Original)"] = {
            "accuracy": float(accuracy_score(y_knn, pred_knn)),
            "mcc": float(matthews_corrcoef(y_knn, pred_knn)),
        }
    else:
        results["KNN (Original)"] = {"accuracy": float("nan"), "mcc": float("nan")}

    return results


def plot_baseline_comparison(results: Dict, output_dir: str, prefix: str = "") -> str:
    """Generate bar chart for lookup and KNN baselines (02)."""
    fig, ax = plt.subplots(figsize=(12, 7))

    models = list(results.keys())
    accuracies = [results[m]["accuracy"] for m in models]
    mccs = [results[m]["mcc"] for m in models]
    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width / 2, accuracies, width, label="Accuracy", color="#3498db", edgecolor="black")
    bars2 = ax.bar(x + width / 2, mccs, width, label="MCC", color="#e74c3c", edgecolor="black")

    ax.set_xlabel("Modelo", fontsize=12)
    ax.set_ylabel("Valor da Metrica", fontsize=12)
    ax.set_title(
        'Comparacao: KNN vs Lookup Baselines\n(Modelos que apenas "olham" a classe majoritaria)',
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend(loc="upper left")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.7, color="gray", linestyle="--", alpha=0.7)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        if np.isnan(h):
            continue
        ax.annotate(
            f"{h:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    filename = f"{prefix}02_baseline_comparison.png" if prefix else "02_baseline_comparison.png"
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def analyze_kinase_imbalance(df: pd.DataFrame, train_idx: np.ndarray) -> pd.DataFrame:
    """Analyze label imbalance per kinase on training split."""
    train_df = df.iloc[train_idx]
    kinase_stats = train_df.groupby("target_kinase").agg(
        n_samples=("label", "count"),
        n_active=("label", "sum"),
        prop_active=("label", "mean"),
    ).reset_index()

    def classify_balance(prop: float) -> str:
        if prop > 0.8 or prop < 0.2:
            return "Desbalanceada (>80% ou <20%)"
        if prop > 0.6 or prop < 0.4:
            return "Moderada"
        return "Balanceada (40-60%)"

    kinase_stats["balance_class"] = kinase_stats["prop_active"].apply(classify_balance)
    return kinase_stats


def plot_kinase_distribution(kinase_stats: pd.DataFrame, output_dir: str, prefix: str = "") -> str:
    """Generate kinase imbalance visualizations (03)."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    ax1 = axes[0]
    counts, _, _ = ax1.hist(
        kinase_stats["prop_active"], bins=20, color="#9b59b6", edgecolor="black", alpha=0.8
    )
    y_max = counts.max() if len(counts) else 1.0
    ax1.set_xlabel("Proporcao de Ativos por Quinase", fontsize=12)
    ax1.set_ylabel("Numero de Quinases", fontsize=12)
    ax1.set_title("Distribuicao de Classes por Quinase", fontsize=14, fontweight="bold")
    for x_val, color in [(0.2, "#c0392b"), (0.8, "#c0392b"), (0.4, "#e67e22"), (0.6, "#e67e22")]:
        ax1.axvline(x=x_val, color=color, linestyle="--", linewidth=2, alpha=0.8)
    ax1.set_ylim(0, y_max * 1.15)
    ax1.set_xlim(-0.05, 1.05)

    ax2 = axes[1]
    balance_counts = kinase_stats["balance_class"].value_counts()
    ordered = ["Desbalanceada (>80% ou <20%)", "Moderada", "Balanceada (40-60%)"]
    short_names = {
        "Desbalanceada (>80% ou <20%)": "Desbalanceadas",
        "Moderada": "Moderadas",
        "Balanceada (40-60%)": "Balanceadas",
    }
    colors = ["#e74c3c", "#f39c12", "#2ecc71"]
    plot_counts = []
    labels = []
    plot_colors = []
    for i, cat in enumerate(ordered):
        if cat in balance_counts.index:
            count = int(balance_counts[cat])
            plot_counts.append(count)
            labels.append(f"{short_names[cat]}\n{count} ({100*count/len(kinase_stats):.1f}%)")
            plot_colors.append(colors[i])

    ax2.pie(
        plot_counts,
        labels=labels,
        colors=plot_colors,
        startangle=90,
        textprops={"fontsize": 11},
        explode=[0.02] * len(plot_counts),
        wedgeprops=dict(edgecolor="white", linewidth=2),
    )
    ax2.set_title("Facilidade de Predicao por Quinase", fontsize=14, fontweight="bold")

    plt.tight_layout()
    filename = f"{prefix}03_kinase_imbalance.png" if prefix else "03_kinase_imbalance.png"
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def analyze_compound_consistency(df: pd.DataFrame, train_idx: np.ndarray) -> pd.DataFrame:
    """Analyze label consistency per compound across kinases."""
    train_df = df.iloc[train_idx]
    compound_stats = train_df.groupby("chembl_id").agg(
        n_kinases=("target_kinase", "nunique"),
        n_samples=("label", "count"),
        n_active=("label", "sum"),
        prop_active=("label", "mean"),
    ).reset_index()

    def classify(row: pd.Series) -> str:
        if row["n_kinases"] == 1:
            return "Uma quinase apenas"
        if row["prop_active"] in (0.0, 1.0):
            return "Perfeitamente consistente"
        return "Inconsistente"

    compound_stats["consistency"] = compound_stats.apply(classify, axis=1)
    return compound_stats


def plot_compound_consistency(compound_stats: pd.DataFrame, output_dir: str, prefix: str = "") -> str:
    """Generate compound consistency plots (04)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    multi_kinase = compound_stats[compound_stats["n_kinases"] > 1]

    ax1 = axes[0]
    ax1.hist(multi_kinase["prop_active"], bins=20, color="#1abc9c", edgecolor="black", alpha=0.8)
    ax1.set_xlabel("Proporcao de Atividade do Composto", fontsize=12)
    ax1.set_ylabel("Numero de Compostos", fontsize=12)
    ax1.set_title(
        "Consistencia de Classe (Multi-quinase)\nComportamento do Composto atraves de Diferentes Quinases",
        fontsize=13,
        fontweight="bold",
    )

    ax2 = axes[1]
    consistency_counts = compound_stats["consistency"].value_counts()
    labels_map = {
        "Perfeitamente consistente": "Perfeitamente\nconsistentes",
        "Inconsistente": "Inconsistentes",
        "Uma quinase apenas": "Uma quinase\napenas",
    }
    labels = [
        f"{labels_map.get(cat, cat)}\n{count} ({100*count/len(compound_stats):.1f}%)"
        for cat, count in consistency_counts.items()
    ]
    colors = ["#27ae60", "#e74c3c", "#95a5a6"]
    ax2.pie(
        consistency_counts.values,
        labels=labels,
        colors=colors[: len(consistency_counts)],
        autopct="",
        startangle=90,
        textprops={"fontsize": 10},
    )
    ax2.set_title("Consistencia dos Compostos", fontsize=14, fontweight="bold")

    plt.tight_layout()
    filename = f"{prefix}04_compound_consistency.png" if prefix else "04_compound_consistency.png"
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def analyze_chemical_similarity(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    sample_size: int = 500,
) -> Optional[Dict]:
    """Analyze max Tanimoto similarity of unseen test compounds to train compounds."""
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    train_compounds = train_df.drop_duplicates("chembl_id")[["chembl_id", "canonical_smiles"]]
    test_compounds = test_df.drop_duplicates("chembl_id")[["chembl_id", "canonical_smiles"]]
    train_compound_ids = set(train_compounds["chembl_id"])
    test_new = test_compounds[~test_compounds["chembl_id"].isin(train_compound_ids)]
    if len(test_new) == 0:
        return None

    if len(test_new) > sample_size:
        test_new = test_new.sample(n=sample_size, random_state=42)

    train_fps = []
    for _, row in train_compounds.iterrows():
        mol = Chem.MolFromSmiles(row["canonical_smiles"])
        if mol is not None:
            train_fps.append(_MORGAN_GEN.GetFingerprint(mol))
    if not train_fps:
        return None

    max_similarities = []
    for _, row in tqdm(
        test_new.iterrows(),
        total=len(test_new),
        desc="    Similarity analysis",
        disable=len(test_new) < 200,
    ):
        mol = Chem.MolFromSmiles(row["canonical_smiles"])
        if mol is None:
            continue
        test_fp = _MORGAN_GEN.GetFingerprint(mol)
        similarities = DataStructs.BulkTanimotoSimilarity(test_fp, train_fps)
        max_similarities.append(float(max(similarities)) if similarities else 0.0)

    if not max_similarities:
        return None

    max_similarities = np.array(max_similarities, dtype=np.float32)
    very_similar = int((max_similarities > 0.8).sum())
    similar = int(((max_similarities > 0.6) & (max_similarities <= 0.8)).sum())
    moderate = int(((max_similarities > 0.4) & (max_similarities <= 0.6)).sum())
    low = int((max_similarities <= 0.4).sum())
    return {
        "max_similarities": max_similarities,
        "very_similar": very_similar,
        "similar": similar,
        "moderate": moderate,
        "low": low,
    }


def plot_similarity_analysis(similarity_info: Optional[Dict], output_dir: str, prefix: str = "") -> Optional[str]:
    """Generate similarity plots (05)."""
    if similarity_info is None:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1 = axes[0]
    ax1.hist(similarity_info["max_similarities"], bins=30, color="#3498db", edgecolor="black", alpha=0.8)
    ax1.axvline(x=0.8, color="red", linestyle="--", linewidth=2, label="Alta (0.8)")
    ax1.axvline(x=0.6, color="orange", linestyle="--", linewidth=2, label="Media (0.6)")
    ax1.set_xlabel("Similaridade Tanimoto Maxima", fontsize=12)
    ax1.set_ylabel("Numero de Compostos de Teste", fontsize=12)
    ax1.set_title(
        "Similaridade ao Vizinho Mais Proximo\n(Compostos Novos no Teste vs Treino)",
        fontsize=13,
        fontweight="bold",
    )
    ax1.legend()

    ax2 = axes[1]
    sizes = [
        similarity_info["very_similar"],
        similarity_info["similar"],
        similarity_info["moderate"],
        similarity_info["low"],
    ]
    total = max(sum(sizes), 1)
    labels = [
        f"Muito similar (>0.8)\n{sizes[0]} ({100*sizes[0]/total:.1f}%)",
        f"Similar (0.6-0.8)\n{sizes[1]} ({100*sizes[1]/total:.1f}%)",
        f"Moderada (0.4-0.6)\n{sizes[2]} ({100*sizes[2]/total:.1f}%)",
        f"Baixa (<0.4)\n{sizes[3]} ({100*sizes[3]/total:.1f}%)",
    ]
    ax2.pie(
        sizes,
        labels=labels,
        colors=["#e74c3c", "#f39c12", "#3498db", "#27ae60"],
        autopct="",
        startangle=90,
        textprops={"fontsize": 10},
    )
    ax2.set_title("Distribuicao de Similaridade", fontsize=14, fontweight="bold")

    plt.tight_layout()
    filename = f"{prefix}05_similarity_analysis.png" if prefix else "05_similarity_analysis.png"
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def _save_report(
    df: pd.DataFrame,
    leakage_info: Dict,
    baseline_results: Dict,
    kinase_stats: pd.DataFrame,
    compound_stats: pd.DataFrame,
    similarity_info: Optional[Dict],
    output_dir: str,
    prefix: str = "",
) -> str:
    """Write a compact textual report mirroring the legacy workflow."""
    lines = []
    lines.append("=" * 70)
    lines.append("RELATORIO: ANALISE DE VIES NO DATASET DE QUINASES")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Total de linhas: {len(df)}")
    lines.append(f"Compostos unicos: {df['chembl_id'].nunique()}")
    lines.append(f"Quinases unicas: {df['target_kinase'].nunique()}")
    lines.append("")

    pct_leaked = 100.0 * leakage_info["n_test_leaked_lines"] / max(leakage_info["n_test_total"], 1)
    pct_exact = 100.0 * leakage_info["n_exact_lines"] / max(leakage_info["n_test_total"], 1)
    lines.append("VAZAMENTO")
    lines.append(f"Linhas de teste com composto vazado: {pct_leaked:.1f}%")
    lines.append(f"Duplicatas exatas (composto+quinase): {pct_exact:.1f}%")
    lines.append("")

    lines.append("BASELINES")
    for model_name, metrics in baseline_results.items():
        acc = metrics.get("accuracy", float("nan"))
        mcc = metrics.get("mcc", float("nan"))
        lines.append(f"{model_name}: Acc={acc:.4f}, MCC={mcc:.4f}")
    lines.append("")

    n_desbalanced = int((kinase_stats["balance_class"] == "Desbalanceada (>80% ou <20%)").sum())
    lines.append("DESBALANCEAMENTO")
    lines.append(f"Quinases desbalanceadas (>80% ou <20%): {n_desbalanced} ({100*n_desbalanced/len(kinase_stats):.1f}%)")
    lines.append("")

    lines.append("CONSISTENCIA")
    consistency_counts = compound_stats["consistency"].value_counts()
    for cat, count in consistency_counts.items():
        lines.append(f"{cat}: {count} ({100*count/len(compound_stats):.1f}%)")
    lines.append("")

    if similarity_info is not None:
        total = max(len(similarity_info["max_similarities"]), 1)
        lines.append("SIMILARIDADE")
        lines.append(f"Muito similar (>0.8): {100*similarity_info['very_similar']/total:.1f}%")
        lines.append(f"Similar (0.6-0.8): {100*similarity_info['similar']/total:.1f}%")
        lines.append("")

    filename = f"{prefix}analysis_report.txt" if prefix else "analysis_report.txt"
    report_path = os.path.join(output_dir, filename)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return report_path


def run_leakage_diagnostics(
    df: pd.DataFrame,
    output_dir: str,
    prefix: str = "",
    seed: int = 42,
    test_fraction: float = 0.10,
    val_fraction: float = 0.10,
    knn_k: int = 5,
    similarity_sample_size: int = 500,
) -> Dict:
    """
    Run complete leakage diagnostics flow and generate artifacts 01-05.

    Returns a dictionary with generated file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    train_idx, _, test_idx = create_stratified_split(
        df=df,
        seed=seed,
        test_fraction=test_fraction,
        val_fraction=val_fraction,
    )

    leakage_info = analyze_compound_leakage(df, train_idx, test_idx)
    baseline_results = evaluate_lookup_baselines(df, train_idx, test_idx, knn_k=knn_k)
    kinase_stats = analyze_kinase_imbalance(df, train_idx)
    compound_stats = analyze_compound_consistency(df, train_idx)
    similarity_info = analyze_chemical_similarity(
        df, train_idx, test_idx, sample_size=similarity_sample_size
    )

    artifacts = {}
    artifacts["01_leakage_analysis"] = plot_leakage_analysis(
        leakage_info, output_dir=output_dir, prefix=prefix
    )
    artifacts["02_baseline_comparison"] = plot_baseline_comparison(
        baseline_results, output_dir=output_dir, prefix=prefix
    )
    artifacts["03_kinase_imbalance"] = plot_kinase_distribution(
        kinase_stats, output_dir=output_dir, prefix=prefix
    )
    artifacts["04_compound_consistency"] = plot_compound_consistency(
        compound_stats, output_dir=output_dir, prefix=prefix
    )
    artifacts["05_similarity_analysis"] = plot_similarity_analysis(
        similarity_info, output_dir=output_dir, prefix=prefix
    )
    artifacts["analysis_report"] = _save_report(
        df,
        leakage_info,
        baseline_results,
        kinase_stats,
        compound_stats,
        similarity_info,
        output_dir=output_dir,
        prefix=prefix,
    )

    return artifacts
