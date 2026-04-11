#!/usr/bin/env python3
"""Generate publication-quality comparison plots: DT-Kinase vs DrugBAN vs GraphBAN."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import os

# ── Output directory ──────────────────────────────────────────────────
OUT_DIR = "results/benchmark_non_human_8M_v7_adapter1L_selfattn_lr5x"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Colors ────────────────────────────────────────────────────────────
COLORS = {
    "DT-Kinase": "#2563EB",   # Blue
    "DrugBAN":   "#DC2626",   # Red
    "GraphBAN":  "#059669",   # Green
}
EDGE_COLORS = {
    "DT-Kinase": "#1D4ED8",
    "DrugBAN":   "#B91C1C",
    "GraphBAN":  "#047857",
}

# ── Data ──────────────────────────────────────────────────────────────
# Test set results (fair protocol)
test_metrics = {
    "DT-Kinase": {
        "MCC": (0.5118, 0.0160),
        "AUROC": (0.8144, 0.0079),
        "F1": (0.7877, 0.0109),
        "Accuracy": (0.7521, 0.0098),
        "Precision": (0.7251, 0.0337),
        "Recall": (0.8674, 0.0609),
    },
    "DrugBAN": {
        "MCC": (0.5048, 0.0189),
        "AUROC": (0.8236, 0.0117),
        "F1": (0.7446, 0.0218),
        "Accuracy": (0.7431, 0.0018),
        "Precision": (0.6894, 0.0285),
        "Recall": (0.8192, 0.0859),
    },
    "GraphBAN": {
        "MCC": (0.5229, 0.0229),
        "AUROC": (0.8294, 0.0117),
        "F1": (0.7595, 0.0099),
        "Accuracy": (0.7511, 0.0164),
        "Precision": (0.6873, 0.0272),
        "Recall": (0.8514, 0.0365),
    },
}

# Train/val results (for generalization gap)
train_metrics = {
    "DT-Kinase": {"MCC": (0.5537, 0.0165)},
    "GraphBAN":  {"MCC": (0.7025, 0.0506)},
}

# Per-seed MCC values
per_seed_mcc = {
    "DT-Kinase": None,  # Don't have individual seeds, use mean±std
    "DrugBAN": [0.4786, 0.5136, 0.5223],
    "GraphBAN": [0.5651, 0.5295, 0.5061, 0.5065, 0.5070],
}

MODELS = ["DT-Kinase", "DrugBAN", "GraphBAN"]


# ══════════════════════════════════════════════════════════════════════
# Plot 1: Test Metrics Bar Chart (Main Comparison)
# ══════════════════════════════════════════════════════════════════════
def plot_test_metrics():
    metrics = ["MCC", "AUROC", "F1", "Accuracy", "Precision", "Recall"]
    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, model in enumerate(MODELS):
        means = [test_metrics[model][m][0] for m in metrics]
        stds = [test_metrics[model][m][1] for m in metrics]
        bars = ax.bar(
            x + (i - 1) * width, means, width,
            yerr=stds, capsize=4,
            label=model,
            color=COLORS[model],
            edgecolor=EDGE_COLORS[model],
            linewidth=0.8,
            alpha=0.85,
            error_kw={"linewidth": 1.2, "capthick": 1.2},
        )
        # Value labels on top
        for bar, mean, std in zip(bars, means, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.008,
                f"{mean:.3f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                color=EDGE_COLORS[model],
            )

    ax.set_ylabel("Score")
    ax.set_title("Test Set Performance — non_human Kinase (Scaffold Split)", fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontweight="bold")
    ax.set_ylim(0.35, 1.0)
    ax.legend(loc="upper right", framealpha=0.9, edgecolor="#ccc")
    ax.axhline(y=0.5, color="#999", linestyle="--", linewidth=0.8, alpha=0.5, label="_nolegend_")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    path = os.path.join(OUT_DIR, "comparison_test_metrics.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════
# Plot 2: MCC Focus — Bar + Error (Zoomed)
# ══════════════════════════════════════════════════════════════════════
def plot_mcc_focus():
    fig, ax = plt.subplots(figsize=(8, 6))

    means = [test_metrics[m]["MCC"][0] for m in MODELS]
    stds = [test_metrics[m]["MCC"][1] for m in MODELS]
    colors_list = [COLORS[m] for m in MODELS]
    edge_list = [EDGE_COLORS[m] for m in MODELS]

    bars = ax.bar(
        MODELS, means, width=0.55,
        yerr=stds, capsize=8,
        color=colors_list, edgecolor=edge_list,
        linewidth=1.5, alpha=0.85,
        error_kw={"linewidth": 2, "capthick": 2},
    )

    for bar, mean, std, ec in zip(bars, means, stds, edge_list):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.003,
            f"{mean:.4f} ± {std:.4f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
            color=ec,
        )

    ax.set_ylabel("Matthews Correlation Coefficient (MCC)", fontweight="bold")
    ax.set_title("Test MCC — Model Comparison\nnon_human Kinase, Scaffold-Disjoint Split", fontweight="bold", pad=12)
    ax.set_ylim(0.42, 0.60)
    ax.axhline(y=0.5, color="#E5E7EB", linestyle="--", linewidth=1.5, zorder=0)
    ax.text(2.55, 0.501, "MCC = 0.5", fontsize=9, color="#999", ha="right")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    # Annotations
    ax.annotate(
        "Δ 0.011",
        xy=(1.5, max(means[0], means[2])),
        xytext=(1.5, 0.56),
        ha="center", fontsize=10, color="#555",
        arrowprops=dict(arrowstyle="-", color="#999", lw=0.8),
    )

    path = os.path.join(OUT_DIR, "comparison_mcc_focus.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════
# Plot 3: Generalization Gap (Train vs Test MCC)
# ══════════════════════════════════════════════════════════════════════
def plot_generalization_gap():
    fig, ax = plt.subplots(figsize=(9, 6))

    models_gap = ["DT-Kinase", "GraphBAN"]
    train_vals = [train_metrics[m]["MCC"][0] for m in models_gap]
    train_stds = [train_metrics[m]["MCC"][1] for m in models_gap]
    test_vals = [test_metrics[m]["MCC"][0] for m in models_gap]
    test_stds = [test_metrics[m]["MCC"][1] for m in models_gap]

    x = np.arange(len(models_gap))
    width = 0.3

    bars_train = ax.bar(
        x - width / 2, train_vals, width,
        yerr=train_stds, capsize=6,
        label="Train (val eval)", color=["#93C5FD", "#6EE7B7"],
        edgecolor=[EDGE_COLORS[m] for m in models_gap],
        linewidth=1.2, alpha=0.7,
        error_kw={"linewidth": 1.5, "capthick": 1.5},
    )
    bars_test = ax.bar(
        x + width / 2, test_vals, width,
        yerr=test_stds, capsize=6,
        label="Test", color=[COLORS[m] for m in models_gap],
        edgecolor=[EDGE_COLORS[m] for m in models_gap],
        linewidth=1.2, alpha=0.85,
        error_kw={"linewidth": 1.5, "capthick": 1.5},
    )

    # Gap annotations
    gaps = [t - te for t, te in zip(train_vals, test_vals)]
    for i, (tv, tev, gap) in enumerate(zip(train_vals, test_vals, gaps)):
        mid = (tv + tev) / 2
        ax.annotate(
            "",
            xy=(i + width / 2, tev + test_stds[i] + 0.005),
            xytext=(i - width / 2, tv - train_stds[i] - 0.005),
            arrowprops=dict(arrowstyle="<->", color="#DC2626", lw=2),
        )
        ax.text(
            i + 0.22, mid, f"Δ {gap:.3f}",
            fontsize=12, fontweight="bold", color="#DC2626",
            ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#DC2626", alpha=0.9),
        )

    # Value labels
    for bars, vals, stds in [(bars_train, train_vals, train_stds), (bars_test, test_vals, test_stds)]:
        for bar, v, s in zip(bars, vals, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + s + 0.008,
                f"{v:.4f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )

    ax.set_ylabel("MCC", fontweight="bold")
    ax.set_title("Generalization Gap — Train vs Test MCC\n(lower gap = better generalization)", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models_gap, fontweight="bold", fontsize=13)
    ax.set_ylim(0.35, 0.80)
    ax.legend(loc="upper left", framealpha=0.9, edgecolor="#ccc")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    path = os.path.join(OUT_DIR, "comparison_generalization_gap.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════
# Plot 4: Radar Chart (Multi-Metric Profile)
# ══════════════════════════════════════════════════════════════════════
def plot_radar():
    metrics = ["MCC", "AUROC", "F1", "Accuracy", "Precision", "Recall"]
    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for model in MODELS:
        values = [test_metrics[model][m][0] for m in metrics]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2.2, label=model, color=COLORS[model], markersize=7)
        ax.fill(angles, values, alpha=0.08, color=COLORS[model])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontweight="bold", fontsize=12)
    ax.set_ylim(0.4, 1.0)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_yticklabels(["0.5", "0.6", "0.7", "0.8", "0.9"], fontsize=9, color="#666")
    ax.set_title("Multi-Metric Profile — Test Set\nnon_human Kinase, Scaffold Split", fontweight="bold", pad=25, fontsize=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), framealpha=0.9, edgecolor="#ccc")
    ax.grid(True, alpha=0.3)

    path = os.path.join(OUT_DIR, "comparison_radar.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════
# Plot 5: Stability (Std comparison)
# ══════════════════════════════════════════════════════════════════════
def plot_stability():
    metrics = ["MCC", "AUROC", "F1", "Accuracy"]
    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for i, model in enumerate(MODELS):
        stds = [test_metrics[model][m][1] for m in metrics]
        bars = ax.bar(
            x + (i - 1) * width, stds, width,
            label=model,
            color=COLORS[model],
            edgecolor=EDGE_COLORS[model],
            linewidth=0.8, alpha=0.85,
        )
        for bar, s in zip(bars, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.0005,
                f"{s:.4f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                color=EDGE_COLORS[model],
            )

    ax.set_ylabel("Standard Deviation (lower = more stable)", fontweight="bold")
    ax.set_title("Cross-Seed Stability — Test Set\n(lower standard deviation = more reproducible)", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontweight="bold")
    ax.set_ylim(0, 0.035)
    ax.legend(loc="upper right", framealpha=0.9, edgecolor="#ccc")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    path = os.path.join(OUT_DIR, "comparison_stability.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════
# Plot 6: Efficiency — MCC vs Embedding Size
# ══════════════════════════════════════════════════════════════════════
def plot_efficiency():
    fig, ax = plt.subplots(figsize=(9, 6))

    # (embedding_params_millions, test_mcc, mcc_std)
    data = {
        "DT-Kinase\n(ESM-2 8M)": (8, 0.5118, 0.0160),
        "DrugBAN\n(GCN, no PLM)": (0.5, 0.5048, 0.0189),  # ~0.5M learned params
        "GraphBAN\n(ESM-1b 650M)": (650, 0.5229, 0.0229),
    }

    for (label, (x_val, mcc, std)), model_key in zip(data.items(), MODELS):
        ax.errorbar(
            x_val, mcc, yerr=std,
            fmt="o", markersize=14,
            color=COLORS[model_key],
            markeredgecolor=EDGE_COLORS[model_key],
            markeredgewidth=2,
            elinewidth=2, capsize=8, capthick=2,
            label=label, zorder=5,
        )
        ax.annotate(
            f"MCC={mcc:.4f}",
            xy=(x_val, mcc),
            xytext=(15, -20),
            textcoords="offset points",
            fontsize=10, fontweight="bold",
            color=EDGE_COLORS[model_key],
        )

    ax.set_xscale("log")
    ax.set_xlabel("Protein Embedding Model Size (M parameters, log scale)", fontweight="bold")
    ax.set_ylabel("Test MCC", fontweight="bold")
    ax.set_title("Efficiency: MCC vs Embedding Model Size\n(higher MCC with smaller model = more efficient)", fontweight="bold", pad=12)
    ax.set_ylim(0.45, 0.58)
    ax.set_xlim(0.2, 1500)
    ax.axhline(y=0.5, color="#E5E7EB", linestyle="--", linewidth=1.5, zorder=0)
    ax.legend(loc="lower right", framealpha=0.9, edgecolor="#ccc", fontsize=11)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # Arrow showing 81x efficiency
    ax.annotate(
        "81× smaller model\nΔMCC = only 0.011",
        xy=(8, 0.5118),
        xytext=(80, 0.555),
        fontsize=10, color="#555", ha="center",
        arrowprops=dict(arrowstyle="->", color="#999", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FEF3C7", edgecolor="#F59E0B", alpha=0.9),
    )

    path = os.path.join(OUT_DIR, "comparison_efficiency.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating comparison plots...")
    paths = [
        plot_test_metrics(),
        plot_mcc_focus(),
        plot_generalization_gap(),
        plot_radar(),
        plot_stability(),
        plot_efficiency(),
    ]
    print(f"\nDone! {len(paths)} plots saved to {OUT_DIR}/")
