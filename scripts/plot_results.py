"""Create empirical figures from the exported analysis tables."""
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from repository_benchmark import ROOT

LABELS = {"path_bm25": "Path BM25", "content_bm25": "Content BM25", "dense_minilm": "Dense MiniLM",
          "hybrid_rrf": "Hybrid RRF", "qwen3-coder:30b/paths": "Qwen3-Coder: paths",
          "qwen3-coder:30b/content": "Qwen3-Coder: content",
          "devstral-small-2:24b/paths": "Devstral: paths", "devstral-small-2:24b/content": "Devstral: content"}
STYLES = {"path_bm25": ("#555555", "o", "-"), "content_bm25": ("#0072B2", "s", "-"),
          "dense_minilm": ("#E69F00", "^", "-"), "hybrid_rrf": ("#009E73", "D", "-"),
          "qwen3-coder:30b/paths": ("#D55E00", "o", "--"),
          "qwen3-coder:30b/content": ("#D55E00", "^", "-"),
          "devstral-small-2:24b/paths": ("#CC79A7", "s", "--"),
          "devstral-small-2:24b/content": ("#CC79A7", "D", "-")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    # Match the official smallextended text width (11.9 cm), so figure labels
    # retain their stated point size in the manuscript.
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    source = args.results / "analysis"
    out = args.results / "figures"
    out.mkdir(exist_ok=True)
    summary = pd.read_csv(source / "summary.csv").set_index("method").reindex(LABELS).reset_index()
    coverage = pd.read_csv(source / "candidate_coverage.csv")
    targets = pd.read_csv(source / "target_availability.csv")
    frame = pd.read_csv(source / "task_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(4.68, 3.7))
    for row in summary.to_dict("records"):
        method = row["method"]
        color, marker, linestyle = STYLES[method]
        for ax, metric in zip(axes, ["recall", "precision"]):
            ax.plot([1, 3, 5, 10], [row[f"{metric}_{k}"] for k in [1, 3, 5, 10]],
                    marker=marker, color=color, linestyle=linestyle, markersize=3.5,
                    linewidth=1.1, label=LABELS.get(method, method))
    for ax, label in zip(axes, ["Macro exact-path recall", "Macro fixed-budget precision"]):
        ax.set(xlabel="Prediction budget k", ylabel=label, xticks=[1, 3, 5, 10], ylim=(0, 1))
        ax.grid(alpha=.2)
    for ax,label in zip(axes,["(a)","(b)"]):
        ax.text(.04,.96,label,transform=ax.transAxes,ha="left",va="top")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8)
    fig.tight_layout(rect=(0, .27, 1, 1))
    fig.savefig(out / "budget_curves.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(4.68, 3.1))
    groups = list(coverage.groupby("language"))
    axes[0].boxplot([g.repository_files for _, g in groups], tick_labels=[name for name, _ in groups], showfliers=True)
    axes[0].set(yscale="log", ylabel="Files at the base commit")
    axes[0].tick_params(axis="x", rotation=30)
    by_language = targets.groupby(["language", "status"]).size().unstack(fill_value=0)
    by_language.reindex(columns=["modified", "added", "deleted", "renamed"], fill_value=0).plot.bar(
        stacked=True, ax=axes[1], color=["#33658a", "#f6ae2d", "#758e4f", "#8f5e90"])
    for container,hatch in zip(axes[1].containers,["", "//", "..", "xx"]):
        for bar in container:
            bar.set_hatch(hatch)
    for ax,label in zip(axes,["(a)","(b)"]):
        ax.text(.04,.96,label,transform=ax.transAxes,ha="left",va="top")
    axes[1].set(ylabel="Patch-derived target occurrences", xlabel="")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].legend(loc="upper left", bbox_to_anchor=(0, 1.23), ncol=2,
                   fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "target_availability.pdf", bbox_inches="tight")
    plt.close(fig)

    contrasts = pd.read_csv(source / "paired_contrasts.csv")
    fig, ax = plt.subplots(figsize=(4.68, 3.1))
    y = np.arange(len(contrasts))
    ax.errorbar(contrasts.difference, y,
                xerr=[contrasts.difference-contrasts.cluster_low, contrasts.cluster_high-contrasts.difference],
                fmt="o", capsize=4, color="#33658a")
    ax.axvline(0, color="black", linewidth=.8, linestyle="--")
    contrast_labels = []
    for a, b in zip(contrasts.a, contrasts.b):
        left, right = LABELS.get(a, a), LABELS.get(b, b)
        if a.split("/")[0] == b.split("/")[0] and "/" in a:
            right = b.split("/")[1]
        contrast_labels.append(left + " minus " + right)
    ax.set(yticks=y, yticklabels=contrast_labels,
           xlabel="Paired difference in R@10\n(95% repository bootstrap interval)")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "paired_differences.pdf", bbox_inches="tight")
    plt.close(fig)

    primary = frame[frame.seed == 11]
    pivot = primary.groupby(["method", "language"]).recall_10.mean().unstack().reindex(LABELS)
    fig, ax = plt.subplots(figsize=(4.68, 3.2))
    im = ax.imshow(pivot.values, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    ax.set(xticks=range(len(pivot.columns)), xticklabels=pivot.columns,
           yticks=range(len(pivot)), yticklabels=[LABELS.get(m,m) for m in pivot.index])
    for i in range(len(pivot)):
        for j in range(len(pivot.columns)):
            value = pivot.iloc[i,j]
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", color="white" if value>.5 else "black")
    fig.colorbar(im, ax=ax, label="Macro exact-path recall at 10")
    fig.tight_layout()
    fig.savefig(out / "language_results.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
