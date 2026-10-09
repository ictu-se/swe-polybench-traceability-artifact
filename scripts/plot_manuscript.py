"""Render the two manuscript curves at the journal's physical column width.

Reads archived measurements only; no inference or statistical re-estimation.
Color denotes method family, while markers and line styles distinguish variants.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

WIDTH_IN = 119 / 25.4
STYLES = {
    'path_bm25': ('Path BM25', '#6b7280', 'o', ':'),
    'content_bm25': ('Content BM25', '#6b7280', 's', '--'),
    'dense_minilm': ('MiniLM', '#6b7280', '^', '-.'),
    'hybrid_rrf': ('Hybrid RRF', '#20252b', 'D', '-'),
    'qwen3-coder:30b/paths': ('Qwen: paths', '#245b8c', 'o', '--'),
    'qwen3-coder:30b/content': ('Qwen: excerpts', '#245b8c', 's', '-'),
    'devstral-small-2:24b/paths': ('Devstral: paths', '#287457', '^', '--'),
    'devstral-small-2:24b/content': ('Devstral: excerpts', '#287457', 'D', '-'),
    'minilm256_pool100': ('MiniLM: 256', '#6b7280', '^', '-.'),
    'coderank256_pool100': ('CodeRank: 256', '#7953a2', 's', '--'),
    'coderank2048_pool100': ('CodeRank: 2048', '#7953a2', 'D', ':'),
    'navigation/restricted40': ('Nav: restricted', '#a85725', 'v', '--'),
    'navigation/repository': ('Nav: repository', '#a85725', 'P', '-'),
}
BASE = list(STYLES)[:8]
EXTENSION = ['hybrid_rrf', 'minilm256_pool100', 'coderank256_pool100',
             'coderank2048_pool100', 'qwen3-coder:30b/content',
             'navigation/restricted40', 'navigation/repository']
BUDGETS = [1, 3, 5, 10]


def draw(ax, values, method):
    label, color, marker, style = STYLES[method]
    ax.plot(BUDGETS, values, color=color, marker=marker, linestyle=style,
            label=label, linewidth=1.25, markersize=4)


def finish(fig, axes, path):
    for ax, tag in zip(axes, ['(a)', '(b)']):
        ax.set(xticks=BUDGETS, ylim=(0, 1), xlabel='Output budget k')
        ax.text(.04, .95, tag, transform=ax.transAxes, va='top')
        ax.grid(axis='y', alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False,
               columnspacing=.8, handlelength=2)
    fig.subplots_adjust(left=.15, right=.98, top=.96, bottom=.35, wspace=.45)
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    plt.rcParams.update({'font.size': 10, 'axes.labelsize': 10,
                         'xtick.labelsize': 10, 'ytick.labelsize': 10,
                         'legend.fontsize': 10, 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False})
    out = args.results / 'figures'
    out.mkdir(exist_ok=True)
    frame = pd.read_csv(args.results / 'analysis/summary.csv').set_index('method')
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 4.45))
    for method in BASE:
        for ax, metric in zip(axes, ['recall', 'precision']):
            draw(ax, [frame.loc[method, f'{metric}_{k}'] for k in BUDGETS], method)
    axes[0].set_ylabel('Macro recall')
    axes[1].set_ylabel('Fixed-budget precision')
    finish(fig, axes, out / 'budget_curves.pdf')

    frame = pd.read_csv(args.results / 'extension/analysis/task_metrics.csv')
    out = args.results / 'extension/figures'
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 4.45), sharey=True)
    for ax, cohort in zip(axes, ['feature', 'external']):
        group = frame[(frame.cohort == cohort) & (frame.seed == 11)]
        for method in EXTENSION:
            rows = group[group.method == method]
            expected = 184 if cohort == 'feature' else 24
            if len(rows) != expected or rows.task_id.nunique() != expected:
                raise ValueError(f'Incomplete or duplicate primary results: {cohort}, {method}')
            draw(ax, [rows[f'recall_{k}'].mean() for k in BUDGETS], method)
    axes[0].set_ylabel('Macro recall')
    finish(fig, axes, out / 'extension_budget_curves.pdf')
    print('Rendered two data-derived figures at 119 mm width with 10 pt labels.')


if __name__ == '__main__':
    main()
