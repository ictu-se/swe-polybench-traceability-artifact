"""Export seven compact publication tables from the frozen analysis outputs."""
import argparse
from pathlib import Path

import pandas as pd

from plot_manuscript import BASE

ROOT = Path(__file__).resolve().parents[1]

METRICS = ['recall_10', 'precision_10', 'code_recall_10', 'test_recall_10']
FEATURE = ['minilm256_pool100', 'coderank256_pool100', 'coderank2048_pool100',
           'navigation/restricted40', 'navigation/repository']
EXTERNAL = ['path_bm25', 'content_bm25', 'dense_minilm', 'hybrid_rrf'] + FEATURE[1:3] + BASE[4:] + FEATURE[3:]


def repair_outcome(row):
    if not row.nonempty_patch:
        return 'No patch'
    if row.f2p_missing + row.p2p_missing:
        return 'Missing tests'
    if row.f2p_passed and not (row.f2p_failed + row.f2p_error) and row.p2p_failed:
        return 'Regression'
    return 'Pass' if row.success else 'Fail'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    root = args.results
    out = root / 'publication_tables'
    out.mkdir(exist_ok=True)

    def save(frame, name):
        frame.to_csv(out / name, index=False, float_format='%.6f')

    coverage = pd.read_csv(root / 'analysis/candidate_coverage.csv')
    targets = pd.read_csv(root / 'analysis/target_availability.csv')
    rows = []
    for language, group in coverage.groupby('language', sort=True):
        target_group = targets[targets.language == language]
        row = {'language': language, 'tasks': len(group),
               'repository_files_median': group.repository_files.median()}
        for role in ['code', 'test']:
            selected = target_group[target_group.kind == role]
            row['existing_' + role] = int(((selected.status != 'added') & selected.exists).sum())
            row['added_' + role] = int((selected.status == 'added').sum())
        rows.append(row)
    save(pd.DataFrame(rows), 'availability.csv')

    base = pd.read_csv(root / 'analysis/summary.csv').set_index('method')
    save(base.loc[BASE, METRICS + ['mrr_10']].reset_index(), 'static_performance.csv')
    panel = pd.read_csv(root / 'analysis/repeat_stability.csv')
    full = pd.read_csv(root / 'extension/analysis/repeat_summary.csv')
    panel['coverage'], full['coverage'] = 'panel', 'full'
    columns = ['method', 'coverage', 'mean_within_task_sd', 'mean_within_task_range']
    save(pd.concat([panel[columns], full[columns]], ignore_index=True), 'repeat_stability.csv')

    extension = pd.read_csv(root / 'extension/analysis/summary.csv').set_index(['cohort', 'method'])
    save(extension.loc['feature'].loc[FEATURE, METRICS].reset_index(), 'feature_extension.csv')
    save(extension.loc['external'].loc[EXTERNAL, METRICS].reset_index(), 'external_performance.csv')
    crossed = pd.read_csv(root / 'extension/cross_window_analysis/summary.csv')
    crossed = crossed.pivot(index=['query_tokens', 'document_tokens'], columns='cohort', values='recall_10')
    save(crossed.reset_index(), 'crossed_windows.csv')

    repair = pd.read_csv(root / 'extension/analysis/repair.csv')
    metrics = pd.read_csv(root / 'extension/analysis/task_metrics.csv')
    metrics = metrics[(metrics.cohort == 'external') & (metrics.seed == 11)]
    pivot = metrics.pivot(index='task_id', columns='method', values='recall_10')
    repair['static_recall'] = repair.task_id.map(pivot['qwen3-coder:30b/content'])
    repair['navigation_recall'] = repair.task_id.map(pivot['navigation/repository'])
    repair['outcome'] = repair.apply(repair_outcome, axis=1)
    save(repair[['task_id', 'static_recall', 'navigation_recall', 'model_calls', 'outcome']], 'repair.csv')
    print('Exported seven tables without re-estimating experimental outcomes.')


if __name__ == '__main__':
    main()
