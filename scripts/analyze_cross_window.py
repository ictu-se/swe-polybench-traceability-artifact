"""Audit and summarize the exploratory CodeRank query/document window analysis."""
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from analyze import gold, interval, metrics
from analyze_extension import contrast
from repository_benchmark import ROOT, load_rows, ranking, save_json
from semantic_rerank import CODE_REVISION, QUERY_PREFIX

CONDITIONS = ('q256_d256', 'q256_d2048', 'q2048_d256', 'q2048_d2048')


def population(cohort):
    extension = ROOT / 'results/extension'
    base = extension if cohort == 'feature' else extension / 'external'
    task_file = ROOT / 'data' / ('tasks.jsonl' if cohort == 'feature' else 'external_tasks.jsonl')
    tasks = load_rows(task_file)
    retrieval = ROOT / 'results/retrieval' if cohort == 'feature' else base / 'retrieval'
    assert {path.stem for path in (base / 'cross_window').glob('*.json')} == {task['instance_id'] for task in tasks}
    rows = []
    for task in tasks:
        task_id = task['instance_id']
        result = json.loads((base / 'cross_window' / (task_id + '.json')).read_text())
        original = json.loads((base / 'semantic' / (task_id + '.json')).read_text())
        retrieved = json.loads((retrieval / (task_id + '.json')).read_text())
        assert result['task_id'] == task_id and result['repo'] == task['repo']
        assert result['code_revision'] == CODE_REVISION
        assert result['pool'] == original['pool'] == retrieved['rankings']['hybrid_rrf'][:100]
        assert result['archive_sha256'] == original['archive_sha256'] == retrieved['archive_sha256']
        assert result['document_sha256'] == original['document_sha256']
        assert result['query_sha256'] == hashlib.sha256((QUERY_PREFIX + task['problem_statement']).encode()).hexdigest()
        assert set(result['rankings']) == set(CONDITIONS)
        for limit in (256, 2048):
            key = f'q{limit}_d{limit}'
            assert result['rankings'][key] == original['rankings'][f'coderank{limit}_pool100']
            assert result['scores'][key] == original['scores'][f'coderank{limit}_pool100']
            assert result['original_score_max_abs_error'][key] <= 1e-6
        targets = gold(task, set(retrieved['universe']))
        for condition in CONDITIONS:
            scores = np.asarray(result['scores'][condition])
            assert len(scores) == len(result['pool']) and np.isfinite(scores).all()
            assert result['rankings'][condition] == [result['pool'][i] for i in ranking(scores, result['pool'])]
            query_limit, document_limit = (int(part[1:]) for part in condition.split('_'))
            rows.append({'cohort': cohort, 'task_id': task_id, 'repo': task['repo'],
                         'method': condition, 'seed': 11, 'query_tokens': query_limit,
                         'document_tokens': document_limit,
                         **metrics(result['rankings'][condition], targets)})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohorts', nargs='+', choices=['feature', 'external'], default=['feature', 'external'])
    args = parser.parse_args()
    protocol = json.loads((ROOT / 'data/cross_window_protocol.json').read_text())
    assert hashlib.sha256((ROOT / 'scripts/derive_cross_window.py').read_bytes()).hexdigest() == protocol['derivation_script_sha256']
    assert protocol['new_model_calls'] == 0
    summaries, contrasts, frames = [], [], []
    for cohort in args.cohorts:
        frame = population(cohort)
        frames.append(frame)
        for (method, query_limit, document_limit), group in frame.groupby(['method', 'query_tokens', 'document_tokens']):
            low, high = interval(group.recall_10, group.repo, seed=20261009)
            summaries.append({'cohort': cohort, 'method': method, 'query_tokens': query_limit,
                              'document_tokens': document_limit, 'tasks': len(group),
                              'recall_10': group.recall_10.mean(), 'precision_10': group.precision_10.mean(),
                              'cluster_low': low, 'cluster_high': high})
        for left, right in [('q256_d2048', 'q256_d256'), ('q2048_d2048', 'q2048_d256'),
                            ('q2048_d256', 'q256_d256'), ('q2048_d2048', 'q256_d2048')]:
            contrasts.append(contrast(frame, left, right, cohort))
        pivot = frame.pivot(index='task_id', columns='method', values='recall_10')
        interaction = (pivot.q2048_d2048 - pivot.q2048_d256) - (pivot.q256_d2048 - pivot.q256_d256)
        repositories = frame.drop_duplicates('task_id').set_index('task_id').loc[pivot.index, 'repo']
        low, high = interval(interaction, repositories, seed=20261009)
        task_low, task_high = interval(interaction, seed=20261009)
        contrasts.append({'cohort': cohort, 'seed_mode': 'primary', 'a': 'interaction',
                          'b': 'zero', 'n': len(pivot), 'difference': interaction.mean(),
                          'cluster_low': low, 'cluster_high': high, 'task_low': task_low,
                          'task_high': task_high, 'wins': int((interaction > 1e-12).sum()),
                          'ties': int((abs(interaction) <= 1e-12).sum()),
                          'losses': int((interaction < -1e-12).sum())})
    output = ROOT / 'results/extension/cross_window_analysis'
    output.mkdir(exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(output / 'task_metrics.csv', index=False)
    pd.DataFrame(summaries).to_csv(output / 'summary.csv', index=False)
    pd.DataFrame(contrasts).to_csv(output / 'contrasts.csv', index=False)
    save_json(output / 'audit.json', {'passed': True, 'complete': set(args.cohorts) == {'feature', 'external'},
                                    'cohorts': args.cohorts, 'task_condition_records': sum(len(frame) for frame in frames),
                                    'original_conditions_unchanged': True})
    print('Audited and summarized cross-window analysis for', ', '.join(args.cohorts))


if __name__ == '__main__':
    main()
