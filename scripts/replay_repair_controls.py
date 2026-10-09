"""Independently rerun base/gold controls using the archived immutable images.

Outputs go to a separate directory; published control evidence is never changed.
No language-model inference is performed.
"""
import argparse
import json
from pathlib import Path

from repair_probe import grade, preflight
from repository_benchmark import ROOT, load_rows, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    archived = ROOT / 'results/extension/repair'
    if args.output.resolve() == archived.resolve():
        parser.error('Choose a fresh output directory, separate from archived controls.')
    args.output.mkdir(parents=True, exist_ok=True)
    logs = args.output / 'logs'
    logs.mkdir(exist_ok=True)
    tasks = {task['instance_id']: task for task in load_rows(ROOT / 'data/external_tasks.jsonl')}
    protocol = json.loads((ROOT / 'data/external_protocol.json').read_text())
    for task_id in protocol['repair_panel_ids']:
        destination = args.output / (task_id + '_preflight.json')
        if destination.exists():
            print('Retaining existing replay:', task_id, flush=True)
            continue
        reference = json.loads((archived / (task_id + '_preflight.json')).read_text())
        task = dict(tasks[task_id], docker_image=reference['image_digest'])
        record = preflight(task, logs)
        mapping = reference['expected_test_ids']
        for phase, result in record['phases'].items():
            result['grade'] = grade(result['statuses'], task, base=phase == 'base', expected_ids=mapping)
        record['expected_test_ids'] = mapping
        record['grading_version'] = 'frozen-full-test-ids-v3'
        record['environment_valid'] = (set(record['phases']) == {'base', 'gold'} and
                                       all(result['grade']['passed'] for result in record['phases'].values()))
        record['replayed_image_digest'] = reference['image_digest']
        save_json(destination, record)
        print(task_id, 'eligible:', record['environment_valid'], flush=True)


if __name__ == '__main__':
    main()
