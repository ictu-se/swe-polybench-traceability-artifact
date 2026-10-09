"""Inspect model-visible input state in fresh containers, without model inference."""
import argparse
import json
from pathlib import Path

from repair_probe import Environment, command
from repository_benchmark import ROOT, load_rows, save_json


CHECKS = {
    'head': 'git rev-parse HEAD',
    'refs': 'git for-each-ref --format="%(refname) %(objectname)"',
    'reflog': 'git reflog --all --format="%H"',
    'nonancestor_reachable_commits': 'git rev-list --all --not HEAD',
    'unreachable_objects': 'git fsck --full --no-reflogs --unreachable',
    'status': 'git status --porcelain',
    'reference_files': ('for p in /eval.sh /tmp/patch.diff /tmp/test_patch.diff /tmp/gold.patch; '
                        'do if test -e "$p"; then echo "$p"; fi; done'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/extension/repair/input_audit.json')
    args = parser.parse_args()
    tasks = {task['instance_id']: task for task in load_rows(ROOT / 'data/external_tasks.jsonl')}
    protocol = json.loads((ROOT / 'data/external_protocol.json').read_text())
    records = []
    for task_id in protocol['repair_panel_ids']:
        reference = json.loads((ROOT / 'results/extension/repair' / (task_id + '_preflight.json')).read_text())
        env = Environment(tasks[task_id], reference['image_digest'])
        try:
            record = {'task_id': task_id, 'base_commit': tasks[task_id]['base_commit'],
                      'image_digest': reference['image_digest']}
            settings = json.loads(command(['docker', 'inspect', env.name]).stdout)[0]
            record['network_mode'] = settings['HostConfig']['NetworkMode']
            record['bind_mounts'] = [mount['Destination'] for mount in settings['Mounts'] if mount['Type'] == 'bind']
            for key, shell_command in CHECKS.items():
                result = env.shell(shell_command, timeout=120)
                record[key] = result.stdout.strip()
                record[key + '_returncode'] = result.returncode
            record['input_checks_passed'] = (
                record['head'] == record['base_commit'] and
                not record['refs'] and not record['nonancestor_reachable_commits'] and
                not record['unreachable_objects'] and not record['reference_files'] and
                not record['bind_mounts'] and record['network_mode'] == 'none' and
                all(record[key + '_returncode'] == 0 for key in CHECKS)
            )
            records.append(record)
            save_json(args.output, records)
            print(task_id, 'input checks:', record['input_checks_passed'],
                  'unreachable objects:', len(record['unreachable_objects'].splitlines()), flush=True)
        finally:
            env.close()
    if not all(record['input_checks_passed'] for record in records):
        raise SystemExit('Input-state anomalies retained; inspect the audit before model execution.')


if __name__ == '__main__':
    main()
