"""Archive only checksummed verification output needed to reproduce test parsing.

Host logs, Docker build output and unrelated files are not collected. Original
and corrected reference-control records may share the same output archive.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

from repair_probe import parse_tests
from repository_benchmark import ROOT, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--logs', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    directory = ROOT / 'results/extension/repair'
    agent_count = len(list(directory.glob('*_agent.json')))
    if agent_count != 12 and not args.allow_partial:
        raise SystemExit('Complete the twelve-case repair panel before final output collection.')
    sources = {}
    for path in args.logs.glob('*.txt'):
        content = path.read_text().encode('utf-8')
        sources[hashlib.sha256(content).hexdigest()] = content
    output = directory / 'verification_outputs'
    output.mkdir(exist_ok=True)
    entries = []
    records = (list(directory.glob('*_preflight.json')) +
               list((directory / 'control_initial').glob('*_preflight.json')) +
               list(directory.glob('*_agent.json')))
    for path in sorted(records):
        record = json.loads(path.read_text())
        phases = dict(record.get('phases', {}))
        if 'evaluation' in record:
            phases['model'] = record['evaluation']
        for phase, result in phases.items():
            digest = result['output_sha256']
            if digest not in sources:
                raise ValueError('Missing verification output for ' + path.name + '/' + phase)
            content = sources[digest]
            assert parse_tests(content.decode('utf-8')) == result['statuses']
            archive = output / (digest + '.txt.gz')
            archive.write_bytes(gzip.compress(content, mtime=0))
            entries.append({'record': path.relative_to(ROOT).as_posix(), 'phase': phase,
                            'archive': archive.relative_to(ROOT).as_posix(),
                            'output_sha256': digest, 'output_bytes': len(content),
                            'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest()})
    save_json(directory / 'verification_manifest.json',
              {'complete': agent_count == 12, 'agent_records': agent_count, 'outputs': entries})
    print('Verified and archived', len(entries), 'record/phase references across',
          len({entry['output_sha256'] for entry in entries}), 'unique verification outputs.')


if __name__ == '__main__':
    main()
