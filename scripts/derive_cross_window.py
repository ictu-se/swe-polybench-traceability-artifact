"""Recombine cached CodeRank query/document vectors in a 2x2 window analysis.

This post-hoc analysis makes no model calls and preserves the original diagonal
conditions. The exploratory decision and timing are recorded in its protocol.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np

from repository_benchmark import ROOT, embedding_text, load_rows, ranking, save_json
from semantic_rerank import CODE_REVISION, QUERY_PREFIX

LIMITS = (256, 2048)


def vectors(database, texts, namespace):
    values = []
    for text in texts:
        key = hashlib.sha256((namespace + '\0' + text).encode()).hexdigest()
        row = database.execute('SELECT vector FROM embeddings WHERE key=?', (key,)).fetchone()
        if row is None:
            raise ValueError('Missing cached vector: ' + key)
        vector = np.frombuffer(row[0], dtype=np.float32)
        if vector.shape != (768,) or not np.isfinite(vector).all():
            raise ValueError('Invalid cached vector: ' + key)
        values.append(vector)
    return np.stack(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--embedding-cache', type=Path, required=True)
    parser.add_argument('--tasks', type=Path, default=ROOT / 'data/tasks.jsonl')
    parser.add_argument('--semantic', type=Path, default=ROOT / 'results/extension/semantic')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/extension/cross_window')
    args = parser.parse_args()
    database = sqlite3.connect(args.embedding_cache.resolve().as_uri() + '?mode=ro', uri=True, timeout=60)
    count = 0
    for task in load_rows(args.tasks):
        task_id = task['instance_id']
        semantic = json.loads((args.semantic / (task_id + '.json')).read_text())
        with gzip.open(args.cache / 'corpus' / (task_id + '.json.gz'), 'rt') as stream:
            corpus = json.load(stream)
        assert corpus['archive_sha256'] == semantic['archive_sha256']
        files = {file['path']: file for file in corpus['files']}
        pool = semantic['pool']
        documents = [embedding_text(files[path]) for path in pool]
        assert [hashlib.sha256(text.encode()).hexdigest() for text in documents] == semantic['document_sha256']
        query = QUERY_PREFIX + task['problem_statement']
        document_vectors = {limit: vectors(database, documents, f'{CODE_REVISION}:{limit}') for limit in LIMITS}
        query_vectors = {limit: vectors(database, [query], f'{CODE_REVISION}:{limit}:query')[0] for limit in LIMITS}
        result = {'task_id': task_id, 'repo': task['repo'], 'pool': pool,
                  'archive_sha256': semantic['archive_sha256'], 'code_revision': CODE_REVISION,
                  'document_sha256': semantic['document_sha256'],
                  'query_sha256': hashlib.sha256(query.encode()).hexdigest(),
                  'scores': {}, 'rankings': {}, 'original_score_max_abs_error': {}}
        for query_limit in LIMITS:
            for document_limit in LIMITS:
                name = f'q{query_limit}_d{document_limit}'
                scores = document_vectors[document_limit] @ query_vectors[query_limit]
                prediction = [pool[index] for index in ranking(scores, pool)]
                if query_limit == document_limit:
                    original = f'coderank{query_limit}_pool100'
                    error = float(np.max(np.abs(scores - np.asarray(semantic['scores'][original]))))
                    assert error <= 1e-6, (task_id, original, error)
                    assert prediction == semantic['rankings'][original], (task_id, original, 'ranking differs')
                    result['original_score_max_abs_error'][name] = error
                    result['scores'][name] = semantic['scores'][original]
                    result['rankings'][name] = semantic['rankings'][original]
                else:
                    result['scores'][name] = [float(value) for value in scores]
                    result['rankings'][name] = prediction
        save_json(args.output / (task_id + '.json'), result)
        count += 1
    database.close()
    print('Recombined cached vectors for', count, 'tasks; original conditions verified unchanged.')


if __name__ == '__main__':
    main()
