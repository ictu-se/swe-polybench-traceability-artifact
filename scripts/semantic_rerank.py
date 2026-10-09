"""Compare code-specialized and general encoders over identical frozen top-100 pools."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import numpy as np

from repository_benchmark import ROOT, EMBEDDING_MODEL, EMBEDDING_REVISION, embedding_text, load_rows, ranking, save_json

CODE_MODEL='nomic-ai/CodeRankEmbed'
CODE_REVISION='3c4b60807d71f79b43f3c4363786d9493691f8b1'
QUERY_PREFIX='Represent this query for searching relevant code: '


def cached_vectors(texts,model,database,namespace,batch_size):
    keys=[hashlib.sha256((namespace+'\0'+s).encode()).hexdigest() for s in texts]
    found={};missing={}
    for k,s in zip(keys,texts):
        row=database.execute('SELECT vector FROM embeddings WHERE key=?',(k,)).fetchone()
        if row:found[k]=np.frombuffer(row[0],dtype=np.float32)
        else:missing[k]=s
    names=list(missing)
    # Commit bounded batches, so interruption never discards a complete repository's work.
    for start in range(0,len(names),batch_size):
        batch=names[start:start+batch_size]
        values=model.encode([missing[k] for k in batch],batch_size=batch_size,normalize_embeddings=True,show_progress_bar=False)
        assert np.isfinite(values).all()
        for k,v in zip(batch,values):
            v=np.asarray(v,dtype=np.float32);found[k]=v
            database.execute('INSERT OR REPLACE INTO embeddings VALUES (?,?)',(k,v.tobytes()))
        database.commit()
    return np.stack([found[k] for k in keys])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--embedding-cache',type=Path,required=True)
    p.add_argument('--code-model-dir',type=Path,required=True)
    p.add_argument('--tasks',type=Path,default=ROOT/'data/tasks.jsonl')
    p.add_argument('--retrieval',type=Path,default=ROOT/'results/retrieval')
    p.add_argument('--output',type=Path,default=ROOT/'results/extension/semantic')
    args=p.parse_args()
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(4)
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    mini=SentenceTransformer(EMBEDDING_MODEL,revision=EMBEDDING_REVISION,device=device)
    code=SentenceTransformer(str(args.code_model_dir),trust_remote_code=True,device=device)
    db=sqlite3.connect(args.embedding_cache)
    db.execute('CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY,vector BLOB)')
    hashes={str(p.relative_to(args.code_model_dir)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in args.code_model_dir.rglob('*') if p.is_file() and '.cache' not in p.parts}
    save_json(args.output.parent/'semantic_environment.json',{'code_model':CODE_MODEL,'code_revision':CODE_REVISION,
              'model_asset_sha256':hashes,'minilm_revision':EMBEDDING_REVISION,'device':device,
              'torch_version':torch.__version__,'dtype':'float32','batch_size':4,'query_prefix':QUERY_PREFIX})
    for task in load_rows(args.tasks):
        tid=task['instance_id'];dest=args.output/(tid+'.json')
        if dest.exists():continue
        with gzip.open(args.cache/'corpus'/(tid+'.json.gz'),'rt') as h:corpus=json.load(h)
        retrieved=json.loads((args.retrieval/(tid+'.json')).read_text())
        paths=retrieved['rankings']['hybrid_rrf'][:100]
        by_path={f['path']:f for f in corpus['files']}
        assert set(paths)<=set(by_path) and len(paths)==len(set(paths))
        docs=[embedding_text(by_path[path]) for path in paths]
        result={'task_id':tid,'repo':task['repo'],'language':task['language'],'base_commit':task['base_commit'],
                'archive_sha256':corpus['archive_sha256'],'pool':paths,'document_sha256':[hashlib.sha256(s.encode()).hexdigest() for s in docs],
                'rankings':{},'scores':{},'seconds':{}}
        for name,model,limit,prefix,revision in [('minilm256_pool100',mini,256,'',EMBEDDING_REVISION),
                    ('coderank256_pool100',code,256,QUERY_PREFIX,CODE_REVISION),('coderank2048_pool100',code,2048,QUERY_PREFIX,CODE_REVISION)]:
            start=time.monotonic();model.max_seq_length=limit
            vectors=cached_vectors(docs,model,db,f'{revision}:{limit}',4)
            query=cached_vectors([prefix+task['problem_statement']],model,db,f'{revision}:{limit}:query',1)[0]
            scores=vectors@query
            result['rankings'][name]=[paths[i] for i in ranking(scores,paths)]
            result['scores'][name]=[float(x) for x in scores]
            result['seconds'][name]=time.monotonic()-start
        save_json(dest,result)
        print(tid,'complete', {k:round(v,1) for k,v in result['seconds'].items()},flush=True)
    db.close()

if __name__=='__main__':main()
