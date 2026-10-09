"""Reproduce the frozen external sample from pinned data and archived public metadata.

No model output is used. The metadata snapshot preserves issue-title matching and
creation-date decisions; re-querying GitHub today could see edited issue titles.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import pandas as pd
import requests

from repository_benchmark import ROOT,load_rows


def normalize(text):
    return re.sub(r'\s+',' ',text).strip().casefold()


def select(rows,original_repos,metadata,languages):
    candidates=[r for r in rows if not r.get('interface') and '_interface' not in r['instance_id']
                and r['repo'] not in original_repos and r['created_at']>='2026-01-01']
    candidates.sort(key=lambda r:hashlib.sha256(('external-20261009:'+r['instance_id']).encode()).hexdigest())
    selected=[];counts={}
    for source in candidates:
        row=dict(source)
        if counts.get(row['repo'],0)>=2:continue
        info=metadata[row['instance_id']]
        if 'pr' not in info:continue
        pr=info['pr'];matches=[i for i in pr['closingIssuesReferences']['nodes']
                              if normalize(i['title']) in normalize(row['problem_statement'])]
        if not matches or any(i['createdAt']<'2026-01-01' for i in matches):continue
        row.update(language=languages[row['repo']],verified_issue_metadata=matches,
                   verified_pr_metadata={k:pr[k] for k in ['url','createdAt','mergedAt']})
        selected.append(row);counts[row['repo']]=counts.get(row['repo'],0)+1
        if len(selected)==24:break
    return selected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True,help='Use a new path to preserve archived tasks')
    args=p.parse_args();args.cache.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((ROOT/'data/external_protocol.json').read_text())
    source=args.cache/'rebench_2026_03.parquet'
    if not source.exists():
        url=f"https://huggingface.co/datasets/{protocol['dataset']}/resolve/{protocol['revision']}/data/2026_03-00000-of-00001.parquet"
        response=requests.get(url,timeout=180);response.raise_for_status();source.write_bytes(response.content)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==protocol['source_sha256']
    rows=json.loads(pd.read_parquet(source).to_json(orient='records'))
    metadata={r['task_id']:r for r in json.loads((ROOT/'data/external_selection_metadata.json').read_text())}
    selected=select(rows,{r['repo'] for r in load_rows(ROOT/'data/tasks.jsonl')},metadata,protocol['repository_languages'])
    assert [r['instance_id'] for r in selected]==protocol['selected_ids']
    text=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected)
    assert hashlib.sha256(text.encode()).hexdigest()==protocol['tasks_sha256']
    if args.output.exists():raise SystemExit('Output exists; choose a fresh path')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text)
    print('Reproduced24 tasks across20 repositories, exact archived task checksum matched')

if __name__=='__main__':main()
