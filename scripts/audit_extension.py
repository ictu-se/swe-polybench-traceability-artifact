"""Audit complete extension matrices, pinned provenance and model/tool budgets."""
import hashlib
import gzip
from datetime import datetime
import json
from pathlib import Path
import numpy as np

from analyze_extension import ENCODERS,FAMILIES,SCOPES,require_matrix
from analyze import patch_targets
from repository_benchmark import ROOT,load_rows,save_json,ranking
from run_models import repeat_panel,make_prompt,SYSTEM,SCHEMA
from repair_probe import parse_tests,expected_test_ids,grade,ACTION_SCHEMA as REPAIR_SCHEMA,SYSTEM as REPAIR_SYSTEM
from navigation_agent import ACTION_SCHEMA as NAV_SCHEMA,SYSTEM as NAV_SYSTEM


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_static(rows,tasks,retrieval_dir):
    by_id={t['instance_id']:t for t in tasks}
    retrieved={tid:json.loads((retrieval_dir/(tid+'.json')).read_text()) for tid in by_id}
    for row in rows:
        task=by_id[row['task_id']];inputs=retrieved[row['task_id']];request=row['request']
        assert row['repo']==task['repo'] and row['language']==task['language']
        assert row['digest']==sha(ROOT/'data/model_manifests'/(row['model'].replace(':','_')+'.json'))
        assert row['candidate_paths']==[c['path'] for c in inputs['candidates']]
        assert request['model']==row['model'] and request['system']==SYSTEM and request['format']==SCHEMA
        assert request['prompt']==make_prompt(task,inputs,row['condition'])
        assert request['options']==dict(temperature=.2,top_p=.9,top_k=40,seed=row['seed'],num_ctx=16384,num_predict=160)
        if row['valid']:
            ids=json.loads(row['response']['response'])['ranking']
            assert len(ids)<=10 and len(ids)==len(set(ids))
            assert all(type(i) is int and 1<=i<=len(inputs['candidates']) for i in ids)
            assert row['ranking']==[inputs['candidates'][i-1]['path'] for i in ids]
        else:assert row['ranking']==[]


def audit_navigation(rows,tasks,retrieval_dir,panel):
    require_matrix(rows,('task_id','scope','seed'),
        {(t['instance_id'],scope,s) for t in tasks for scope in SCOPES for s in (11,29,47)
         if s==11 or t['instance_id'] in panel},'navigation')
    paired={}
    for row in rows:paired.setdefault((row['task_id'],row['seed']),{})[row['scope']]=row
    for pair in paired.values():
        restricted,repository=pair['restricted40'],pair['repository']
        if restricted['steps'] and repository['steps']:
            assert restricted['steps'][0]['request']==repository['steps'][0]['request'], 'Initial navigation inputs differ across scopes'
    digest=sha(ROOT/'data/model_manifests/qwen3-coder_30b.json')
    retrieved={t['instance_id']:json.loads((retrieval_dir/(t['instance_id']+'.json')).read_text()) for t in tasks}
    for row in rows:
        r=retrieved[row['task_id']];universe=r['universe'];seen=set(r['rankings']['hybrid_rrf'][:10])
        allowed=set(universe) if row['scope']=='repository' else set(r['rankings']['hybrid_rrf'][:40])
        assert row['digest']==digest and row['archive_sha256']==r['archive_sha256']
        assert len(row['steps'])<=5
        for turn,step in enumerate(row['steps']):
            request=step['request'];opts=request['options']
            assert request['model']=='qwen3-coder:30b'
            assert request['system']==NAV_SYSTEM and request['format']==(SCHEMA if turn==4 else NAV_SCHEMA)
            assert opts==dict(temperature=.2,top_p=.9,top_k=40,seed=row['seed'],num_ctx=16384,num_predict=160 if turn==4 else 256)
            observation=step.get('observation',{})
            files=observation.get('matches',[]) or ([observation] if 'id' in observation else [])
            assert len(files)<=12
            for file in files:
                assert type(file['id']) is int and 1<=file['id']<=len(universe) and universe[file['id']-1]==file['path'] and file['path'] in allowed
                seen.add(file['path'])
        if row.get('error'):
            assert not row['valid'] and row['ranking']==[]
            continue
        assert set(row['seen_paths'])==seen
        assert row['tool_actions']==sum('observation' in s for s in row['steps'])<=4
        try:final=json.loads(row['steps'][-1]['response']['response'])
        except (ValueError,TypeError,KeyError,IndexError):final={}
        ids=final.get('ranking') if isinstance(final,dict) else None
        valid=isinstance(ids,list) and len(ids)<=10 and all(type(i) is int and 1<=i<=len(universe) for i in ids) and len(ids)==len(set(ids))
        paths=[universe[i-1] for i in ids] if valid else []
        valid=valid and set(paths)<=seen&allowed
        assert row['valid']==valid and row['ranking']==(paths if valid else []), 'Navigation response-to-ranking mismatch'
        if row['valid']:
            assert len(row['ranking'])<=10 and len(row['ranking'])==len(set(row['ranking']))
            assert set(row['ranking'])<=seen&allowed
        else:assert row['ranking']==[]


def main():
    extension=ROOT/'results/extension';protocol=json.loads((ROOT/'data/extension_protocol.json').read_text())
    external=json.loads((ROOT/'data/external_protocol.json').read_text())
    assert sha(ROOT/'scripts/navigation_agent.py')==protocol['navigation_script_sha256']
    assert sha(ROOT/'scripts/repair_probe.py')==external['repair_protocol']['repair_script_sha256']
    assert sha(ROOT/'data/external_tasks.jsonl')==external['tasks_sha256']
    core=load_rows(ROOT/'data/tasks.jsonl');other=load_rows(ROOT/'data/external_tasks.jsonl')
    assert len(core)==184 and len(other)==24 and len({t['repo'] for t in other})==20
    assert not {t['repo'] for t in core}&{t['repo'] for t in other}
    assert all(t['language']=='Python' and t['verified_issue_metadata'] and all(i['createdAt']>='2026-01-01' for i in t['verified_issue_metadata']) for t in other)
    counts={}
    for name,tasks,base,semantic,nav,panel in [
        ('feature',core,ROOT/'results',extension/'semantic',extension/'navigation.jsonl',set(repeat_panel(core))),
        ('external',other,extension/'external',extension/'external/semantic',extension/'external/navigation.jsonl',set())]:
        rows=load_rows(nav);audit_navigation(rows,tasks,base/'retrieval',panel)
        environment=json.loads((semantic.parent/'semantic_environment.json').read_text())
        assert environment['code_revision']==protocol['semantic_reranking']['revision']
        assert environment['query_prefix']==protocol['semantic_reranking']['query_prefix']
        assert environment['dtype']=='float32' and environment['batch_size']==4
        if name=='feature':encoder_assets=environment['model_asset_sha256']
        else:assert environment['model_asset_sha256']==encoder_assets
        assert {p.stem for p in semantic.glob('*.json')}=={t['instance_id'] for t in tasks}
        for task in tasks:
            r=json.loads((base/'retrieval'/(task['instance_id']+'.json')).read_text())
            s=json.loads((semantic/(task['instance_id']+'.json')).read_text())
            assert s['pool']==r['rankings']['hybrid_rrf'][:100]
            assert s['archive_sha256']==r['archive_sha256']
            assert set(s['rankings'])==set(ENCODERS)
            assert all(len(pred)==len(s['pool']) and set(pred)==set(s['pool']) for pred in s['rankings'].values())
            for method in ENCODERS:
                scores=np.asarray(s['scores'][method])
                assert len(scores)==len(s['pool']) and np.isfinite(scores).all()
                assert s['rankings'][method]==[s['pool'][i] for i in ranking(scores,s['pool'])]

        counts[name]={'semantic':len(tasks),'navigation':len(rows),'invalid_navigation':sum(not r['valid'] for r in rows)}
    for family in FAMILIES:
        rows=load_rows(extension/'external'/('model_'+family.replace(':','_')+'.jsonl'))
        require_matrix(rows,('task_id','condition','seed'),{(t['instance_id'],c,11) for t in other for c in ('paths','content')},family+' external')
        audit_static(rows,other,extension/'external/retrieval')
        expected=sha(ROOT/'data/model_manifests'/(family.replace(':','_')+'.json'))
        assert all(r['digest']==expected and (r['valid'] or not r['ranking']) for r in rows)
    repeat=load_rows(extension/'repeats/model_qwen3-coder_30b.jsonl')
    require_matrix(repeat,('task_id','condition','seed'),
                   {(t['instance_id'],c,s) for t in core for c in ('paths','content') for s in (11,29,47)},'full repeats')
    audit_static(repeat,core,ROOT/'results/retrieval')
    original=load_rows(ROOT/'results/model_qwen3-coder_30b.jsonl')
    by_key={(r['task_id'],r['condition'],r['seed']):r for r in repeat}
    assert all(by_key[(r['task_id'],r['condition'],r['seed'])]==r for r in original),'Original outputs were modified'
    input_records=json.loads((extension/'repair/input_audit.json').read_text())
    input_meta=json.loads((extension/'repair/input_audit_metadata.json').read_text())
    assert input_meta['passed'] and input_meta['generated_patch_records_at_completion']==0
    assert sha(extension/'repair/input_audit.json')==input_meta['records_sha256']
    assert sha(ROOT/'scripts/audit_repair_inputs.py')==input_meta['inspection_script_sha256']
    assert {r['task_id'] for r in input_records}==set(external['repair_panel_ids'])
    assert all(r['input_checks_passed'] and r['head']==r['base_commit'] and
               not any(r[k] for k in ('refs','reflog','status','unreachable_objects','nonancestor_reachable_commits','reference_files','bind_mounts')) and
               r['network_mode']=='none' for r in input_records)
    verification=json.loads((extension/'repair/verification_manifest.json').read_text())
    assert verification['complete'] and verification['agent_records']==12
    for entry in verification['outputs']:
        archive=ROOT/entry['archive']
        assert sha(archive)==entry['archive_sha256']
        content=gzip.decompress(archive.read_bytes())
        assert len(content)==entry['output_bytes'] and hashlib.sha256(content).hexdigest()==entry['output_sha256']
        record=json.loads((ROOT/entry['record']).read_text())
        phase=record['evaluation'] if entry['phase']=='model' else record['phases'][entry['phase']]
        assert phase['output_sha256']==entry['output_sha256'] and phase['statuses']==parse_tests(content.decode('utf-8'))
    external_tasks={task['instance_id']:task for task in other}
    for tid in external['repair_panel_ids']:
        pre=json.loads((extension/'repair'/(tid+'_preflight.json')).read_text())
        agent=json.loads((extension/'repair'/(tid+'_agent.json')).read_text())
        assert datetime.fromisoformat(input_meta['completed_at'])<datetime.fromisoformat(agent['started_at'])
        task=external_tasks[tid]
        protected={change['new'] if change['status']=='added' else change['old'] for change in patch_targets(task['test_patch'])}
        verification_paths={name.split('::',1)[0] for name in task['FAIL_TO_PASS']+task['PASS_TO_PASS']}
        assert verification_paths<=protected, 'Verification tests outside restored paths'
        reference=pre.get('phases',{}).get('gold',{}).get('statuses',{})
        mapping=expected_test_ids(task,reference)
        assert pre.get('grading_version')=='frozen-full-test-ids-v3'
        assert pre['expected_test_ids']==mapping
        for phase,result in pre.get('phases',{}).items():
            assert result['grade']==grade(result['statuses'],task,base=phase=='base',expected_ids=mapping)
        if pre.get('environment_valid'):
            assert pre['architecture']=='amd64' and '@sha256:' in pre['image_digest']
            assert all(pre['phases'][phase]['grade']['passed'] for phase in ('base','gold'))
            assert agent['status'] in ('evaluated','infrastructure_error')
            assert len(agent.get('steps',[]))<=12
            for step in agent.get('steps',[]):
                request=step['request']
                assert request['model']=='qwen3-coder:30b'
                assert request['system']==REPAIR_SYSTEM and request['format']==REPAIR_SCHEMA
                assert request['options']==dict(temperature=.2,top_p=.9,top_k=40,seed=11,num_ctx=32768,num_predict=768)
                assert request['prompt'].startswith('Issue:\n'+task['problem_statement'][:8000]+'\n\n')
            if agent['status']=='evaluated':
                assert agent['digest']==sha(ROOT/'data/model_manifests/qwen3-coder_30b.json')
                assert agent['grade']==grade(agent['evaluation']['statuses'],task,expected_ids=mapping)
        else:assert agent['status']=='environment_ineligible'
    result={'passed':True,'populations':counts,'full_repeat_records':len(repeat),'original_repeat_records_unchanged':len(original),'repair_panel':len(external['repair_panel_ids'])}
    save_json(extension/'audit.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
