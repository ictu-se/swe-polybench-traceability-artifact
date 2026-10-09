"""Exact-path analysis of frozen navigation, semantic, transfer and repair extensions."""
import argparse
import hashlib
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from analyze import gold,interval,metrics
from repository_benchmark import ROOT,load_rows,save_json
from run_models import repeat_panel

ENCODERS=('minilm256_pool100','coderank256_pool100','coderank2048_pool100')
SCOPES=('restricted40','repository')
FAMILIES=('qwen3-coder:30b','devstral-small-2:24b')


def require_matrix(records,fields,expected,label):
    actual=[tuple(r[k] for k in fields) for r in records]
    if len(actual)!=len(set(actual)) or set(actual)!=set(expected):
        raise ValueError(f'{label}: missing={len(set(expected)-set(actual))}, extra={len(set(actual)-set(expected))}, duplicates={len(actual)-len(set(actual))}')


def contrast(frame,a,b,cohort,seed_mode='primary'):
    keys=['task_id','repo']
    if seed_mode=='task_seed_mean':
        data=frame.groupby(keys+['method'],as_index=False).recall_10.mean()
    else:data=frame[frame.seed==11]
    left=data[data.method==a][keys+['recall_10']]
    right=data[data.method==b][['task_id','recall_10']]
    paired=left.merge(right,on='task_id',suffixes=('_a','_b')).dropna()
    if paired.empty:return None
    diff=paired.recall_10_a-paired.recall_10_b
    lo,hi=interval(diff,paired.repo,seed=20261009)
    tlo,thi=interval(diff,seed=20261009)
    return dict(cohort=cohort,seed_mode=seed_mode,a=a,b=b,n=len(paired),difference=diff.mean(),
                cluster_low=lo,cluster_high=hi,task_low=tlo,task_high=thi,
                wins=int((diff>1e-12).sum()),ties=int((abs(diff)<=1e-12).sum()),losses=int((diff< -1e-12).sum()))


def summarize(frame,cohort):
    rows=[]
    for method,g in frame[frame.seed==11].groupby('method'):
        item={'cohort':cohort,'method':method,'tasks':len(g),'valid_rate':g.valid.mean(),
              'mean_returned_10':g.n_returned_10.mean()}
        for col in g:
            if col.startswith(('recall_','precision_','returned_precision_','code_recall_','test_recall_','mrr_')):
                item[col]=g[col].mean()
        item['cluster_low'],item['cluster_high']=interval(g.recall_10,g.repo,seed=20261009)
        rows.append(item)
    return rows


def analyze_population(name,tasks,retrieval_dir,semantic_dir,nav_path,model_dir,complete):
    rows=[];targets=[];coverage=[];nav_stats=[];all_nav=[]
    model_records=[r for family in FAMILIES
                   for r in (load_rows(model_dir/('model_'+family.replace(':','_')+'.jsonl'))
                             if (model_dir/('model_'+family.replace(':','_')+'.jsonl')).exists() else [])]
    nav=load_rows(nav_path) if nav_path.exists() else []
    by_nav={};by_model={}
    for r in nav:by_nav.setdefault(r['task_id'],[]).append(r)
    for r in model_records:by_model.setdefault(r['task_id'],[]).append(r)
    if complete:
        panel=set(repeat_panel(tasks)) if name=='feature' else set()
        expected={(t['instance_id'],scope,s) for t in tasks for scope in SCOPES for s in (11,29,47)
                  if s==11 or t['instance_id'] in panel}
        require_matrix(nav,('task_id','scope','seed'),expected,name+' navigation')
        for family in FAMILIES:
            expected={(t['instance_id'],c,11) for t in tasks for c in ('paths','content')}
            require_matrix([r for r in model_records if r['model']==family and r['seed']==11],
                           ('task_id','condition','seed'),expected,name+' '+family)
    for task in tasks:
        tid=task['instance_id'];path=retrieval_dir/(tid+'.json')
        if not path.exists():
            if complete:raise ValueError('Missing retrieval '+tid)
            continue
        r=json.loads(path.read_text());gs=gold(task,set(r['universe']))
        common={'cohort':name,'task_id':tid,'repo':task['repo'],'language':task['language']}
        targets.extend({**common,**t} for t in gs)
        existing={t['path'] for t in gs if t['exists'] and t['status']!='added'}
        initial=set(r['rankings']['hybrid_rrf'][:40]);pool100=set(r['rankings']['hybrid_rrf'][:100])
        ratio=lambda s:len(existing&s)/len(existing) if existing else np.nan
        coverage.append({**common,'files':len(r['universe']),'existing':len(existing),'added':sum(t['status']=='added' for t in gs),
                         'absent_nonadded':sum(not t['exists'] and t['status']!='added' for t in gs),
                         'no_existing_tests':not any(t['path'] in existing and t['kind']=='test' for t in gs),
                         'coverage40':ratio(initial),'coverage100':ratio(pool100),
                         'issue_mentions_exact_target':any(p in task['problem_statement'] for p in existing),
                         'issue_characters':len(task['problem_statement'])})
        def add(method,pred,seed=11,valid=True):
            rows.append({**common,'method':method,'seed':seed,'valid':valid,**metrics(pred,gs)})
        for method,pred in r['rankings'].items():add(method,pred)
        semantic=semantic_dir/(tid+'.json')
        if semantic.exists():
            sr=json.loads(semantic.read_text())
            assert sr['pool']==r['rankings']['hybrid_rrf'][:100]
            assert set(sr['rankings'])==set(ENCODERS)
            for method,pred in sr['rankings'].items():
                assert set(pred)==pool100 and len(pred)==len(pool100)
                add(method,pred)
        elif complete:raise ValueError('Missing semantic '+tid)
        for mr in by_model.get(tid,[]):
            if mr['seed']==11:add(mr['model']+'/'+mr['condition'],mr['ranking'],valid=mr['valid'])
        for nr in by_nav.get(tid,[]):
            add('navigation/'+nr['scope'],nr['ranking'],nr['seed'],nr['valid'])
            assert nr['valid'] or nr['ranking']==[]
            seen=set(nr.get('seen_paths',[]));pred=set(nr['ranking'][:10]);outside=existing-initial
            reason='valid'
            if not nr['valid']:
                if nr.get('error'):reason='inference_error'
                else:
                    try:obj=json.loads(nr['steps'][-1]['response']['response'])
                    except (ValueError,KeyError,IndexError):obj={}
                    ids=obj.get('ranking')
                    if not isinstance(ids,list):reason='missing_ranking'
                    elif len(ids)!=len(set(map(str,ids))):reason='duplicate_ids'
                    else:reason='invalid_or_unobserved_ids'
            first_request=first_response=None
            if nr.get('steps'):
                step=nr['steps'][0]
                first_request=hashlib.sha256(json.dumps(step['request'],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
                try:
                    obj=json.loads(step['response']['response'])
                    if isinstance(obj,dict):first_response=hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
                except (ValueError,TypeError,KeyError):pass
            nav_stats.append({**common,'scope':nr['scope'],'seed':nr['seed'],'valid':nr['valid'],'reason':reason,
                    'first_request_sha256':first_request,'first_response_sha256':first_response,
                    'seen_paths':len(seen),'seen_gold_coverage':ratio(seen),'outside40_gold':len(outside),
                    'outside40_hits':len(pred&outside),'outside40_recall':len(pred&outside)/len(outside) if outside else np.nan,
                    'tool_actions':nr.get('tool_actions',0),'model_calls':len(nr.get('steps',[])),
                    'prompt_tokens':nr.get('prompt_tokens',0),'output_tokens':nr.get('output_tokens',0),
                    'elapsed_seconds':nr.get('elapsed_seconds',np.nan)})
    frame=pd.DataFrame(rows)
    if not frame.empty and frame.duplicated(['task_id','method','seed']).any():raise ValueError('Duplicate metric rows')
    return frame,pd.DataFrame(targets),pd.DataFrame(coverage),pd.DataFrame(nav_stats)


def repeats(tasks,path,retrieval_dir,complete):
    records=load_rows(path) if path.exists() else []
    if complete:
        require_matrix(records,('task_id','condition','seed'),
                       {(t['instance_id'],c,s) for t in tasks for c in ('paths','content') for s in (11,29,47)},'full Qwen repeat')
    by_task={t['instance_id']:t for t in tasks};rows=[]
    for r in records:
        task=by_task[r['task_id']];retrieval=json.loads((retrieval_dir/(r['task_id']+'.json')).read_text())
        rows.append({'task_id':r['task_id'],'repo':task['repo'],'method':'qwen3-coder:30b/'+r['condition'],
                     'seed':r['seed'],'valid':r['valid'],**metrics(r['ranking'],gold(task,set(retrieval['universe'])))})
    frame=pd.DataFrame(rows);summary=[]
    for method,g in frame.groupby('method'):
        pivot=g.pivot(index='task_id',columns='seed',values='recall_10').dropna()
        values=pivot.mean(axis=1)
        repos=[by_task[tid]['repo'] for tid in pivot.index]
        lo,hi=interval(values,repos,seed=20261009)
        summary.append({'method':method,'complete_tasks':len(pivot),'mean_across_task_seed_means':values.mean(),
                        'cluster_low':lo,'cluster_high':hi,'mean_within_task_sd':pivot.std(axis=1,ddof=1).mean(),
                        'mean_within_task_range':(pivot.max(axis=1)-pivot.min(axis=1)).mean(),
                        'tasks_with_varying_recall':int((pivot.nunique(axis=1)>1).sum()),
                        'invalid_outputs':int((~g.valid).sum())})
    return frame,pd.DataFrame(summary)


def repair(directory,ids,complete):
    rows=[]
    for tid in ids:
        pp=directory/(tid+'_preflight.json');ap=directory/(tid+'_agent.json')
        if not pp.exists() or not ap.exists():
            if complete:raise ValueError('Missing repair record '+tid)
            continue
        p=json.loads(pp.read_text());a=json.loads(ap.read_text())
        eligible=p.get('environment_valid',False)
        statuses={phase:Counter(status for names in a.get('grade',{}).get(phase,{}).values() for status in names.values()) for phase in ('f2p','p2p')}
        diagnostics={phase+'_'+status.lower():counts.get(status,0) for phase,counts in statuses.items() for status in ('PASSED','FAILED','ERROR','MISSING','SKIPPED','XFAIL','XPASS')}
        rows.append({'task_id':tid,'environment_valid':eligible,'status':a['status'],
                     'success':eligible and a.get('grade',{}).get('passed',False),
                     'base_valid':p.get('phases',{}).get('base',{}).get('grade',{}).get('passed',False),
                     'gold_valid':p.get('phases',{}).get('gold',{}).get('grade',{}).get('passed',False),
                     'model_calls':a.get('model_calls',0),'prompt_tokens':a.get('prompt_tokens',0),
                     'output_tokens':a.get('output_tokens',0),'nonempty_patch':bool(a.get('model_patch','').strip()),
                     'patch_bytes':len(a.get('model_patch','').encode('utf-8')),**diagnostics,
                     'error':p.get('error','') or a.get('error','')})
    frame=pd.DataFrame(rows)
    if not rows:return frame,{'attempted':0,'eligible':0,'successes':0}
    n=int(frame.environment_valid.sum());k=int(frame.success.sum());z=1.95996398454
    lo=hi=None
    if n:
        center=(k/n+z*z/(2*n))/(1+z*z/n)
        half=z*math.sqrt((k/n)*(1-k/n)/n+z*z/(4*n*n))/(1+z*z/n)
        lo,hi=max(0,center-half),min(1,center+half)
    return frame,{'attempted':len(rows),'eligible':n,'successes':k,'wilson_low':lo,'wilson_high':hi,
                  'interpretation':'descriptive task-level Wilson interval; small fixed panel, not independent population sampling or repository-cluster inference',
                  'status_counts':dict(Counter(frame.status))}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--allow-partial',action='store_true')
    p.add_argument('--results',type=Path,default=ROOT/'results');args=p.parse_args()
    ext=args.results/'extension';out=ext/'analysis';out.mkdir(parents=True,exist_ok=True)
    summaries=[];contrasts=[];pop={};navall=[];covall=[];targetall=[];frames=[]
    for name,taskfile,rdir,sdir,npath,mdir in [
        ('feature',ROOT/'data/tasks.jsonl',args.results/'retrieval',ext/'semantic',ext/'navigation.jsonl',args.results),
        ('external',ROOT/'data/external_tasks.jsonl',ext/'external/retrieval',ext/'external/semantic',ext/'external/navigation.jsonl',ext/'external')]:
        tasks=load_rows(taskfile)
        f,t,c,n=analyze_population(name,tasks,rdir,sdir,npath,mdir,not args.allow_partial)
        if f.empty:continue
        frames.append(f);targetall.append(t);covall.append(c);navall.append(n)
        summaries.extend(summarize(f,name))
        for a,b in [('coderank256_pool100','minilm256_pool100'),('coderank2048_pool100','coderank256_pool100'),
                    ('coderank2048_pool100','hybrid_rrf'),('navigation/repository','navigation/restricted40'),
                    ('navigation/repository','qwen3-coder:30b/content'),('navigation/restricted40','qwen3-coder:30b/content'),
                    ('qwen3-coder:30b/content','hybrid_rrf'),('devstral-small-2:24b/content','hybrid_rrf')]:
            con=contrast(f,a,b,name)
            if con:contrasts.append(con)
        pop[name]={'tasks':len(c),'repositories':len(set(c.repo)),'targets':len(t),'existing_targets':int(c.existing.sum()),
                   'added_targets':int(c.added.sum()),'absent_nonadded':int(c.absent_nonadded.sum()),
                   'tasks_no_existing':int((c.existing==0).sum()),'tasks_no_existing_test':int(c.no_existing_tests.sum()),
                   'median_files':float(c.files.median()),'min_files':int(c.files.min()),'max_files':int(c.files.max()),
                   'coverage40':float(c.coverage40.mean()),'coverage100':float(c.coverage100.mean()),
                   'issue_mentions_exact_target':int(c.issue_mentions_exact_target.sum()),'truncated_issues':int((c.issue_characters>8000).sum())}
    for filename,parts in [('task_metrics',frames),('targets',targetall),('coverage',covall),('navigation',navall)]:
        if parts:pd.concat(parts,ignore_index=True).to_csv(out/(filename+'.csv'),index=False)
    if navall:
        navigation=pd.concat(navall,ignore_index=True)
        if not navigation.empty:
            keys=['cohort','task_id','repo','seed']
            fields=keys+['first_request_sha256','first_response_sha256']
            pairs=navigation[navigation.scope=='restricted40'][fields].merge(
                navigation[navigation.scope=='repository'][fields],on=keys,suffixes=('_restricted','_repository'))
            pairs['comparable_requests']=pairs.first_request_sha256_restricted.notna()&pairs.first_request_sha256_repository.notna()
            pairs['same_initial_request']=pairs.first_request_sha256_restricted==pairs.first_request_sha256_repository
            pairs['comparable_responses']=pairs.first_response_sha256_restricted.notna()&pairs.first_response_sha256_repository.notna()
            pairs['same_initial_structured_response']=pairs.first_response_sha256_restricted==pairs.first_response_sha256_repository
            pairs.to_csv(out/'navigation_pairs.csv',index=False)
    pd.DataFrame(summaries).to_csv(out/'summary.csv',index=False)
    tasks=load_rows(ROOT/'data/tasks.jsonl')
    rf,rs=repeats(tasks,ext/'repeats/model_qwen3-coder_30b.jsonl',args.results/'retrieval',not args.allow_partial)
    rf.to_csv(out/'repeat_task_metrics.csv',index=False);rs.to_csv(out/'repeat_summary.csv',index=False)
    con=contrast(rf,'qwen3-coder:30b/content','qwen3-coder:30b/paths','feature','task_seed_mean')
    if con:contrasts.append(con)
    pd.DataFrame(contrasts).to_csv(out/'contrasts.csv',index=False)
    ids=json.loads((ROOT/'data/external_protocol.json').read_text())['repair_panel_ids']
    repair_frame,repair_summary=repair(ext/'repair',ids,not args.allow_partial)
    repair_frame.to_csv(out/'repair.csv',index=False)
    save_json(out/'summary.json',{'complete':not args.allow_partial,'populations':pop,'repair':repair_summary})
    print('Extension analysis written; complete =',not args.allow_partial)

if __name__=='__main__':main()
