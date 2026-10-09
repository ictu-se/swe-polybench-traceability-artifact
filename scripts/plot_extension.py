"""Generate extension figures from archived empirical task-level measurements."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from repository_benchmark import ROOT

LABELS={'hybrid_rrf':'Hybrid RRF','minilm256_pool100':'MiniLM, 256','coderank256_pool100':'CodeRank, 256',
        'coderank2048_pool100':'CodeRank, 2048','qwen3-coder:30b/content':'Qwen, static excerpts',
        'navigation/restricted40':'Navigation, restricted','navigation/repository':'Navigation, repository'}
COLORS=['#6b7280','#b58a31','#5b9a6b','#15754c','#8856a7','#3182bd','#08306b']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,default=ROOT/'results/extension')
    args=p.parse_args();out=args.results/'figures';out.mkdir(exist_ok=True)
    frame=pd.read_csv(args.results/'analysis/task_metrics.csv');nav=pd.read_csv(args.results/'analysis/navigation.csv')
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(4.68,4.0),sharey=True)
    for ax,cohort,title in zip(axes,['feature','external'],['(a) n=184','(b) n=24']):
        g=frame[(frame.cohort==cohort)&(frame.seed==11)]
        for ((method,label),color,marker,style) in zip(LABELS.items(),COLORS,['o','s','^','D','x','v','P'],['-','--','-.',':','--','-.','-']):
            subset=g[g.method==method]
            ax.plot([1,3,5,10],[subset['recall_'+str(k)].mean() for k in [1,3,5,10]],marker=marker,linestyle=style,markersize=3,label=label,color=color,linewidth=1.3)
        ax.set(title=title,xlabel='Joint prediction budget k',xticks=[1,3,5,10],ylim=(0,1))
        ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Macro existing-target recall')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,fontsize=8)
    fig.tight_layout(rect=(0,.26,1,1));fig.savefig(out/'extension_budget_curves.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(4.68,3.5))
    for ax,cohort,title in zip(axes,['feature','external'],['(a)','(b)']):
        f=frame[(frame.cohort==cohort)&(frame.seed==11)&frame.method.str.startswith('navigation/')]
        paired=f.pivot(index='task_id',columns='method',values='recall_10').dropna()
        delta=(paired['navigation/repository']-paired['navigation/restricted40']).sort_values().to_numpy()
        ax.bar(np.arange(len(delta)),delta,color=np.where(delta>=0,'#15754c','#c26a4a'),width=1)
        ax.axhline(0,color='#555555',linewidth=.6)
        ax.set(title=title,xlabel='Tasks ordered by\npaired difference',ylabel='Repository minus restricted R@10',ylim=(-1,1))
    fig.tight_layout();fig.savefig(out/'navigation_scope_differences.pdf');plt.close(fig)
    print('Generated two extension figures from archived measurements')

if __name__=='__main__':main()
