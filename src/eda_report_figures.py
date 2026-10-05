"""Readable report figures from existing executed EDA results, no new fitting."""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

COLORS={'batch1':'#27647B','batch2':'#D58037','batch3':'#6B7A51'}

def run(project_root: Path):
    root=Path(project_root);out=root/'outputs/day1';figs=out/'figures'
    q14=json.loads((out/'q1_q4_results.json').read_text())
    q35=json.loads((out/'q3_q5_results.json').read_text())
    rows=pd.DataFrame(q14['cell_records'])
    with plt.rc_context({'font.size':12,'axes.titlesize':13,'axes.labelsize':12,
                         'xtick.labelsize':11,'ytick.labelsize':11,'axes.spines.top':False,
                         'axes.spines.right':False}):
        from src.data import RawBatch
        snapshot=json.loads((root/'configs/data-snapshot.json').read_text())
        tables={}
        fig,axes=plt.subplots(1,3,figsize=(9.6,3.2),layout='constrained')
        for ax,record in zip(axes,snapshot['files']):
            batch=record['batch_id']
            with RawBatch(root/record['path'],batch) as raw:
                t=raw.time_frame(0,10)
                tables[batch]=raw.summary_table()
            ax.plot(t.t,t.I,color=COLORS[batch],lw=1.3)
            ax.axhline(0,color='#53616C',lw=.6)
            ax.set_title(batch.replace('batch','Batch ')+' | cell 0 / cycle 10')
            ax.set_xlabel('Original t (raw units)');ax.grid(alpha=.12)
        axes[0].set_ylabel('Current I (raw units)')
        fig.savefig(figs/'q4_report_current_patterns.png',dpi=300,bbox_inches='tight');plt.close(fig)
        fig,axes=plt.subplots(2,3,figsize=(9.6,6.0),layout='constrained')
        for j,(batch,color) in enumerate(COLORS.items()):
            t=tables[batch]
            for _,g in t.groupby('cell_id'):
                for ax in axes[:,j]:ax.plot(g.cycle,g.QDischarge,color=color,lw=.6,alpha=.3)
            flag=t[t.QDischarge>1.3];zero=t[t.QDischarge==0]
            axes[0,j].scatter(flag.cycle,flag.QDischarge,color='#9C3D35',s=18,zorder=3)
            axes[0,j].scatter(zero.cycle,zero.QDischarge,color='#9C3D35',marker='x',s=20,zorder=3)
            axes[0,j].set_ylim(-.08,3.05);axes[0,j].set_title(batch.replace('batch','Batch ')+': full range')
            axes[1,j].set_ylim(.75,1.15);axes[1,j].set_title('Zoom: 0.75-1.15 Ah')
            for ax in axes[:,j]:
                ax.set_xlabel('Original cycle number');ax.grid(alpha=.12)
                ax.locator_params(axis='x',nbins=4)
        axes[0,0].set_ylabel('Discharge capacity (Ah)')
        axes[1,0].set_ylabel('Discharge capacity (Ah)')
        fig.savefig(figs/'q2_report_degradation.png',dpi=300,bbox_inches='tight');plt.close(fig)
        for outcome,name,ylabel in (
            ('cycle_life','q4_report_current_life.png','Cycle life (cycles)'),
            ('qd_slope_cycle10_100','q4_report_current_slope.png','Initial Qd slope (raw units / cycle)'),
        ):
            fig,axes=plt.subplots(1,3,figsize=(10.5,3.6),layout='constrained')
            for ax,(batch,color) in zip(axes,COLORS.items()):
                group=rows[rows.batch_id==batch]
                good=np.isfinite(group.positive_current_mean)&np.isfinite(group[outcome])
                g=group[good]
                ax.scatter(g.positive_current_mean,g[outcome],s=35,color=color,alpha=.8,edgecolor='white',linewidth=.4)
                corr=next(r for r in q14['correlations'] if r['scope']==batch and
                          r['x']=='positive_current_mean' and r['y']==outcome)
                ax.set_title(f"{batch.replace('batch','Batch ')} | n={len(g)}\nSpearman rho={corr['spearman_rho']:+.3f}")
                ax.set_xlabel('Mean I, samples I > 0\n(cycle 10; raw units)')
                ax.grid(alpha=.12)
            axes[0].set_ylabel(ylabel)
            fig.savefig(figs/name,dpi=300,bbox_inches='tight');plt.close(fig)
        selected=['delta_q_min','delta_q_mean','delta_q_var','delta_q_log10var','delta_q_range']
        labels=['Delta Q min','Delta Q mean','Delta Q var','log10 var','Delta Q range']
        matrix=pd.DataFrame(q35['q5']['multicollinearity']['batch1']['pearson_matrix']).loc[selected,selected]
        fig,ax=plt.subplots(figsize=(6.5,4.8),layout='constrained')
        im=ax.imshow(matrix,vmin=-1,vmax=1,cmap='RdBu_r')
        ax.set_xticks(range(5),labels,rotation=30,ha='right')
        ax.set_yticks(range(5),labels)
        ax.set_title('Batch 1: redundancy among Delta Q statistics')
        for i in range(5):
            for j in range(5):
                value=matrix.iloc[i,j]
                ax.text(j,i,f'{value:.3f}',ha='center',va='center',fontsize=12,
                        color='white' if abs(value)>.6 else '#172E39')
        fig.colorbar(im,ax=ax,shrink=.85,label='Pearson r')
        fig.savefig(figs/'q5_report_redundancy.png',dpi=300,bbox_inches='tight');plt.close(fig)
        selected=['qd10','qd100','delta_qd100_10','qd_slope10_100','ir10',
                  'temperature_mean10_100','charge_time_mean10_100','delta_q_min','delta_q_log10var']
        labels=['Qd at cycle 10','Qd at cycle 100','Qd100 - Qd10','Qd slope 10-100','IR at cycle 10',
                'Mean Tavg 10-100','Mean charge time 10-100','Delta Q min','log10 Var(Delta Q)']
        scopes=['batch1','batch2','batch3','pooled']
        array=np.array([[next(r['spearman'] for r in q35['q5']['correlations'] if r['scope']==scope and r['feature']==feature)
                         for scope in scopes] for feature in selected])
        fig,ax=plt.subplots(figsize=(7.8,5.5),layout='constrained')
        im=ax.imshow(array,vmin=-1,vmax=1,cmap='RdBu_r',aspect='auto')
        ax.set_xticks(range(4),['Batch 1\nn=46','Batch 2\nn=39','Batch 3\nn=44','Pooled\nn=129'])
        ax.set_yticks(range(len(labels)),labels)
        ax.set_title('Selected early signals: Spearman correlation with life')
        for i in range(len(labels)):
            for j in range(4):
                value=array[i,j]
                ax.text(j,i,f'{value:+.2f}',ha='center',va='center',fontsize=12,
                        color='white' if abs(value)>.6 else '#172E39')
        fig.colorbar(im,ax=ax,shrink=.9,label='Spearman rho')
        fig.savefig(figs/'q5_report_correlations.png',dpi=300,bbox_inches='tight');plt.close(fig)
    result={'purpose':'Report readability views of executed Q4/Q5 exploratory results',
            'new_model_fitting':False,'source_results':['q1_q4_results.json','q3_q5_results.json'],
            'figures':['q2_report_degradation.png','q4_report_current_patterns.png',
                       'q4_report_current_life.png','q4_report_current_slope.png',
                       'q5_report_redundancy.png','q5_report_correlations.png']}
    (out/'report-figure-views.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    print(run(Path(__file__).resolve().parents[1]))
