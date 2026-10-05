"""DAY 1 Q3/Q5 exploratory early-cycle signals; original cells retained.

No model fitting, final feature selection, outlier removal or interpolation.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import RawBatch

COLORS = {'batch1': '#27647B', 'batch2': '#D58037', 'batch3': '#6B7A51'}
DEFINITIONS = {
    'qd10': 'Original summary.QDischarge at actual cycle 10 (Ah).',
    'qd100': 'Original summary.QDischarge at actual cycle 100 (Ah).',
    'delta_qd100_10': 'QDischarge(cycle 100) - QDischarge(cycle 10), Ah.',
    'qd_slope10_100': 'Least-squares descriptive QDischarge slope against actual cycle number, cycles 10 through 100 inclusive, Ah/cycle; finite observed pairs only.',
    'ir10': 'Original summary.IR at actual cycle 10 (original units).',
    'ir100': 'Original summary.IR at actual cycle 100 (original units).',
    'delta_ir100_10': 'IR(cycle 100) - IR(cycle 10), original units.',
    'ir_mean10_100': 'Mean finite observed summary.IR values over actual cycles 10 through 100 inclusive; zero values retained.',
    'temperature_mean10_100': 'Mean finite observed summary.Tavg values over actual cycles 10 through 100 inclusive (deg C); zero values retained.',
    'charge_time_mean10_100': 'Mean finite observed summary.chargetime values over actual cycles 10 through 100 inclusive (original units); zero values retained.',
    'delta_q_min': 'Minimum Qdlin(100,V)-Qdlin(10,V) across the exactly matching original Vdlin grid (Ah).',
    'delta_q_mean': 'Mean Qdlin(100,V)-Qdlin(10,V) across exactly matching original Vdlin grid (Ah).',
    'delta_q_var': 'Population variance (ddof=0) of Qdlin(100,V)-Qdlin(10,V) across original grid (Ah^2).',
    'delta_q_log10var': 'log10(population variance of delta Q expressed numerically in Ah^2); defined only for finite strictly positive variance; no epsilon added.',
    'delta_q_range': 'Maximum minus minimum delta Q over original grid (Ah).',
}


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.bool_,)): return bool(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def finite_mean(values):
    x = np.asarray(values, float)
    z = x[np.isfinite(x)]
    return float(z.mean()) if len(z) else np.nan


def exact_value(summary, cycle, field):
    x = summary.loc[summary.cycle == cycle, field].to_numpy()
    return float(x[0]) if len(x) == 1 else np.nan


def describe_values(values):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    return {'n': len(x), 'mean': x.mean() if len(x) else np.nan,
            'median': np.median(x) if len(x) else np.nan,
            'sd': x.std(ddof=1) if len(x)>1 else np.nan,
            'min': x.min() if len(x) else np.nan, 'max': x.max() if len(x) else np.nan,
            'q25': np.quantile(x, .25) if len(x) else np.nan,
            'q75': np.quantile(x, .75) if len(x) else np.nan}


def pair_correlations(x, y):
    good = np.isfinite(x) & np.isfinite(y)
    a, b = np.asarray(x)[good], np.asarray(y)[good]
    if len(a)<3 or np.ptp(a)==0 or np.ptp(b)==0:
        return {'n': len(a), 'pearson': None, 'spearman': None}
    return {'n': len(a), 'pearson': float(stats.pearsonr(a,b).statistic),
            'spearman': float(stats.spearmanr(a,b).statistic)}


def collect():
    snapshots = json.loads((ROOT/'configs/data-snapshot.json').read_text())
    records, curves, quality = [], {}, {}
    grids = {}
    for file_info in snapshots['files']:
        batch_id = file_info['batch_id']
        with RawBatch(ROOT/file_info['path'], batch_id) as raw:
            cell_table = raw.cell_table()
            qv, availability = raw.qv_table((10,100))
            batch_curves = []
            source_q = {field:{'n_observed':0,'n_nonfinite':0,'n_zero':0} for field in ('QDischarge','IR','Tavg','chargetime')}
            qv_status = []
            for row in cell_table.itertuples(index=False):
                summary = raw.summary_for_cell(int(row.cell_id))
                early = summary[(summary.cycle>=10)&(summary.cycle<=100)]
                entry = {'batch_id': batch_id, 'cell_id': int(row.cell_id),
                         'cycle_life': float(row.cycle_life), 'policy_readable': row.policy_readable,
                         'life_group': 'unknown' if not np.isfinite(row.cycle_life) else ('long' if row.cycle_life>1000 else ('short' if row.cycle_life<500 else 'middle'))}
                for field in source_q:
                    x=early[field].to_numpy(float)
                    source_q[field]['n_observed'] += len(x)
                    source_q[field]['n_nonfinite'] += int((~np.isfinite(x)).sum())
                    source_q[field]['n_zero'] += int((x==0).sum())
                    entry['n_finite_'+field+'_10_100'] = int(np.isfinite(x).sum())
                    entry['n_zero_'+field+'_10_100'] = int((x==0).sum())
                entry['n_observed_cycles10_100'] = len(early)
                entry['qd10'] = exact_value(summary,10,'QDischarge')
                entry['qd100'] = exact_value(summary,100,'QDischarge')
                entry['delta_qd100_10'] = entry['qd100']-entry['qd10']
                x,y=early.cycle.to_numpy(float),early.QDischarge.to_numpy(float)
                good=np.isfinite(x)&np.isfinite(y)
                entry['qd_slope10_100'] = float(stats.linregress(x[good],y[good]).slope) if good.sum()>=2 and np.ptp(x[good])>0 else np.nan
                entry['ir10'] = exact_value(summary,10,'IR')
                entry['ir100'] = exact_value(summary,100,'IR')
                entry['delta_ir100_10'] = entry['ir100']-entry['ir10']
                entry['ir_mean10_100'] = finite_mean(early.IR)
                entry['temperature_mean10_100'] = finite_mean(early.Tavg)
                entry['charge_time_mean10_100'] = finite_mean(early.chargetime)
                a = qv[(qv.cell_id==row.cell_id)&(qv.cycle==10)]
                b = qv[(qv.cell_id==row.cell_id)&(qv.cycle==100)]
                grid_ok = bool(len(a)>0 and len(a)==len(b) and np.array_equal(a.Vdlin.to_numpy(),b.Vdlin.to_numpy()))
                v=a.Vdlin.to_numpy(float)
                dq=b.Qdlin.to_numpy(float)-a.Qdlin.to_numpy(float) if grid_ok else np.empty(0)
                usable=bool(grid_ok and np.isfinite(v).all() and np.isfinite(dq).all())
                entry['delta_q_status'] = 'available' if usable else ('grid_mismatch_or_empty' if not grid_ok else 'nonfinite_curve')
                qv_status.append({'cell_id':row.cell_id, 'cycle10_points':len(a), 'cycle100_points':len(b),
                                 'exact_grid_match':grid_ok,'finite_curve':usable,'status':entry['delta_q_status']})
                for name in ('delta_q_min','delta_q_mean','delta_q_var','delta_q_log10var','delta_q_range'): entry[name]=np.nan
                if usable:
                    variance = float(np.var(dq,ddof=0))
                    entry.update(delta_q_min=float(dq.min()),delta_q_mean=float(dq.mean()),delta_q_var=variance,
                                 delta_q_log10var=float(np.log10(variance)) if variance>0 else np.nan,
                                 delta_q_range=float(np.ptp(dq)))
                    batch_curves.append({'cell_id':row.cell_id, 'cycle_life':row.cycle_life,
                                         'life_group':entry['life_group'],'voltage':v,'delta_q':dq})
                    grids[(batch_id,int(row.cell_id))]=v
                records.append(entry)
            curves[batch_id]=batch_curves
            quality[batch_id]={'n_cells':len(cell_table),'qv_availability':availability.status.value_counts().to_dict(),
                               'qv_cell_checks':qv_status,'early_source_quality':source_q,
                               'unique_voltage_grids_within_batch':sum(not any(np.array_equal(c['voltage'],prev['voltage']) for prev in batch_curves[:i]) for i,c in enumerate(batch_curves))}
    feature_table=pd.DataFrame(records)
    ref=next(iter(grids.values()))
    grid_summary={'all_139_cells_exactly_equal':all(np.array_equal(ref,v) for v in grids.values()),
                  'point_count':len(ref),'voltage_first':ref[0],'voltage_last':ref[-1],
                  'voltage_min':ref.min(),'voltage_max':ref.max(),
                  'interpolation_performed':False,
                  'scope':'Actual cycles 10 and 100 only; equality here does not establish alignment at every later cycle or equality of Qdlin capacity-origin/collection definitions across batches.'}
    return feature_table,curves,quality,grid_summary


def hypothesis_test(table):
    b2=table[table.batch_id=='batch2']
    long=b2.loc[b2.life_group=='long','delta_q_log10var'].to_numpy(float)
    short=b2.loc[b2.life_group=='short','delta_q_log10var'].to_numpy(float)
    if not np.isfinite(long).all() or not np.isfinite(short).all():
        raise ValueError('Nonfinite primary test values require an explicit handling decision')
    welch=stats.ttest_ind(long,short,equal_var=False,alternative='two-sided',nan_policy='raise')
    ci=welch.confidence_interval(.95)
    perm=stats.permutation_test((long,short),lambda x,y:np.mean(x)-np.mean(y),
                                permutation_type='independent',vectorized=False,n_resamples=np.inf,
                                alternative='two-sided')
    # The primary test is Welch; the permutation test is an approved robustness
    # check of the same pre-described statistic, not another feature search.
    return {'status':'executed', 'primary_feature':'delta_q_log10var', 'scope':'batch2 only',
            'cell_unit':'One original cell per observation; 1000 voltage coordinates are not independent samples.',
            'alpha':.05, 'alternative':'two-sided',
            'welch_null':'Equal population means of log10(Var(delta Q)); unequal variances allowed.',
            'n_long':len(long),'n_short':len(short),
            'long_cell_ids':b2.loc[b2.life_group=='long','cell_id'].tolist(),
            'short_cell_ids':b2.loc[b2.life_group=='short','cell_id'].tolist(),
            'mean_long':long.mean(),'mean_short':short.mean(),'difference_long_minus_short':long.mean()-short.mean(),
            'geometric_mean_variance_ratio_long_over_short':float(10**(long.mean()-short.mean())),
            'variance_ratio_note':'Back-transform of the difference in mean log10 variance; ratio of geometric means, not ratio of arithmetic means.',
            'welch_t':welch.statistic,'welch_df':welch.df,'welch_p_two_sided':welch.pvalue,
            'welch_95pct_ci_difference':[ci.low,ci.high],
            'welch_reject_at_0_05':bool(welch.pvalue<.05),
            'permutation_null':'Same underlying distributions with exchangeable cell group labels; stronger than equality of means alone.',
            'permutation_statistic':'mean(long)-mean(short)',
            'permutation_n_distinct_partitions':len(perm.null_distribution),
            'permutation_p_two_sided':perm.pvalue,
            'permutation_two_sided_convention':'SciPy: 2 times the smaller one-sided tail probability, capped at 1; exact enumeration includes all 4495 distinct 3-of-31 partitions.',
            'permutation_reject_at_0_05':bool(perm.pvalue<.05),
            'permutation_null_distribution':perm.null_distribution.tolist(),
            'protocols_by_group':{g:b2.loc[b2.life_group==g,'policy_readable'].value_counts().to_dict() for g in ('long','short')},
            'limitations':['Long group n=3 is too small to meaningfully validate normality; no Shapiro gatekeeping test performed.',
                           'Independent cell measurements and permutation exchangeability are assumptions, not established by this observational test.',
                           'Charging protocol differences remain possible confounding even within Batch 2.',
                           'The feature and comparison were chosen after EDA, so p-values are exploratory and do not supply independent confirmation.',
                           'The result is an association, not proof of a causal mechanism or proven predictive generalization.',
                           'Only this primary feature was hypothesis-tested. No other feature p-values or multiple-testing feature search reported.'],
            'official_references':['https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html',
                                   'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html']}


def analysis(table,curves,quality,grids):
    features=list(DEFINITIONS)
    corr=[]
    missing={}
    group_values={}
    for batch_id in ('pooled','batch1','batch2','batch3'):
        t=table if batch_id=='pooled' else table[table.batch_id==batch_id]
        missing[batch_id]={name:int((~np.isfinite(t[name])).sum()) for name in features}
        for name in features:
            corr.append({'scope':batch_id,'feature':name,'target':'original_cycle_life',
                         **pair_correlations(t[name].to_numpy(float),t.cycle_life.to_numpy(float))})
        group_values[batch_id]={}
        for group in ('short','middle','long','unknown'):
            rows=t[t.life_group==group]
            group_values[batch_id][group]={name:describe_values(rows[name]) for name in ('delta_q_log10var','delta_q_min','delta_q_mean','delta_q_var')}
            group_values[batch_id][group]['n_cells']=len(rows)
    collinearity={}
    for batch_id in ('pooled','batch1','batch2','batch3'):
        t=table if batch_id=='pooled' else table[table.batch_id==batch_id]
        complete=t[features].replace([np.inf,-np.inf],np.nan).dropna()
        pear=complete.corr(method='pearson')
        high=[]
        for i,name in enumerate(features):
            for other in features[i+1:]:
                r=pear.loc[name,other]
                if np.isfinite(r) and abs(r)>=.9: high.append({'a':name,'b':other,'pearson':r})
        nonconstant=[f for f in features if len(complete)>1 and complete[f].std(ddof=0)>0]
        X=complete[nonconstant].to_numpy(float)
        if X.size:
            Z=(X-X.mean(axis=0))/X.std(axis=0,ddof=0)
            singular=np.linalg.svd(Z,compute_uv=False)
            condition=float(singular.max()/singular.min()) if singular.min()>0 else np.inf
            rank=int(np.linalg.matrix_rank(Z))
        else: condition=np.nan; rank=0; singular=np.array([])
        vif={}
        for i,name in enumerate(nonconstant):
            z=Z[:,i]; others=np.delete(Z,i,axis=1)
            design=np.column_stack([np.ones(len(z)),others])
            residual=z-design@np.linalg.lstsq(design,z,rcond=None)[0]
            residual_fraction=float(residual@residual/(z@z))
            vif[name]={'value':float(1/residual_fraction) if residual_fraction>1e-12 else None,
                       'status':'finite' if residual_fraction>1e-12 else 'infinite_or_numerically_singular'}
        collinearity[batch_id]={'n_complete_cells':len(complete),'n_candidate_features':len(features),
                               'n_nonconstant':len(nonconstant),'standardized_matrix_rank':rank,
                               'standardized_condition_number':condition,'vif':vif,
                               'high_pairs_abs_pearson_ge_0_90':high,
                               'pearson_matrix':clean(pear.to_dict()),
                               'note':'Diagnostic only; all candidate features include deliberate algebraic redundancy. No final feature set selected or OLS prediction model fitted.'}
    return {'analysis_scope':'139 original cells retained; first 100 cycles used for candidate early signals; actual cycles 10 through 100 for summary candidates.',
            'q3':{'delta_q_definition':'Qdlin(actual cycle100,V)-Qdlin(actual cycle10,V)',
                  'grid_checks':grids,'group_definitions':{'long':'cycle_life > 1000','short':'cycle_life < 500','middle':'500 <= cycle_life <= 1000','unknown':'Nonfinite original cycle_life; retained and not assigned to a duration group'},
                  'descriptive_group_statistics':group_values,
                  'hypothesis_test':hypothesis_test(table),
                  'short_group_confounding':'All 28 short-life cells are from batch2; no short group in batch1 or batch3. Within-batch comparison of long versus short is available only for batch2 (long n=3, short n=28).'},
            'q5':{'feature_definitions':DEFINITIONS,'target':'original_cycle_life (cycles), without log target transformation',
                  'correlations':corr,'feature_missing_counts':missing,'multicollinearity':collinearity},
            'quality':quality,
            'target_nonfinite_counts':{b:int((~np.isfinite(table.loc[table.batch_id==b,'cycle_life'])).sum()) for b in COLORS},
            'limitations':['Correlations are exploratory associations, not causal effects or confirmed feature selection.',
                           'Several candidate statistics and correlations are inspected; no multiplicity-adjusted confirmatory claim from this exploration.',
                           'Within-batch samples are small (46,47,46), and batch2 has only 3 long-life cells.',
                           'Pooled correlations can reflect batch differences; within-batch estimates are reported.',
                           'Batch2 and Batch3 were examined in assignment-required EDA; neither can be described as completely unseen data.',
                           'Original zero and nonfinite source values are counted, not replaced. Per-feature finite values support descriptive means/slopes/correlations; all cells remain in the candidate table.',
                           'No outlier removal, interpolation, model training, tuning, or final feature selection performed.']}


def savefig(fig,path):
    fig.savefig(path,dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def plots(table,curves,result,out):
    figdir=out/'figures'; figdir.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'semibold',
                         'axes.labelcolor':'#24353E','text.color':'#24353E','xtick.color':'#24353E','ytick.color':'#24353E'})
    fig,axs=plt.subplots(1,3,figsize=(12,4.6),sharey=True)
    for ax,batch_id in zip(axs,COLORS):
        cc=curves[batch_id]
        for group,style in (('long','-'),('short','--')):
            g=[c for c in cc if c['life_group']==group]
            for c in g: ax.plot(c['voltage'],c['delta_q'],color=('#27647B' if group=='long' else '#D58037'),ls=style,lw=.75,alpha=.15)
            if g:
                ax.plot(g[0]['voltage'],np.mean([c['delta_q'] for c in g],axis=0),color=('#27647B' if group=='long' else '#D58037'),ls=style,lw=2.4,label=f'{group.capitalize()} (n={len(g)})')
        ax.axhline(0,color='#BDC6CB',lw=.7)
        ax.set_title(batch_id.replace('batch','Batch '));ax.set_xlabel('Voltage (V)');ax.grid(alpha=.16);ax.legend(frameon=False,fontsize=11)
        if not any(c['life_group']=='short' for c in cc): ax.text(.04,.06,'No short-life cells (<500)',transform=ax.transAxes,fontsize=11)
    axs[0].set_ylabel(r'$\Delta Q_{100-10}(V)$ (Ah)')
    fig.suptitle('Early voltage-capacity changes: original matching grid',fontsize=15,y=1.03)
    fig.text(.5,-.04,'Thin lines: individual cells; thick lines: group mean. Middle-life and unknown-life cells are outside this long/short comparison.',ha='center',fontsize=11)
    fig.tight_layout();savefig(fig,figdir/'q3_delta_q_curves.png')
    fig,axs=plt.subplots(1,3,figsize=(12,4.6),sharey=True)
    for ax,batch_id in zip(axs,COLORS):
        t=table[table.batch_id==batch_id]
        for i,group in enumerate(('short','middle','long','unknown')):
            y=t.loc[t.life_group==group,'delta_q_log10var'].to_numpy()
            if len(y):
                ax.scatter(i+np.linspace(-.13,.13,len(y)),y,color=COLORS[batch_id],alpha=.72,s=26)
                ax.plot([i-.20,i+.20],[np.median(y)]*2,color='#24353E',lw=2)
            ax.text(i,-5.25,f'n={len(y)}',ha='center',fontsize=11)
        ax.set_xticks(range(4),['Short','Middle','Long','Unknown']);ax.set_xlim(-.5,3.5);ax.set_ylim(-5.4,-2.4)
        ax.set_title(batch_id.replace('batch','Batch '));ax.grid(axis='y',alpha=.17)
    axs[0].set_ylabel(r'$\log_{10}[\mathrm{Var}(\Delta Q)]$ (Ah$^2$ convention)')
    fig.suptitle('Long versus short comparison is feasible only within Batch 2',fontsize=14,y=1.03)
    fig.text(.5,-.025,'Short <500; middle 500-1000; long >1000; unknown = nonfinite life. Black line: median.',ha='center',fontsize=11)
    fig.tight_layout();savefig(fig,figdir/'q3_logvariance_groups.png')
    fig,ax=plt.subplots(figsize=(8.5,5.2))
    for batch_id,color in COLORS.items():
        t=table[table.batch_id==batch_id]
        ax.scatter(t.delta_q_log10var,t.cycle_life,color=color,label=batch_id.replace('batch','Batch '),s=32,alpha=.75)
    ax.set_xlabel(r'$\log_{10}[\mathrm{Var}(\Delta Q)]$ (Ah$^2$ convention)');ax.set_ylabel('Original cycle life (cycles)')
    ax.set_title('Early capacity-change spread versus cycle life');ax.grid(alpha=.17);ax.legend(frameon=False)
    fig.tight_layout();savefig(fig,figdir/'q3_logvariance_life.png')
    test=result['q3']['hypothesis_test']
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),gridspec_kw={'width_ratios':[1,1.35]})
    b2=table[table.batch_id=='batch2']
    ys=[b2.loc[b2.life_group==g,'delta_q_log10var'].to_numpy() for g in ('long','short')]
    boxes=axs[0].boxplot(ys,positions=[0,1],widths=.42,patch_artist=True,showfliers=False)
    for patch,color in zip(boxes['boxes'],['#27647B','#D58037']):patch.set(facecolor=color,alpha=.25)
    for i,y in enumerate(ys):axs[0].scatter(i+np.linspace(-.12,.12,len(y)),y,s=32,color=('#27647B' if i==0 else '#D58037'),zorder=3)
    axs[0].set_xticks([0,1],['Long (>1000)\nn=3','Short (<500)\nn=28'])
    axs[0].set_ylabel(r'$\log_{10}[\mathrm{Var}(\Delta Q)]$');axs[0].set_title('Batch 2: one value per cell');axs[0].grid(axis='y',alpha=.17)
    null=np.asarray(test['permutation_null_distribution'])
    axs[1].hist(null,bins=38,color=COLORS['batch2'],alpha=.72,edgecolor='white')
    axs[1].axvline(test['difference_long_minus_short'],color='#24353E',lw=2.2,label='Observed mean difference')
    axs[1].set_xlabel('Mean(long) - mean(short)');axs[1].set_ylabel('Exact partitions');axs[1].set_title('Exact permutation reference (4495 partitions)');axs[1].legend(frameon=False,fontsize=11)
    fig.suptitle(f"Welch p={test['welch_p_two_sided']:.4g}; exact permutation p={test['permutation_p_two_sided']:.4g}",fontsize=14,y=1.03)
    fig.text(.5,-.04,'Two-sided alpha=0.05. Exploratory observational evidence; long-group n=3 and protocol confounding limit the claim.',ha='center',fontsize=11)
    fig.tight_layout();savefig(fig,figdir/'q3_test_result.png')

    c=pd.DataFrame(result['q5']['correlations'])
    names=list(DEFINITIONS)
    labels={name:name.replace('delta_q_log10var','log10 var Delta Q').replace('delta_q_','Delta Q ').replace('_',' ') for name in names}
    mat=c.pivot(index='feature',columns='scope',values='spearman').reindex(index=names,columns=['batch1','batch2','batch3','pooled'])
    fig,ax=plt.subplots(figsize=(8.5,8.2))
    im=ax.imshow(mat.to_numpy(),cmap='RdBu_r',vmin=-1,vmax=1,aspect='auto')
    ax.set_xticks(range(4),['Batch 1\nn=46','Batch 2\nn=39','Batch 3\nn=44','Pooled\nn=129']);ax.set_yticks(range(len(names)),[labels[f] for f in names])
    for i in range(len(names)):
        for j in range(4):
            r=mat.iloc[i,j]
            if np.isfinite(r):ax.text(j,i,f'{r:.2f}',ha='center',va='center',color='white' if abs(r)>.55 else '#24353E',fontsize=11)
    ax.set_title('Spearman correlation with original cycle life',pad=13)
    fig.colorbar(im,ax=ax,label='Spearman rho',shrink=.65)
    fig.text(.5,.015,'Exploratory candidates; all correlations use finite cell pairs. No final feature selection.',ha='center',fontsize=11)
    fig.tight_layout(rect=(0,.035,1,1));savefig(fig,figdir/'q5_target_correlations.png')
    selected=['qd10','qd100','delta_qd100_10','qd_slope10_100','ir10','ir100','delta_ir100_10','ir_mean10_100','temperature_mean10_100','charge_time_mean10_100','delta_q_min','delta_q_mean','delta_q_log10var','delta_q_range']
    t=table[table.batch_id=='batch1'][selected]
    cm=t.corr(method='pearson')
    fig,ax=plt.subplots(figsize=(11.3,10.0))
    im=ax.imshow(cm,cmap='RdBu_r',vmin=-1,vmax=1)
    labs=[labels[f] for f in selected]
    ax.set_xticks(range(len(selected)),labs,rotation=65,ha='right',fontsize=11)
    ax.set_yticks(range(len(selected)),labs,fontsize=11)
    ax.set_title('Batch 1 candidate redundancy: Pearson feature correlations',pad=12)
    for i in range(len(selected)):
        for j in range(len(selected)):
            if i!=j and abs(cm.iloc[i,j])>=.9:
                ax.text(j,i,f'{cm.iloc[i,j]:.2f}',ha='center',va='center',fontsize=9,color='white')
    fig.colorbar(im,ax=ax,label='Pearson r',shrink=.6)
    fig.tight_layout();savefig(fig,figdir/'q5_feature_redundancy.png')


def run(project_root: Path = ROOT):
    global ROOT
    ROOT = Path(project_root).resolve()
    out=ROOT/'outputs/day1';out.mkdir(parents=True,exist_ok=True)
    table,curves,quality,grids=collect()
    result=analysis(table,curves,quality,grids)
    plots(table,curves,result,out)
    table.to_csv(out/'early_feature_candidates.csv',index=False)
    (out/'q3_q5_results.json').write_text(json.dumps(clean(result),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('Cells retained:',len(table))
    print('Groups:',table.groupby(['batch_id','life_group']).size().to_dict())
    print('Global exact grid match:',grids['all_139_cells_exactly_equal'])
    c=pd.DataFrame(result['q5']['correlations'])
    for b in ('batch1','batch2','batch3','pooled'):
        top=c[c.scope==b].assign(abs_r=lambda x:x.spearman.abs()).sort_values('abs_r',ascending=False).head(5)
        print(b, top[['feature','n','pearson','spearman']].to_dict('records'))
    print('Output:',out)
    return clean(result)


def main():
    return run(ROOT)

if __name__=='__main__':main()
