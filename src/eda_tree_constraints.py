"""DAY 1 target-support diagnostic; no prediction model or training.

The range-constrained MAPE floor uses true targets as an oracle and is not a
model prediction or measured model performance. All original cells are kept.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = {'batch1': '#27647B', 'batch2': '#D58037', 'batch3': '#6B7A51'}


def run(project_root: Path) -> dict:
    root = Path(project_root).resolve()
    source = root/'outputs/day1/early_feature_candidates.csv'
    data = pd.read_csv(source)
    if data.duplicated(['batch_id','cell_id']).any():
        raise ValueError('Expected one record per original cell')
    b1 = data.loc[data.batch_id == 'batch1', 'cycle_life'].to_numpy(float)
    if not np.isfinite(b1).all() or (b1 <= 0).any():
        raise ValueError('Batch 1 target support requires finite positive labels')
    lo, hi = float(b1.min()), float(b1.max())
    summary = {}
    for name in COLORS:
        batch = data[data.batch_id == name]
        y_all = batch.cycle_life.to_numpy(float)
        finite_positive = np.isfinite(y_all) & (y_all > 0)
        y = y_all[finite_positive]
        oracle_error = np.abs(np.clip(y, lo, hi) - y) / y
        summary[name] = {
            'original_cell_count': len(batch),
            'finite_positive_label_count': len(y),
            'nonfinite_label_count': int((~np.isfinite(y_all)).sum()),
            'nonpositive_finite_label_count': int((np.isfinite(y_all) & (y_all <= 0)).sum()),
            'below_b1_full_label_min_count': int((y < lo).sum()),
            'above_b1_full_label_max_count': int((y > hi).sum()),
            'within_b1_full_label_range_count': int(((y >= lo) & (y <= hi)).sum()),
            'original_finite_label_min': float(y.min()),
            'original_finite_label_max': float(y.max()),
            'oracle_range_constrained_mape_lower_bound_percent': float(100*oracle_error.mean()),
            'below_range_cell_ids': batch.loc[finite_positive & (y_all < lo),'cell_id'].astype(int).tolist(),
            'above_range_cell_ids': batch.loc[finite_positive & (y_all > hi),'cell_id'].astype(int).tolist(),
            'nonfinite_label_cell_ids': batch.loc[~np.isfinite(y_all),'cell_id'].astype(int).tolist(),
        }
    result = {
        'status': 'DAY 1 descriptive structural diagnostic; no model fitted',
        'source_csv': str(source.relative_to(root)),
        'source_csv_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'raw_snapshot_reference': 'configs/data-snapshot.json',
        'raw_snapshots': json.loads((root/'configs/data-snapshot.json').read_text())['files'],
        'original_cells_retained': len(data),
        'target_definition': 'Original cycle_life field, without target transformation or correction.',
        'b1_full_label_support_cycles': [lo, hi],
        'batch_statistics': summary,
        'floor_formula': '100/n * sum(abs(clip(y_i, min_y_B1, max_y_B1)-y_i)/y_i) over finite positive evaluation labels.',
        'floor_interpretation': 'The true target is used as an oracle to choose the closest allowed output. This is the smallest possible MAPE of any predictor whose outputs are restricted to the full Batch 1 label interval; it is not a trained-model prediction, a model score, or an achievable score estimate.',
        'decision_tree_random_forest_scope': 'For ordinary squared-error regression trees, leaf outputs are means of training labels. RandomForestRegressor averages these tree predictions, so outputs stay inside the training-label interval. The full Batch 1 interval yields an optimistic bound; a smaller training subset can narrow it further.',
        'catboost_scope': 'CatBoost predicts scale * sum(tree leaf values) + bias, rather than averaging target-label means. Its output is not guaranteed to lie inside the training-label interval. This does not establish reliable extrapolation: numeric tree thresholds still define piecewise-constant regions and no unobserved linear trend is automatically continued.',
        'comparison_target': {'assignment_regression_mape_percent': 9.1,
                              'is_measured_model_result': False,
                              'caution': 'The assignment target and these original batches can differ in label corrections, exclusion rules and evaluation denominator; no reproduction claim.'},
        'limitations': ['The diagnostic preserves all 139 original rows. Nonfinite targets remain present but cannot contribute to a finite MAPE denominator.',
                       'The eventual missing-label handling and Day 2 evaluation denominator remain subject to user design decisions.',
                       'No target correction, interpolation, cell deletion, extra model, package installation, hyperparameter tuning or prediction fitting.',
                       'Batch 2 and Batch 3 target distributions were observed in assignment-required EDA. This is not a completely unseen test analysis.',
                       'The support floor applies only to predictors constrained to the stated label range. It is not a universal lower bound for all tree boosting methods.'],
        'official_references': {
            'tree_piecewise_constant_leaf_mean': 'https://scikit-learn.org/stable/modules/tree.html',
            'random_forest_average_predictions': 'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html',
            'catboost_sum_leaf_values_formula': 'https://catboost.ai/docs/en/concepts/python-reference_catboostregressor_get_scale_and_bias',
        },
    }
    figdir=root/'outputs/day1/figures'; figdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size':13, 'axes.spines.top':False, 'axes.spines.right':False,
                         'axes.labelcolor':'#24353E', 'text.color':'#24353E',
                         'xtick.color':'#24353E','ytick.color':'#24353E'})
    fig,ax=plt.subplots(figsize=(9.4,5.2))
    ax.axhspan(lo,hi,color='#27647B',alpha=.10,label=f'Batch 1 target support: {lo:.0f}-{hi:.0f}')
    ax.axhline(lo,color='#27647B',lw=1.1,ls='--');ax.axhline(hi,color='#27647B',lw=1.1,ls='--')
    for i,(name,color) in enumerate(COLORS.items()):
        batch=data[data.batch_id==name]
        batch=batch[np.isfinite(batch.cycle_life)]
        y=batch.cycle_life.to_numpy(float)
        xpos=i+np.linspace(-.15,.15,len(batch))
        ax.scatter(xpos,y,color=color,s=38,alpha=.76,zorder=3)
        outside=(y<lo)|(y>hi)
        ax.scatter(xpos[outside],y[outside],facecolors='none',edgecolors='#24353E',s=62,lw=1.0,zorder=4)
    labels=[]
    for name in COLORS:
        b=summary[name]
        labels.append(f"{name.replace('batch','Batch ')}\nfinite n={b['finite_positive_label_count']}\noutside: {b['below_b1_full_label_min_count']+b['above_b1_full_label_max_count']}")
    ax.set_xticks([0,1,2],labels);ax.set_xlim(-.5,2.5)
    ax.set_ylabel('Original cycle life (cycles)')
    ax.set_title('Target support differs across batches',fontsize=16,pad=12)
    ax.grid(axis='y',alpha=.16);ax.legend(frameon=False,loc='upper left',fontsize=12)
    fig.text(.5,.005,'Dark rings: outside Batch 1 target range. Diagnostic only; no model fitted.',ha='center',fontsize=12)
    fig.tight_layout(rect=(0,.035,1,1))
    fig.savefig(figdir/'model_target_range.png',dpi=300,bbox_inches='tight',facecolor='white');plt.close(fig)
    (root/'outputs/day1/tree_constraints.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({name: summary[name] for name in ('batch2','batch3')},ensure_ascii=False))
    return result


if __name__=='__main__':
    run(Path(__file__).resolve().parents[1])
