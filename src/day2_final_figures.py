"""Render Day 2 report figures from saved results; never fit or predict."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

BLUE = '#315b92'
ORANGE = '#ca7438'
GREEN = '#3b856a'

def create_figures(root, result):
    root = Path(root)
    out = root/'outputs/day2'
    out.mkdir(parents=True, exist_ok=True)
    scores = result['candidate_scores']
    score_key = next(k for k in ['mean_mape_percent','mape_percent_mean','cv_mape_percent_mean','cv_mape_percent'] if k in scores.columns)
    best = scores.loc[scores.groupby('model_class')[score_key].idxmin()].sort_values(score_key)
    summary = result['summary']
    test = result['batch2_predictions']
    test = test[np.isfinite(test.y_true_cycle_life)].copy()
    cv_metric = summary['train_cv']['mape_percent_mean']
    hold_metric = summary['holdout']['mape_percent']
    test_metric = summary['batch2']['mape_percent']
    target = 9.1
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'axes.titleweight':'bold','savefig.dpi':170})
    figures = []

    fig, ax = plt.subplots(figsize=(8,4.6),layout='constrained')
    names = best.model_class.str.replace('LinearRegression','Linear regression',regex=False).str.replace('ElasticNet','Elastic Net',regex=False)
    bars = ax.barh(names, best[score_key], color=[GREEN,BLUE,ORANGE][:len(best)], height=.5)
    ax.invert_yaxis()
    ax.set(xlabel='Mean validation-fold MAPE (%)',title='Best setting of each model family\n69 configurations, fixed grouped 5-fold CV')
    ax.set_xlim(0,max(best[score_key])*1.25)
    for bar, value in zip(bars,best[score_key]):
        ax.text(value+.06,bar.get_y()+bar.get_height()/2,f'{value:.4f}%',va='center')
    ax.grid(axis='x',alpha=.18)
    path = out/'candidate_comparison.png'
    fig.savefig(path); plt.close(fig); figures.append(path)

    fig, axes = plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    values = [cv_metric,hold_metric,test_metric]
    bars = axes[0].bar(['CV mean\n35 development','Hold-out\n11 cells','Batch 2\n39 targets'],values,color=[BLUE,ORANGE,GREEN],width=.6)
    axes[0].axhline(target,color='#777777',ls='--',lw=1.2,label='Assignment target: 9.1% (Test comparison)')
    for bar,value in zip(bars,values):
        axes[0].text(bar.get_x()+bar.get_width()/2,value+.25,f'{value:.2f}%',ha='center')
    axes[0].set(ylabel='MAPE (%)',title='MAPE across evaluation stages')
    axes[0].set_ylim(0,max(values+[target])*1.25)
    axes[0].legend(loc='upper left',fontsize=8)
    axes[0].grid(axis='y',alpha=.18)
    lo = min(test.y_true_cycle_life.min(),test.y_pred_cycle_life.min())
    hi = max(test.y_true_cycle_life.max(),test.y_pred_cycle_life.max())
    span = hi-lo
    axes[1].plot([lo-.05*span,hi+.05*span],[lo-.05*span,hi+.05*span],color='#777777',ls='--',lw=1)
    axes[1].scatter(test.y_true_cycle_life,test.y_pred_cycle_life,s=37,c=GREEN,alpha=.8,edgecolors='white',linewidth=.5)
    axes[1].set(xlabel='Actual Cycle Life (cycles)',ylabel='Predicted Cycle Life (cycles)',title='Batch 2 predictions\nFit: all 46 Batch 1 cells')
    axes[1].grid(alpha=.18)
    path = out/'final_evaluation.png'
    fig.savefig(path); plt.close(fig); figures.append(path)

    fig, axes = plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    low_group = test.y_true_cycle_life < 534
    for mask, color, label in [(low_group,ORANGE,'Below Batch 1 life range'),(~low_group,BLUE,'Within Batch 1 life range')]:
        axes[0].scatter(test.loc[mask,'y_true_cycle_life'],test.loc[mask,'absolute_percentage_error_percent'],s=40,color=color,alpha=.8,label=label)
    for row in test.nlargest(3,'absolute_percentage_error_percent').itertuples():
        axes[0].annotate(f'cell {row.cell_id}',(row.y_true_cycle_life,row.absolute_percentage_error_percent),xytext=(5,5),textcoords='offset points',fontsize=8)
    axes[0].axvline(534,color='#777777',ls='--',lw=1)
    axes[0].set(xlabel='Actual Cycle Life (cycles)',ylabel='Absolute percentage error (%)',title='Batch 2: error versus actual life')
    axes[0].legend(fontsize=8); axes[0].grid(alpha=.18)
    signed = test.y_pred_cycle_life-test.y_true_cycle_life
    axes[1].hist(signed,bins=10,color=GREEN,alpha=.8,edgecolor='white')
    axes[1].axvline(0,color='#333333',lw=1)
    axes[1].set(xlabel='Prediction - actual life (cycles)',ylabel='Number of cells',title='Batch 2 signed error\nPositive: overprediction')
    axes[1].grid(axis='y',alpha=.18)
    path = out/'batch2_error_analysis.png'
    fig.savefig(path); plt.close(fig); figures.append(path)
    return figures
