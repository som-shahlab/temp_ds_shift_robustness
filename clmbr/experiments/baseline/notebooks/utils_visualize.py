import os
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from prediction_utils.pytorch_utils.metrics import StandardEvaluator
from matplotlib.ticker import MaxNLocator
from matplotlib.pyplot import savefig
from matplotlib.ticker import FormatStrFormatter

from scipy.stats import pearsonr

model_colours={
    'lr':[0, 0, 0],
    'gbm':[0, 0, 0]
}

def get_result_table(artifacts_fpath,train_years,tasks,metrics,models):
    
    df_results = pd.DataFrame(columns = ['Task','Train Group','Performance','Metric','Model'])

    c = 0
    
    for task in tasks:
        for train_year in train_years:
            for model in models:
                
                df_eval = pd.read_csv(os.path.join(
                    artifacts_fpath,
                    task,
                    "eval",
                    f"{model}_{train_year}",
                    "by_group.csv"
                ))
            
                df_eval=df_eval.round(3)
            
                df_eval['test_group']=df_eval['test_group'].astype(str)
            
                for metric in metrics:
                    
                    c+=1
                    
                    mid = df_eval.query(
                        f"metric==@metric and CI_quantile_95==['mid'] and test_group==@train_year"
                    )['baseline'].values[0]
                    
                    lower = df_eval.query(
                        "metric==@metric and CI_quantile_95==['lower'] and test_group==@train_year"
                    )['baseline'].values[0]
                    
                    upper = df_eval.query(
                        "metric==@metric and CI_quantile_95==['upper'] and test_group==@train_year"
                    )['baseline'].values[0]
                    
                    df_results.loc[c,'Performance'] = f"{mid} ({lower}, {upper})"
                    df_results.loc[c,'Metric'] = metric
                    df_results.loc[c,'Task'] = task
                    df_results.loc[c,'Train Group'] = train_years[train_year]
                    df_results.loc[c,'Model']=model
                    
    df_results = df_results.replace({**tasks,**metrics,**models})
    df_results = df_results.pivot(index=['Task','Metric'],columns=['Train Group','Model'],values='Performance')
    df_results = df_results.reindex(labels = ['In-Hospital Mortality','LOS > 7 Days','Readmission in 30 Days','Admission to ICU'], level=0)
    df_results = df_results.reindex(labels = ['AUROC','AUPRC','Calibration'],level=1)
    
    return df_results
    

def plot_raw_ood_perf(
    artifacts_fpath,
    train_years,
    tasks,
    metrics,
    models,
    y_axis,
    model_colours=model_colours,
    baseline_label='',
    ood_label_suffix='',
    ood_models_label='Oracle Models',
    plot_oracles=True,
    plot_legend=True,
    legend_bbox_to_anchor=(0.5,-0.5),
    legend_ncols=5,
    figsize=(12,5),
    save_path=None,
    save_res_dpi=300,
    ):
    
    fig, axes = plt.subplots(nrows = len(metrics), ncols=len(tasks),figsize=figsize)
    
    for c,task in enumerate(tasks):
        for icolor,train_year in enumerate(train_years):
            
            train_year_val = train_years[train_year]
            
            for m,model in enumerate(models):
                df_eval_base = pd.read_csv(os.path.join(
                    artifacts_fpath,
                    task,
                    "eval",
                    f"{model}_{train_year}",
                    "by_group.csv"
                ))
                
                df_eval_base = df_eval_base.replace(train_years)

                year_groups = list(set([
                    x for x in df_eval_base['test_group'] 
                    if str(x) not in train_year.split("_")
                ]))

                year_groups.sort()
                
                for r,metric in enumerate(metrics):

                    # baseline (0)
                    if icolor == 0:
                        
                        base_value = df_eval_base.query(
                            "metric==@metric and test_group==@train_year_val and CI_quantile_95=='mid'"
                        )['baseline'].values[0]

                        axes[r][c].plot(
                            [-0.5, len(year_groups)-0.5],
                            [base_value,base_value],
                            '--',
                            c=model_colours[model],
                            zorder=0,
                        )

                    idf_eval_base = df_eval_base.query("test_group==@year_groups")
                    
                    if plot_oracles:
                        df_eval_id = pd.concat((
                            pd.read_csv(os.path.join(
                                artifacts_fpath,
                                task,
                                "eval",
                                f"{model}_{group}",
                                "by_group.csv"
                            )).assign(train_group=int(group)) 
                            for group in year_groups
                            if group != train_year_val
                        ))
                        df_eval_id['test_group'] = df_eval_id['test_group'].astype(str)
                        df_eval_id['train_group'] = df_eval_id['train_group'].astype(str)
                        df_eval_id = df_eval_id.query("test_group==train_group")

                    # line
                    axes[r][c].plot(
                        idf_eval_base.query("CI_quantile_95=='mid' and metric==@metric")['test_group'],
                        idf_eval_base.query("CI_quantile_95=='mid' and metric==@metric")['comparator'],
                        '-o',
                        linewidth=2,
                        color = model_colours[model],
                        label=f'{models[model]} [{train_year_val}]'
                    )
                    
                    h_hatch_1 = axes[r][c].fill_between(
                        idf_eval_base.query("CI_quantile_95=='mid' and metric==@metric")['test_group'],
                        idf_eval_base.query("CI_quantile_95=='mid' and metric==@metric")['comparator'],
                        idf_eval_base.query("test_group==@train_year_val and CI_quantile_95=='mid' and metric==@metric")['comparator'].values,
                        where=(
                            idf_eval_base.query("CI_quantile_95=='mid' and metric==@metric")['comparator']
                            <
                            idf_eval_base.query("CI_quantile_95=='mid' and metric==@metric")['baseline']
                            if metric!='ace_abs_logistic_logit' 
                            else 
                            idf_eval_base.query("CI_quantile_95=='mid' and metric==@metric")['comparator']
                            >
                            idf_eval_base.query("CI_quantile_95=='mid' and metric==@metric")['baseline']
                        ),
                        facecolor='none',
                        hatch='//////',
                        edgecolor='red',
                        linewidth=0,
                        alpha=0.4,
                        interpolate=True,
                        zorder=0,
                    )
                    
                    
                    h_hatch_2 = axes[r][c].fill_between(
                        idf_eval_base.query("CI_quantile_95=='mid' and metric==@metric")['test_group'],
                        idf_eval_base.query("CI_quantile_95=='mid' and metric==@metric")['comparator'],
                        idf_eval_base.query("test_group==@train_year_val and CI_quantile_95=='mid' and metric==@metric")['comparator'].values,
                        where=(
                            idf_eval_base.query("CI_quantile_95=='mid' and metric==@metric")['comparator']
                            >
                            idf_eval_base.query("CI_quantile_95=='mid' and metric==@metric")['baseline']
                            if metric!='ace_abs_logistic_logit' 
                            else 
                            idf_eval_base.query("CI_quantile_95=='mid' and metric==@metric")['comparator']
                            <
                            idf_eval_base.query("CI_quantile_95=='mid' and metric==@metric")['baseline']
                        ),
                        facecolor='none',
                        hatch='XXXXX',
                        edgecolor='green',
                        linewidth=0,
                        alpha=0.4,
                        interpolate=True,
                        zorder=0,
                    )
                    
                    for cg,group in enumerate(year_groups):
                        base = idf_eval_base.query("test_group==@group and metric==@metric and CI_quantile_95=='mid'")[
                            ['test_group','comparator']
                        ]

                        base['CI_upper'] = np.abs(
                            idf_eval_base.query("test_group==@group and metric==@metric and CI_quantile_95=='upper'")['comparator'].values - 
                            idf_eval_base.query("test_group==@group and metric==@metric and CI_quantile_95=='mid'")['comparator'].values
                        )

                        base['CI_lower'] = np.abs(
                            idf_eval_base.query("test_group==@group and metric==@metric and CI_quantile_95=='lower'")['comparator'].values - 
                            idf_eval_base.query("test_group==@group and metric==@metric and CI_quantile_95=='mid'")['comparator'].values
                        )

                        # error bars
                        axes[r][c].errorbar(
                            base['test_group'],
                            base['comparator'],
                            base[['CI_lower','CI_upper']].values.T,
                            zorder = 0,
                            linewidth = 1.5,
                            color = model_colours[model]
                        )
                        
                        if plot_oracles:
                            
                            retrained = df_eval_id.query("test_group==@group and metric==@metric and CI_quantile_95=='mid'")[
                                ['test_group','baseline']
                            ]

                            retrained['CI_upper'] = np.abs(
                                df_eval_id.query("test_group==@group and metric==@metric and CI_quantile_95=='upper'")['baseline'].values - 
                                df_eval_id.query("test_group==@group and metric==@metric and CI_quantile_95=='mid'")['baseline'].values
                            )

                            retrained['CI_lower'] = np.abs(
                                df_eval_id.query("test_group==@group and metric==@metric and CI_quantile_95=='lower'")['baseline'].values - 
                                df_eval_id.query("test_group==@group and metric==@metric and CI_quantile_95=='mid'")['baseline'].values
                            )
                            
                            # scatter
                            axes[r][c].scatter(
                                retrained['test_group'],
                                retrained['baseline'],
                                s = 30,
                                linewidth=1,
                                color = 'grey',
                                edgecolors='k',
                                label = f"{ood_models_label}" if cg==0 else None
                            )

                            # error bars
                            axes[r][c].errorbar(
                                retrained['test_group'],
                                retrained['baseline'],
                                retrained[['CI_lower','CI_upper']].values.T,
                                zorder = 0,
                                linewidth = 1.5,
                                color = 'k'
                            )
                    
                    ## Axes settings
                    #print(year_groups)
                    if train_year == '2009_2010_2011_2012':
                        axes[r][c].set_ylim(y_axis[metric]['lim_raw'])
                        axes[r][c].yaxis.set_major_locator(MaxNLocator(nbins=4,prune='both'))
                        axes[r][c].grid(which='major', linewidth=0.5, axis='y')
                        if r==0:
                            axes[r][c].set_title(tasks[task])

                        if r==len(metrics)-1:
                            axes[r][c].set_xticks(year_groups)
                            axes[r][c].set_xticklabels([train_year_val]+[x[-2:] for x in year_groups if x != train_year_val])
                            axes[r][c].set_xlabel('Evaluation Years')
                        else:
                            axes[r][c].set_xticklabels('')
                            axes[r][c].set_xlabel('')
                            axes[r][c].tick_params(axis='x', length=0)

                        if c == 0:
                            axes[r][c].set_ylabel(f"{y_axis[metric]['label']}")
                            axes[r][c].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                        else:
                            axes[r][c].set_yticklabels('')
                            axes[r][c].set_ylabel('')
                            #axes[r][c].spines['left'].set_color('white')
                            axes[r][c].tick_params(axis='y', length=0)

                if r == len(metrics)-1 and c == len(tasks)-1 and plot_legend:
                    leg = axes[r][c].legend(
                        bbox_to_anchor=legend_bbox_to_anchor,
                        ncol=legend_ncols,
                        title='Training Year Group',
                    )
                    #leg._legend_box.align='left'

                #sns.despine(offset=10, trim = True) 
    if save_path is not None:
        plt.savefig(save_path, dpi=save_res_dpi)
    plt.show()