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
    'lr':[0.988, 0.463, 0.416],
    'gbm':[0.357, 0.518, 0.694]
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
    
def plot_rel_ood_perf(
    artifacts_fpath,
    train_years,
    tasks,
    metrics,
    models,
    y_axis,
    model_colours=model_colours,
    baseline_label='09-12[ID]',
    ood_label_suffix='[OOD]',
    figsize=(12,5),
    legend_bbox_to_anchor=(0.5,-0.5),
    legend_ncols=5,
    save_path=None,
    save_res_dpi=300
    ):
    
    fig, axes = plt.subplots(nrows = len(metrics), ncols=len(tasks),figsize=figsize)

    for c,task in enumerate(tasks):
        for icolor,train_year in enumerate(train_years):
            for m,model in enumerate(models):
                
                df_eval = pd.read_csv(os.path.join(
                    artifacts_fpath,
                    task,
                    "eval",
                    f"{model}_{train_year}",
                    "by_group.csv"
                )).assign(task=task)

                df_eval['test_group']=df_eval['test_group'].astype(str)

                for r,metric in enumerate(metrics):

                    year_groups = list(set([
                        x for x in df_eval['test_group']
                        if x > max(train_year.split("_"))
                    ]))

                    year_groups.sort()

                    # baseline (0)
                    axes[r][c].plot(
                        [-0.5, len(year_groups)-0.5],
                        [0,0],
                        '--',
                        c = 'black',
                        zorder = 0,
                        label=f"{baseline_label}" if m==0 else None
                    )

                    temp = df_eval.query("metric==@metric and test_group==@year_groups")
                    data = temp.query("CI_quantile_95=='mid'").reset_index(drop=True)

                    data['CI_upper'] = (
                        temp.query("CI_quantile_95=='upper'").reset_index()['delta'] - 
                        temp.query("CI_quantile_95=='mid'").reset_index()['delta']
                    ).abs()

                    data['CI_lower'] = (
                        temp.query("CI_quantile_95=='lower'").reset_index()['delta'] - 
                        temp.query("CI_quantile_95=='mid'").reset_index()['delta']
                    ).abs()

                    # line
                    axes[r][c].plot(
                        data['test_group'],
                        data['delta'],
                        '-o',
                        linewidth=2,
                        color = model_colours[model],
                        label = f"{models[model]} {train_years[train_year]}{ood_label_suffix}"
                    )

                    # error bars
                    axes[r][c].errorbar(
                        data['test_group'],
                        data['delta'],
                        data[['CI_lower','CI_upper']].values.T,
                        zorder = 0,
                        linewidth = 1.5,
                        color = model_colours[model],
                    )

                    ## Axes settings
                    if icolor == 0:
                        axes[r][c].set_ylim(y_axis[metric]['lim'])
                        axes[r][c].yaxis.set_major_locator(MaxNLocator(nbins=4,prune='both'))
                        axes[r][c].grid(which='major', linewidth=0.5, axis='y')
                        if r==0:
                            axes[r][c].set_title(tasks[task])

                        if r==len(metrics)-1:
                            axes[r][c].set_xticks(year_groups)
                            axes[r][c].set_xticklabels([x[-2:] for x in year_groups])
                            axes[r][c].set_xlabel('Year Groups')
                        else:
                            axes[r][c].set_xticklabels('')
                            axes[r][c].set_xlabel('')
                            axes[r][c].tick_params(axis='x', length=0)

                        if c == 0:
                            axes[r][c].set_ylabel(f"Relative\n{y_axis[metric]['label']}")
                            axes[r][c].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                        else:
                            axes[r][c].set_yticklabels('')
                            axes[r][c].set_ylabel('')
                            #axes[r][c].spines['left'].set_color('white')
                            axes[r][c].tick_params(axis='y', length=0)

                    if r == len(metrics)-1 and c == len(tasks)-1:
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
    

def plot_raw_ood_perf(
    artifacts_fpath,
    train_years,
    tasks,
    metrics,
    models,
    y_axis,
    model_colours=model_colours,
    baseline_label='09-12[ID]',
    ood_label_suffix='[OOD]',
    ood_models_label='Year-Specific Model[ID]',
    legend_bbox_to_anchor=(0.5,-0.5),
    legend_ncols=5,
    figsize=(12,5),
    save_path=None,
    save_res_dpi=300,
    ):
    
    fig, axes = plt.subplots(nrows = len(metrics), ncols=len(tasks),figsize=figsize)
    
    for c,task in enumerate(tasks):
        for icolor,train_year in enumerate(train_years):
            for m,model in enumerate(models):
                
                df_eval_base = pd.read_csv(os.path.join(
                    artifacts_fpath,
                    task,
                    "eval",
                    f"{model}_{train_year}",
                    "by_group.csv"
                ))

                year_groups = list(set([
                    x for x in df_eval_base['test_group'] 
                    if x not in [train_year]
                    and x > max(train_year.split("_"))
                ]))

                year_groups.sort()

                for r,metric in enumerate(metrics):

                    # baseline (0)
                    if icolor == 0:
                        base_value = df_eval_base.query(
                            "metric==@metric and test_group==@train_year and CI_quantile_95=='mid'"
                        )['baseline'].values[0]

                        axes[r][c].plot(
                            [-0.5, len(year_groups)-0.5],
                            [base_value,base_value],
                            '--',
                            c=model_colours[model],
                            zorder=0,
                            label=f"{models[model]} {baseline_label}"
                        )

                    idf_eval_base = df_eval_base.query("test_group==@year_groups")

                    df_eval_id = pd.concat((
                        pd.read_csv(os.path.join(
                            artifacts_fpath,
                            task,
                            "eval",
                            f"{model}_{group}",
                            "by_group.csv"
                        )).assign(train_group=int(group)) for group in year_groups
                    ))
                    df_eval_id['test_group'] = df_eval_id['test_group'].astype(str)
                    df_eval_id['train_group'] = df_eval_id['train_group'].astype(str)
                    df_eval_id = df_eval_id.query("test_group==train_group")

                    # line
                    axes[r][c].plot(
                        idf_eval_base.query("CI_quantile_95=='mid' and metric==@metric and test_group != @train_year")['test_group'],
                        idf_eval_base.query("CI_quantile_95=='mid' and metric==@metric and test_group != @train_year")['comparator'],
                        '-o',
                        linewidth=2,
                        color = model_colours[model],
                        label=f"{models[model]} {train_years[train_year]}{ood_label_suffix}",
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

                        # error bars
                        axes[r][c].errorbar(
                            base['test_group'],
                            base['comparator'],
                            base[['CI_lower','CI_upper']].values.T,
                            zorder = 0,
                            linewidth = 1.5,
                            color = model_colours[model]
                        )
                        
                        if model=='lr':
                            # scatter
                            axes[r][c].scatter(
                                retrained['test_group'],
                                retrained['baseline'],
                                s = 20,
                                linewidth=2,
                                color = 'black',
                                label = f"{ood_models_label}" if cg==0 else None
                            )

                            # error bars
                            axes[r][c].errorbar(
                                retrained['test_group'],
                                retrained['baseline'],
                                retrained[['CI_lower','CI_upper']].values.T,
                                zorder = 0,
                                linewidth = 1.5,
                                color = 'black'
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
                            axes[r][c].set_xticklabels([x[-2:] for x in year_groups])
                            axes[r][c].set_xlabel('Year Groups')
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

                if r == len(metrics)-1 and c == len(tasks)-1:
                    leg = axes[r][c].legend(
                        bbox_to_anchor=legend_bbox_to_anchor,
                        ncol=legend_ncols,
                        title='Training Year Group',
                    )
                    #leg._legend_box.align='left'

                #sns.despine(offset=10, trim = True) 
    plt.show()
    
    
def plot_heatmap(
    artifacts_fpath,
    train_years,
    tasks,
    y_axis,
    models,
    metric={'name':'auc','label':'AUROC'},
    test_group=['2019','2020','2021'],
    dv='delta',
    cmap='magma',
    figsize=(13,4),
    save_path=None,
    save_res_dpi=300,
    ):
    
    train_year_labels=[v[:2]+'\n'+v[-2:] for k,v in train_years.items()]
    df_results = pd.DataFrame()
    
    ## read files
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

                df_eval['test_group']=df_eval['test_group'].astype(str)

                df_eval.loc[
                    df_eval['test_group']==train_year,'test_group'
                ]=train_year.split('_')[-1]

                df_results = pd.concat(( 
                    df_results,
                    df_eval.query(
                        "test_group==@test_group and \
                         metric==@metric['name'] and \
                         CI_quantile_95=='mid'\
                        "
                    )[['test_group',dv]].assign(
                        train_year=train_years[train_year],
                        metric=metric['name'],
                        task=task,
                        model=model
                    )
               )).reset_index(drop=True)

    ## plot
    fig, axes = plt.subplots(nrows = len(models), ncols=len(tasks),figsize=figsize)
    for c,task in enumerate(tasks):
        for r,model in enumerate(models):
            df = df_results.query("task==@task and model==@model")
            
            df.loc[
                (df['test_group'].str.slice(start=-2)<df['train_year'].str.slice(start=-2)),
                dv
            ]=np.nan
            
            if y_axis=='auto':
                vmin = df[dv].min()
                vmax = df[dv].max()
            else:
                vmin=y_axis[metric]['lim_raw' if dv=='comparator' else 'lim'][0]
                vmax=y_axis[metric]['lim_raw' if dv=='comparator' else 'lim'][1]
                
            df = df.pivot_table(values=dv,index='test_group',columns='train_year')
            df=df[[v for k,v in train_years.items()]]

            pos = axes[r][c].imshow(
                df.values,
                cmap=f"{cmap}" if metric['name'] in ['auc','auprc'] else f"{cmap}_r",
                vmin=vmin,
                vmax=vmax,
            )

            if c==len(tasks)-1 and y_axis!='auto':
                fig.colorbar(
                    pos,ax=axes[r][c],
                    label=metric['label'] if dv=='comparator' else f"Relative\n{metric['label']}",
                    fraction=0.046,
                    pad=0.04
                )
            else:
                fig.colorbar(
                    pos,ax=axes[r][c],
                    label=metric['label'] if dv=='comparator' else f"Relative\n{metric['label']}",
                    fraction=0.046,
                    pad=0.04
                )

            if r==0:
                axes[r][c].set_title(tasks[task])

            if r==len(models)-1:
                axes[r][c].set_xticks(np.arange(0,len(train_year_labels),1))
                axes[r][c].set_xticklabels(train_year_labels)
                axes[r][c].set_xlabel('Train Year Groups')
            else:
                axes[r][c].set_xticklabels('')
                axes[r][c].set_xlabel('')
                axes[r][c].tick_params(axis='x', length=0)


            axes[r][c].set_ylabel(f"{models[model]}\nEvaluation Year [OOD]")
            axes[r][c].set_yticks(np.arange(0,len(test_group),1))
            axes[r][c].set_yticklabels(test_group)

   
    #sns.despine(offset=10, trim = True) 
    if save_path is not None:
        plt.savefig(save_path, dpi=save_res_dpi)
    
    plt.tight_layout()
    plt.show()