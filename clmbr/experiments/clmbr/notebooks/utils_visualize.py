import os
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

models_dict = {
    "gru_lr": {
        'label':'GRU\nLR',
        'colour':sns.color_palette("Paired")[1],
    },
    "transformer_lr": {
        'label':'Transformer\nLR',
        'colour':sns.color_palette("Paired")[3],
    },
    "count_lr": {
        'label':'Count\nLR',
        'colour':sns.color_palette("Paired")[5],
    },
    "gru_gbm": {
        'label':'GRU\nGBM',
        'colour':sns.color_palette("Paired")[0],
    },
    "transformer_gbm": {
        'label':'Transformer\nGBM',
        'colour':sns.color_palette("Paired")[2],
    },
    "count_gbm": {
        'label':'Count\nGBM',
        'colour':sns.color_palette("Paired")[4],
    },
} 

def get_result_table(artifacts_fpath,train_years,tasks,metrics,models,encoders):
    
    df_results = pd.DataFrame(columns = ['Task','Train Group','Performance','Metric','Encoder','Model'])

    c = 0
    
    for task in tasks:
        for train_year in train_years:
            for encoder in encoders:
                for model in models:

                    df_eval = pd.read_csv(os.path.join(
                        artifacts_fpath,
                        task,
                        "eval",
                        f"{encoder}_{model}_{train_year}",
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
                        df_results.loc[c,'Encoder']=encoder
                        df_results.loc[c,'Model']=model
                        
    df_results = df_results.replace({**tasks,**metrics,**models,**encoders})
    df_results = df_results.pivot(index=['Task','Metric'],columns=['Train Group','Encoder','Model'],values='Performance')
    df_results = df_results.reindex(labels = ['In-Hospital Mortality','LOS > 7 Days','Readmission in 30 Days','Admission to ICU'], level=0)
    df_results = df_results.reindex(labels = ['AUROC','AUPRC','Calibration'],level=1)
    
    return df_results
    
def plot_rel_ood_perf(
    artifacts_fpath,
    train_years,
    tasks,
    metrics,
    models,
    encoders,
    y_axis,
    model_colours=model_colours,
    baseline_label='09-12[ID]',
    ood_label_suffix='[OOD]',
    figsize=(12,5),
    legend_bbox_to_anchor=(0.5,-0.5),
    legend_ncols=5,
    save_path=None,
    save_res_dpi=300,
    save_format='png',
    ):
    
    for encoder in encoders:
        fig, axes = plt.subplots(nrows = len(metrics), ncols=len(tasks),figsize=figsize)

        for c,task in enumerate(tasks):
            for icolor,train_year in enumerate(train_years):
                for m,model in enumerate(models):

                    df_eval = pd.read_csv(os.path.join(
                        artifacts_fpath,
                        task,
                        "eval",
                        f"{encoder}_{model}_{train_year}",
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
                                title='Model',
                            )
                        #leg._legend_box.align='left'

        #sns.despine(offset=10, trim = True) 
        if save_path is not None:
            plt.savefig(f"{save_path}/{encoder}.{save_format}", dpi=save_res_dpi)

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
                        title='Model',
                    )
                    #leg._legend_box.align='left'

                #sns.despine(offset=10, trim = True) 
    plt.show()
    

def plot_change_rel_ood_perf(
    artifacts_fpath,
    tasks,
    metrics,
    models,
    y_axis,
    point_size=50,
    plot_significance=False,
    train_years = [2009,2010,2011,2012],
    model_colours=model_colours,
    baseline_label='09-12[ID]',
    ood_label_suffix='[OOD]',
    legend_bbox_to_anchor=(0.5,-0.5),
    legend_ncols=3,
    figsize=(12,5),
    save_path=None,
    save_res_dpi=300,
    ):
    
    train_group_str = '_'.join([str(x) for x in train_years])
    df_eval = pd.DataFrame()
    
    for task in tasks:
        for model in models:
            
            fpath = os.path.join(
                artifacts_fpath,
                task,
                "eval",
                '_'.join([
                    model,
                    '_'.join([str(x) for x in train_years]),
                ])
            )

            df_eval = pd.concat((
                df_eval,
                pd.read_csv(f"{fpath}/by_group_comp_rel_ood_with_base.csv")
            ))

    df_eval['change_rel_ood']=(
        df_eval['performance_clmbr_delta']
        -
        df_eval['performance_base_delta']
    )

    df_eval_ci = (
        df_eval
        .groupby(['task','model','metric','test_group'])['change_rel_ood']
        .quantile([.025,.5,.975])
        .unstack()
        .reset_index()
    )
    
    df_eval_ci['sig'] = df_eval_ci[0.025]*df_eval_ci[0.975]>0

    fig,axes=plt.subplots(
        len(metrics),
        len(tasks),
        figsize=figsize
    )

    for c,task in enumerate(tasks):
        for r,metric in enumerate(metrics):

            exclude_test_groups = [train_group_str] + [str(x) for x in train_years]
            
            year_groups = [
                x for x in df_eval_ci['test_group'].unique().tolist() 
                if x not in exclude_test_groups
            ]
            
            axes[r][c].plot(
                [0, len(year_groups)-1],
                [0,0],
                'k--',
            )

            for model in models:
                
                data = df_eval_ci.query(
                    "task==@task and metric==@metric and model==@model and test_group==@year_groups"
                ).reset_index(drop=True)

                data[0.025] = np.abs(data[0.025] - data[0.5])
                data[0.975] = np.abs(data[0.975] - data[0.5])

                axes[r][c].scatter(
                    data['test_group'],
                    data[0.5],
                    s = point_size,
                    label=models[model],
                    color = model_colours[model],
                )
                
                if plot_significance: 
                    sig_data = data.query("sig==True")
                    axes[r][c].scatter(
                        sig_data['test_group'],
                        sig_data[0.5],
                        s= point_size*1.8,
                        color='black',
                        zorder=0,
                    )

                axes[r][c].errorbar(
                    data['test_group'],
                    data[0.5],
                    data[[0.025,0.975]].values.T,
                    zorder=0,
                    color = model_colours[model],
                )

                ## axes setting
                axes[r][c].set_ylim(y_axis[metric][task])
                axes[r][c].tick_params(axis='y', length=0)
                axes[r][c].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
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
                    axes[r][c].set_ylabel(f"Change in Rel.\n{y_axis[metric]['label']}")

                else:
                    axes[r][c].set_ylabel('')


            if r == len(metrics)-1 and c == len(tasks)-1:
                leg = axes[r][c].legend(
                    bbox_to_anchor=legend_bbox_to_anchor,
                    ncol=legend_ncols,
                    title='Model',
                )

    #plt.tight_layout()
    plt.show()
    

def plot_change_rel_ood_int(
    artifacts_fpath,
    tasks,
    metrics,
    models,
    encoders,
    y_axis,
    train_years=[2009,2010,2011,2012],
    model_colours=models_dict,
    add_swarmplot=True,
    swarmplot_sample_frac=0.2,
    swarmplot_size=1,
    swarmplot_color=".3",
    boxplot_alpha=0.8,
    hatch_pattern='//////',
    hatch_alpha=0.3,
    legend_bbox_to_anchor=(0.5,-0.5),
    legend_ncols=5,
    figsize=(12,5),
    save_path=None,
    save_res_dpi=300,
    ):
    
    train_group_str = '_'.join([str(x) for x in train_years])
    df_eval = pd.DataFrame()
    exclude_test_groups = [train_group_str] + [str(x) for x in train_years]

    for task in tasks:
        for encoder in encoders:
            for model in models:

                fpath = os.path.join(
                    artifacts_fpath,
                    task,
                    "eval",
                    '_'.join([
                        encoder,
                        model,
                        train_group_str,
                    ])
                )

                df = pd.read_csv(f"{fpath}/by_group_comp_rel_ood_with_base.csv")

                if encoder=='gru':

                    df = pd.concat((
                        (
                            df[['task','metric','model','boot_num','test_group']]
                            .assign(
                                encoder=encoder,
                                performance=df['performance_clmbr'],
                                performance_base=df['performance_clmbr_baseline'],
                                performance_delta=df['performance_clmbr_delta'],
                            )
                        ),
                        (
                            df[['task','metric','model','boot_num','test_group']]
                            .assign(
                                encoder='count',
                                performance=df['performance_base'],
                                performance_base=df['performance_base_baseline'],
                                performance_delta=df['performance_base_delta'],
                            )
                        ),
                    ))

                else: 

                    df = (
                        df[['task','metric','model','boot_num','test_group']]
                        .assign(
                            encoder=encoder,
                            performance=df['performance_clmbr'],
                            performance_base=df['performance_clmbr_baseline'],
                            performance_delta=df['performance_clmbr_delta'],
                        )
                    )


                df_eval = pd.concat((
                    df_eval,
                    df
                ))

    # get integral
    results = (
        df_eval
        .query("test_group!=@exclude_test_groups")
        .groupby(['task','encoder','model','metric','boot_num'])
        .agg({
            'performance_delta': lambda x: np.trapz(x),
        })
        .reset_index()
    )

    results['model_combination']=results['encoder']+'_'+results['model']
    
    model_combination = [
        x for x in model_colours.keys()
        if x in results['model_combination'].unique()
    ]

    fig,axes=plt.subplots(
        len(metrics),
        len(tasks),
        figsize=figsize
    )

    for c,task in enumerate(tasks):
        for r,metric in enumerate(metrics):

            data = results.query(
                "task==@task and metric==@metric"
            )

            axes[r][c].plot(
                [-1,len(model_combination)+1],
                [0,0],
                'k--',
                zorder=0
            )

            sns.boxplot(
                data = data,
                x = "model_combination",
                y = "performance_delta",
                order = model_combination,
                palette = sns.set_palette(
                    [model_colours[x]['colour'] for x in model_colours.keys()]
                ),
                ax=axes[r][c],
                width=0.5,
                linewidth=1,
                fliersize=0,
                boxprops=dict(alpha=boxplot_alpha),
            )
            
            if add_swarmplot:
                sns.swarmplot(
                    data = data.groupby(
                        'model_combination'
                    ).sample(frac=swarmplot_sample_frac),
                    x = "model_combination",
                    #hue = "encoder",
                    y = "performance_delta",
                    order = model_combination,
                    size = swarmplot_size,
                    ax=axes[r][c],
                    zorder=0,
                    color=swarmplot_color,
                )

            # add hatch
            rect = patches.Rectangle(
                (-1,-2) if metric in ['auc','auprc','auprc_c'] else (-1,0),
                len(model_combination)+1,
                2,
                linewidth=0,
                edgecolor='grey',
                facecolor='None',
                hatch=hatch_pattern,
                alpha=hatch_alpha,
                zorder=0
            )
            axes[r][c].add_patch(rect)

            ## axes setting
            axes[r][c].set_ylim(y_axis[metric][task])
            axes[r][c].tick_params(axis='y', length=0)
            axes[r][c].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            axes[r][c].yaxis.set_major_locator(MaxNLocator(nbins=4,prune='both'))
            axes[r][c].grid(which='major', linewidth=0.5, axis='y')

            if r==0:
                axes[r][c].set_title(tasks[task])

            if r==len(metrics)-1:
                axes[r][c].set_xticks(list(range(0,len(model_combination),1)))
                axes[r][c].set_xticklabels([
                    model_colours[x]['label'] 
                    for x in model_colours.keys()
                    if x in model_combination
                ])
                axes[r][c].set_xlabel('Model')
            else:
                axes[r][c].set_xticklabels('')
                axes[r][c].set_xlabel('')
                axes[r][c].tick_params(axis='x', length=0)

            if c == 0:
                axes[r][c].set_ylabel(f"Integral of Rel.\n{y_axis[metric]['label']}")

            else:
                axes[r][c].set_ylabel('')

    if save_path is not None:
        plt.savefig(save_path, dpi=save_res_dpi)
        
    plt.show()
    
    results = results.pivot(
        index=['task','model','metric','boot_num'],
        columns=['encoder'],
        values=['performance_delta']
    ).reset_index()
    
    results.columns = [
        x[0] if x[0]!='performance_delta' else x[1]
        for x in results.columns
    ]
    
    use_metrics = list(y_axis.keys())
    sig_results = pd.DataFrame()
    
    if 'gru' in encoders.keys():
        results = results.assign(gru_count=results['gru']-results['count'])
        results_ci = (
            results
            .groupby(['task','model','metric'])['gru_count']
            .quantile([.025,.5,.975])
            .unstack()
            .reset_index()
        )
        results_ci['sig']=results_ci[0.025]*results_ci[0.975]>0
        results_ci.query("sig==1 and metric==@use_metrics")
        results_ci['comparison'] = 'GRU_Count'
        sig_results = pd.concat((sig_results, results_ci))

    if 'transformer' in encoders.keys():
        results = results.assign(transformer_count=results['transformer']-results['count'])
        results_ci = (
            results
            .groupby(['task','model','metric'])['transformer_count']
            .quantile([.025,.5,.975])
            .unstack()
            .reset_index()
        )
        results_ci['sig']=results_ci[0.025]*results_ci[0.975]>0
        results_ci.query("sig==1 and metric==@use_metrics")
        results_ci['comparison'] = 'Transformer_Count'
        sig_results = pd.concat((sig_results, results_ci))

    if 'gru' in encoders.keys() and 'transformer' in encoders.keys():
        results = results.assign(gru_transformer=results['gru']-results['transformer'])
        results_ci = (
            results
            .groupby(['task','model','metric'])['gru_transformer']
            .quantile([.025,.5,.975])
            .unstack()
            .reset_index()
        )
        results_ci['sig']=results_ci[0.025]*results_ci[0.975]>0
        results_ci.query("sig==1 and metric==@use_metrics")
        results_ci['comparison'] = 'GRU_Transformer'
        sig_results = pd.concat((sig_results, results_ci))
    return sig_results
    
    
    
def plot_heatmap_ood(
    artifacts_fpath,
    tasks,
    y_axis,
    encoders,
    models,
    metrics,
    models_dict=models_dict,
    train_years=[2009,2010,2011,2012],
    dv='performance_delta', #performance / performance_delta
    cmap='magma',
    figsize=(13,4),
    save_path=None,
    save_res_dpi=300,
    ):
    
    train_years_str = [str(x) for x in train_years]
    train_years_joined = "_".join(train_years_str)
    
    ## read files
    df_eval = pd.DataFrame()
    for c,task in enumerate(tasks):
        for encoder in encoders:
            for m,model in enumerate(models):

                df = pd.read_csv(os.path.join(
                    artifacts_fpath,
                    task,
                    "eval",
                    "_".join([
                        encoder,
                        model,
                        train_years_joined
                    ]),
                    "by_group_comp_rel_ood_with_base.csv"
                )).assign(task=task)

                df = df.query("test_group!=@train_years_str and test_group!=@train_years_joined")

                if encoder=='gru':

                    df = pd.concat((
                        (
                            df[['task','metric','model','boot_num','test_group']]
                            .assign(
                                encoder=encoder,
                                performance=df['performance_clmbr'],
                                performance_base=df['performance_clmbr_baseline'],
                                performance_delta=df['performance_clmbr_delta'],
                            )
                        ),
                        (
                            df[['task','metric','model','boot_num','test_group']]
                            .assign(
                                encoder='count',
                                performance=df['performance_base'],
                                performance_base=df['performance_base_baseline'],
                                performance_delta=df['performance_base_delta'],
                            )
                        ),
                    ))

                else: 

                    df = (
                        df[['task','metric','model','boot_num','test_group']]
                        .assign(
                            encoder=encoder,
                            performance=df['performance_clmbr'],
                            performance_base=df['performance_clmbr_baseline'],
                            performance_delta=df['performance_clmbr_delta'],
                        )
                    )


                df_eval = pd.concat((
                    df_eval,
                    df
                ))
                
    df_eval = (
        df_eval
        .groupby(['task','metric','model','test_group','encoder'])['performance_delta']
        .quantile(0.5)
        .reset_index()
    )
    
    ## plot
    fig, axes = plt.subplots(nrows = len(metrics), ncols=len(tasks),figsize=figsize)
    for c,task in enumerate(tasks):
        for r,metric in enumerate(metrics):
            df = df_eval.query("task==@task and metric==@metric")
            
            df = df.pivot(
                index=['model','encoder'], 
                columns='test_group',
                values=dv
            )
            
            data = df.values
            
            # x & y ticks 
            df = df.reset_index()
            df['model_combination'] = df['encoder']+'_'+df['model']
            xlabels = [x[-2:] for x in df_eval['test_group'].unique()]
            ylabels = [models_dict[x]['label'] for x in df['model_combination']]
            
            if y_axis=='auto':
                vmin = data.min()
                vmax = data.max()
            else:
                vmin=y_axis[metric][task]['lim_raw' if dv=='performance' else 'lim'][0]
                vmax=y_axis[metric][task]['lim_raw' if dv=='performance' else 'lim'][1]

            pos = axes[r][c].imshow(
                data,
                cmap=f"{cmap}" if metric in ['auc','auprc','auprc_c'] else f"{cmap}_r",
                vmin=vmin,
                vmax=vmax,
            )

            fig.colorbar(
                pos,ax=axes[r][c],
                label=None,
                fraction=0.046,
                pad=0.04
            )
            
            axes[r][c].plot(
                [-0.5,len(xlabels)-0.5],
                [len(encoders)+0.5,len(encoders)+0.5],
                'k-'
            )

            if r==0:
                axes[r][c].set_title(tasks[task])

            if r==len(metrics)-1:
                axes[r][c].set_xticks(np.arange(0,len(xlabels),1))
                axes[r][c].set_xticklabels(xlabels)
                axes[r][c].set_xlabel('Evaluation Year')
            else:
                axes[r][c].set_xticklabels('')
                axes[r][c].set_xlabel('')
                axes[r][c].tick_params(axis='x', length=0)

            if c==0:    
                axes[r][c].set_ylabel(f"Relative\n{metrics[metric]}")
                axes[r][c].set_yticks(np.arange(0,len(ylabels),1))
                axes[r][c].set_yticklabels(ylabels)
            else:
                axes[r][c].set_yticklabels('')
                axes[r][c].set_ylabel('')
                axes[r][c].tick_params(axis='y', length=0)
                
    if save_path is not None:
        plt.savefig(save_path, dpi=save_res_dpi)
    
    plt.tight_layout()
    plt.show()