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
    

def plot_perf_grouped(
    artifacts_fpath,
    tasks,
    metrics,
    encoders,
    y_axis,
    clmbr_artifacts_fpath='/labs/shahlab/projects/lguo/temp_ds_shift_robustness/clmbr/experiments/clmbr/clmbr_artifacts/eval_end_to_end',
    ete_models = ['gru','transformer'],
    train_years=[2009,2010,2011,2012],
    groups = {
        '2009_2010_2011_2012':'09-12',
        '2013_2014_2015_2016':'13-16',
        '2017_2018_2019_2020_2021':'17-21',
    },
    model = 'lr',
    figsize = (18,6),
    save_path=None,
    save_res_dpi=300,
    legend_bbox_to_anchor=(0.8,-0.5),
    legend_ncols=1,
    fill_alpha=0.05,
    plot_major=False,
    ):
    
    df_eval = pd.DataFrame()
    
    train_years_str = '_'.join([str(x) for x in train_years])
    train_years_abbrev = '-'.join([
        train_years_str.split('_')[0][-2:],
        train_years_str.split('_')[-1][-2:],
    ])
    
    # count-based + CLMBR
    for encoder in encoders:
        for task in tasks:

            df = pd.read_csv(os.path.join( 
                artifacts_fpath,
                task,
                'eval',
                f"{encoder}_{model}_{train_years_str}",
                'by_group_comp_rel_ood_with_base.csv'
            ))
            
            df = df.replace(groups)
            
            test_groups = [v for k,v in groups.items()]
            df=df.query("test_group==@test_groups")

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
    
    # end-to-end model results
    if ete_models is not None:
        for ete_model in ete_models:
            for task in tasks:
                df = pd.read_csv(os.path.join( 
                    clmbr_artifacts_fpath,
                    task,
                    f"{ete_model}_{train_years_str}",
                    'by_group_all_results.csv'
                ))

                df = df.replace(groups)
            
                test_groups = [v for k,v in groups.items()]
                df=df.query("test_group==@test_groups")
                
                df = df.rename(columns={'boot_id':'boot_num'})
                
                df = (
                    df[['metric','boot_num','test_group']]
                    .assign(
                        encoder=ete_model+'_ete',
                        model=ete_model+'_ete',
                        task=task,
                        performance=df['performance'],
                        performance_base=df['performance_baseline'],
                        performance_delta=df['performance_delta'],
                    )
                )
                
                df_eval = pd.concat((
                    df_eval, df
                ))
                
    fig,axes=plt.subplots(
        nrows=len(metrics),
        ncols=len(tasks),
        figsize=figsize,
        constrained_layout=True
    )
    
    es = ['count'] + list(encoders.keys())
    
    if ete_models is not None:
        es += [x+'_ete' for x in ete_models]
        
    es.sort()
        
    markers = {
        'count':{'style':'o','color':'black','label':'Count-LR'},
        'gru':{'style':'s','color':'steelblue','label':'GRU-CLMBR'},
        'gru_ete':{'style':'d','color':'steelblue','label':'GRU-ETE'},
        'transformer':{'style':'s','color':'firebrick','label':'Transf-CLMBR'},
        'transformer_ete':{'style':'d','color':'firebrick','label':'Transf-ETE'},
    }
    
    es = {k:markers[k] for k in es}

    for c,task in enumerate(tasks):
        for r,metric in enumerate(metrics):
            
            handles_main = []
            handles_hatch = []
            
            for e,encoder in enumerate(es):
                
                for t,test_group in enumerate(test_groups):
                    if '_ete' in encoder:
                        data = df_eval.query(
                            "task==@task and metric==@metric and test_group==@test_group and encoder==@encoder"
                        ).reset_index()
                    else:
                        data = df_eval.query(
                            "task==@task and metric==@metric and test_group==@test_group and encoder==@encoder and model==@model"
                        ).reset_index()

                    data = data.groupby(
                        'test_group'
                    )['performance'].quantile(
                        [0.5,0.025,0.975]
                    ).unstack().reset_index()

                    data[0.025]=np.abs(data[0.025].values-data[0.5].values)
                    data[0.975]=np.abs(data[0.975].values-data[0.5].values)

                    h_main = axes[r][c].scatter(
                        t*len(es)+t + e,
                        data[0.5].values,
                        marker=es[encoder]['style'],
                        edgecolors='k',
                        facecolors=es[encoder]['color'],
                        label=es[encoder]['label'],
                    )
                    
                    if t==0:
                        handles_main.append(h_main)
                    
                    if t>0:
                        axes[r][c].plot(
                            [t*len(es)+t-1,t*len(es)+t-1],[-5,5],
                            color='grey',
                            linestyle='-',
                            linewidth=0.5,
                            zorder=0
                        )

                    axes[r][c].errorbar(
                        t*len(es)+t + e,
                        data[0.5],
                        data[[0.025,0.975]].values.T,
                        zorder=0,
                        linewidth=1,
                        c='k',
                        fmt='',
                    )
                
                    if e==0:
                        axes[r][c].plot(
                            [t*len(es)+t-0.5, t*len(es)+t+len(es)-0.5],
                            [data[0.5],data[0.5]],
                            'k--',
                            zorder=0,
                            linewidth=0.5,
                        )

                        h_hatch_1 = axes[r][c].fill_between(
                            [t*len(es)+t-0.5, t*len(es)+t+len(es)-0.5],
                            [-5, -5] if metric in ['auc','auprc','auprc_c'] else [5,5],
                            data[0.5],
                            facecolor='red',
                            #hatch='///',
                            #edgecolor='red',
                            linewidth=0,
                            alpha=fill_alpha,
                            interpolate=True,
                            zorder=0,
                            label='Worse than Cnt'
                        )

                        h_hatch_2 = axes[r][c].fill_between(
                            [t*len(es)+t-0.5, t*len(es)+t+len(es)-0.5],
                            [5, 5] if metric in ['auc','auprc','auprc_c'] else [-5, -5],
                            data[0.5],
                            facecolor='green',
                            #hatch='XXX',
                            #edgecolor='green',
                            linewidth=0,
                            alpha=fill_alpha,
                            interpolate=True,
                            zorder=0,
                            label='Better than Cnt'
                        )
                        
                        if t==0:
                            handles_hatch = [h_hatch_1, h_hatch_2]
                    
                axes[r][c].set_ylim(y_axis[metric][task])
                axes[r][c].set_xlim([-1,3*len(es)+2])
                axes[r][c].yaxis.set_major_locator(MaxNLocator(nbins=4,prune='both'))
                if plot_major:
                    axes[r][c].grid(which='major', linewidth=0.5, axis='y')
                axes[r][c].set_xticklabels('')
                axes[r][c].set_xlabel('')
                axes[r][c].tick_params(axis='x', length=0)
                
                if r==0:
                    axes[r][c].set_title(tasks[task])
                    
                if c==0:
                    axes[r][c].set_ylabel(metrics[metric])
                    axes[r][c].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                else:
                    axes[r][c].set_ylabel('')
                 
            if r == 1 and c == len(tasks)-1:
                leg1 = axes[r][c].legend(
                    handles=handles_main,
                    bbox_to_anchor=legend_bbox_to_anchor,
                    ncol=legend_ncols,
                )
    
    if save_path is not None:
        plt.savefig(save_path, dpi=save_res_dpi)
    
    plt.show()
    
    return df_eval

        
        


def plot_perf_grouped_ood_rel(
    artifacts_fpath,
    tasks,
    metrics,
    encoders,
    y_axis, clmbr_artifacts_fpath='/labs/shahlab/projects/lguo/temp_ds_shift_robustness/clmbr/experiments/clmbr/clmbr_artifacts/eval_end_to_end',
    ete_models = ['gru','transformer'],
    train_years=[2009,2010,2011,2012],
    groups = {
        '2013_2014_2015_2016':'13-16',
        '2017_2018_2019_2020_2021':'17-21',
    },
    model = 'lr',
    figsize = (18,6),
    save_path=None,
    save_res_dpi=300,
    show_legend=True,
    legend_bbox_to_anchor=(0.8,-0.5),
    legend_ncols=1,
    fill_alpha=0.05,
    plot_major=False,
    ):
    
    df_eval = pd.DataFrame()
    
    train_years_str = '_'.join([str(x) for x in train_years])
    train_years_abbrev = '-'.join([
        train_years_str.split('_')[0][-2:],
        train_years_str.split('_')[-1][-2:],
    ])
    
    # count-based + CLMBR
    for encoder in encoders:
        for task in tasks:

            df = pd.read_csv(os.path.join( 
                artifacts_fpath,
                task,
                'eval',
                f"{encoder}_{model}_{train_years_str}",
                'by_group_comp_rel_ood_with_base.csv'
            ))
            
            df = df.replace(groups)
            
            test_groups = [v for k,v in groups.items()]
            df=df.query("test_group==@test_groups")

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
    
    # end-to-end model results
    if ete_models is not None:
        for ete_model in ete_models:
            for task in tasks:
                df = pd.read_csv(os.path.join( 
                    clmbr_artifacts_fpath,
                    task,
                    f"{ete_model}_{train_years_str}",
                    'by_group_all_results.csv'
                ))

                df = df.replace(groups)
            
                test_groups = [v for k,v in groups.items()]
                df=df.query("test_group==@test_groups")
                
                df = df.rename(columns={'boot_id':'boot_num'})
                
                df = (
                    df[['metric','boot_num','test_group']]
                    .assign(
                        encoder=ete_model+'_ete',
                        model=ete_model+'_ete',
                        task=task,
                        performance=df['performance'],
                        performance_base=df['performance_baseline'],
                        performance_delta=df['performance_delta'],
                    )
                )
                
                df_eval = pd.concat((
                    df_eval, df
                ))
                
    fig,axes=plt.subplots(
        nrows=len(metrics),
        ncols=len(tasks),
        figsize=figsize,
        constrained_layout=True
    )
    
    es = ['count'] + list(encoders.keys())
    
    if ete_models is not None:
        es += [x+'_ete' for x in ete_models]
        
    es.sort()
        
    markers = {
        'count':{'style':'o','color':'black','label':'Count-LR'},
        'gru':{'style':'s','color':'steelblue','label':'GRU-CLMBR'},
        'gru_ete':{'style':'d','color':'steelblue','label':'GRU-ETE'},
        'transformer':{'style':'s','color':'darkorange','label':'Transf-CLMBR'},
        'transformer_ete':{'style':'d','color':'darkorange','label':'Transf-ETE'},
    }
    
    es = {k:markers[k] for k in es}

    for c,task in enumerate(tasks):
        for r,metric in enumerate(metrics):
            
            handles_main = []
            handles_hatch = []
            
            for e,encoder in enumerate(es):
                
                for t,test_group in enumerate(test_groups):
                    if '_ete' in encoder:
                        data = df_eval.query(
                            "task==@task and metric==@metric and test_group==@test_group and encoder==@encoder"
                        ).reset_index()
                    else:
                        data = df_eval.query(
                            "task==@task and metric==@metric and test_group==@test_group and encoder==@encoder and model==@model"
                        ).reset_index()

                    data = data.groupby(
                        'test_group'
                    )['performance_delta'].quantile(
                        [0.5,0.025,0.975]
                    ).unstack().reset_index()

                    data[0.025]=np.abs(data[0.025].values-data[0.5].values)
                    data[0.975]=np.abs(data[0.975].values-data[0.5].values)

                    h_main = axes[r][c].scatter(
                        t*len(es)+t + e,
                        data[0.5].values,
                        marker=es[encoder]['style'],
                        edgecolors=None,
                        facecolors=es[encoder]['color'],
                        label=es[encoder]['label'],
                    )

                    if t==0:
                        handles_main.append(h_main)
                        axes[r][c].plot(
                            [-1,20],[0,0],
                            color='black',
                            linestyle='-',
                            linewidth=0.5,
                            zorder=0
                        )
                        axes[r][c].plot(
                            [len(es),len(es)],[-5,5],
                            color='grey',
                            linestyle='-',
                            linewidth=0.5,
                            zorder=0
                        )

                    axes[r][c].errorbar(
                        t*len(es)+t + e,
                        data[0.5],
                        data[[0.025,0.975]].values.T,
                        zorder=0,
                        linewidth=1,
                        c=es[encoder]['color'],
                        fmt='',
                    )

                    
                    if e==0:
                        axes[r][c].plot(
                            [t*len(es)+t-0.5, t*len(es)+t+len(es)-0.5],
                            [data[0.5],data[0.5]],
                            'k--',
                            zorder=0,
                            linewidth=0.5,
                        )

                        h_hatch_1 = axes[r][c].fill_between(
                            [t*len(es)+t-0.5, t*len(es)+t+len(es)-0.5],
                            [-5, -5] if metric in ['auc','auprc','auprc_c'] else [5,5],
                            data[0.5],
                            facecolor='red',
                            #hatch='///',
                            #edgecolor='red',
                            linewidth=0,
                            alpha=fill_alpha,
                            interpolate=True,
                            zorder=0,
                            label='Worse than Cnt'
                        )

                        h_hatch_2 = axes[r][c].fill_between(
                            [t*len(es)+t-0.5, t*len(es)+t+len(es)-0.5],
                            [5, 5] if metric in ['auc','auprc','auprc_c'] else [-5, -5],
                            data[0.5],
                            facecolor='green',
                            #hatch='XXX',
                            #edgecolor='green',
                            linewidth=0,
                            alpha=fill_alpha,
                            interpolate=True,
                            zorder=0,
                            label='Better than Cnt'
                        )
                        handles_hatch = [h_hatch_1, h_hatch_2]
                    
                    
                axes[r][c].set_ylim(y_axis[metric][task])
                axes[r][c].set_xlim([-1,len(groups)*len(es)+(len(groups)-1)])
                axes[r][c].yaxis.set_major_locator(MaxNLocator(nbins=4,prune='both'))
                if plot_major:
                    axes[r][c].grid(which='major', linewidth=0.5, axis='y')
                axes[r][c].set_xticklabels('')
                axes[r][c].set_xlabel('')
                axes[r][c].tick_params(axis='x', length=0)
                
                if r==0:
                    axes[r][c].set_title(tasks[task])
                    
                if c==0:
                    axes[r][c].set_ylabel(metrics[metric])
                    axes[r][c].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                else:
                    axes[r][c].set_ylabel('')
                 
            if r == 1 and c == len(tasks)-1 and show_legend:
                leg1 = axes[r][c].legend(
                    handles=handles_main,
                    bbox_to_anchor=legend_bbox_to_anchor,
                    ncol=legend_ncols,
                )
    
    if save_path is not None:
        plt.savefig(save_path, dpi=save_res_dpi)
    
    plt.show()
    
    return df_eval
    
    
def plot_heatmap_ood(
    artifacts_fpath,
    tasks,
    y_axis,
    encoders,
    models,
    metrics,
    models_dict=models_dict,
    clmbr_artifacts_fpath="/labs/shahlab/projects/lguo/temp_ds_shift_robustness/clmbr/experiments/clmbr/clmbr_artifacts",
    train_years=[2009,2010,2011,2012],
    dv='performance_delta', 
    normalize_by_baseline=False,
    cmap='magma',
    figsize=(13,4),
    save_path=None,
    save_res_dpi=300,
    ete_models=['gru','transformer'],
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
                
                if normalize_by_baseline:
                    
                    df = df.assign(
                        performance_delta=df['performance_clmbr_delta']-df['performance_base_delta'],
                        performance_base=df['performance_base_delta'],
                        encoder=encoder
                    )
                
                else:
                
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
    
    if ete_models is not None:
        for ete_model in ete_models:
            for task in tasks:
                df = pd.read_csv(os.path.join( 
                    clmbr_artifacts_fpath,
                    'eval_end_to_end',
                    task,
                    f"{ete_model}_{train_years_joined}",
                    'by_group_all_results.csv'
                ))
            
                df=df.query("test_group!=@train_years_str and test_group!=@train_years_joined")
                
                df = df.rename(columns={'boot_id':'boot_num'})
                df = df.assign(task=task)
                
                tmp = df_eval.query("task==@task and encoder=='gru'")
                
                df = df.merge(
                    tmp[['task','metric','boot_num','performance_base']],
                    left_on=['task','metric','boot_num'],
                    right_on=['task','metric','boot_num'],
                )
                
                if normalize_by_baseline:
                    
                    df = df.assign(
                        performance_delta=df['performance_delta']-df['performance_base']
                    )
                
                df = (
                    df[['task','metric','boot_num','test_group']]
                    .assign(
                        encoder=ete_model+'_ete',
                        model=ete_model+'_ete',
                        performance=df['performance'],
                        performance_base=df['performance_baseline'],
                        performance_delta=df['performance_delta'],
                    )
                )
                
                df_eval = pd.concat((
                    df_eval, df
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
                if normalize_by_baseline:
                    abs_max = max(np.abs(data.min()), np.abs(data.max()))
                    vmin = abs_max*-1
                    vmax = abs_max
                else:
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
            
            if normalize_by_baseline:
                axes[r][c].plot(
                    [-0.5,len(xlabels)-0.5],
                    [len(encoders)-0.5,len(encoders)-0.5],
                    'k-'
                )
                
            else:
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
    
    return df_eval
    