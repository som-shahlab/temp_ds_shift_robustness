import os
import pickle

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from matplotlib.ticker import MaxNLocator
from matplotlib.pyplot import savefig
from matplotlib.ticker import FormatStrFormatter
from sklearn.calibration import calibration_curve

def get_result_table(
    train_year,
    tasks,
    algos,
    metrics,
    use_comparator='comparator',
    artifacts_fpath='/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/experiments/dg/artifacts',
    baseline_artifacts_fpath='/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/experiments/baseline/artifacts'
    ):
    
    df_results = pd.DataFrame(columns = ['Algorithm','Task','Metric','Performance'])
    c = 0
    for task in tasks:
        for algo in algos:
            if algo=='erm':
                df_eval = pd.read_csv(os.path.join(
                    baseline_artifacts_fpath,
                    task,
                    "eval",
                    f"nn_{train_year}",
                    "by_group.csv"
                )).round(3).assign(algo=algo)
            else:
                df_eval = pd.read_csv(os.path.join(
                    artifacts_fpath,
                    task,
                    "eval",
                    f"{algo}_{train_year}",
                    "by_group.csv"
                )).round(3).assign(algo=algo)

            for metric in metrics:
                c+=1
                test_group=train_year
                # if erm, test_group and comparator are baseline
                if algo=='erm':
                    comparator='baseline'
                else:
                    comparator=use_comparator

                mid = df_eval.query("metric==@metric and CI_quantile_95==['mid'] and test_group==@test_group")[comparator].values[0]
                lower = df_eval.query("metric==@metric and CI_quantile_95==['lower'] and test_group==@test_group")[comparator].values[0]
                upper = df_eval.query("metric==@metric and CI_quantile_95==['upper'] and test_group==@test_group")[comparator].values[0]
                df_results.loc[c,:] = [algo,task,metric,f"{mid} ({lower}, {upper})"]

    # rename values
    df_results = df_results.replace({**tasks,**algos,**metrics})
    # pivot table
    df_results = df_results.pivot(index=['Metric','Task'], columns='Algorithm',values='Performance')

    # re-order index and columns
    df_results = df_results.reindex(labels=[metrics['auc'],metrics['auprc'],metrics['ace_abs_logistic_logit']],level=0)
    df_results = df_results[[algos['erm'],algos['irm'],algos['dro'],algos['coral'],algos['adversarial']]]

    return df_results


def plot_rel_ood_perf(
    train_year,
    tasks,
    algos,
    metrics,
    y_axis,
    cmap='viridis',
    baseline_label='ERM[09-12]',
    ood_label='ERM[Year-Specific]',
    figsize=(12,5),
    legend_bbox_to_anchor=(0.5,-0.5),
    legend_ncols=5,
    save_path=None,
    save_res_dpi=300,
    artifacts_fpath='/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/experiments/dg/artifacts',
    baseline_artifacts_fpath='/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/experiments/baseline/artifacts',
    test_group=[2019,2020,2021],
    year_color='#f5ad31',
    year_hatch='....',
    ):
    
    fig, axes = plt.subplots(nrows = len(metrics), ncols=len(tasks),figsize=(12,5))

    for c,task in enumerate(tasks):
        df_eval = pd.DataFrame()

        # load results
        df_base = pd.read_csv(os.path.join(
            baseline_artifacts_fpath,
            task,
            "eval",
            f"nn_{train_year}",
            "by_group.csv"
        )).query("test_group==[@train_year] and CI_quantile_95=='mid'")[
            ['metric','baseline']
        ]

        for algo in algos:
            if algo=='erm':
                df_eval = pd.concat(
                    (df_eval,pd.read_csv(os.path.join(
                        baseline_artifacts_fpath,
                        task,
                        "eval",
                        f"nn_{train_year}",
                        "by_group.csv"
                    )).round(3).assign(algo=algo)
                    )
                )
            else:
                df_eval = pd.concat(
                    (df_eval,pd.read_csv(os.path.join(
                        artifacts_fpath,
                        task,
                        "eval",
                        f"{algo}_{train_year}",
                        "by_group.csv"
                    )).round(3).assign(algo=algo)
                    )
                )

        # get results from 2019,2020,2021
        all_metrics=list(metrics.keys())
        df_eval = df_eval.query("test_group==['2019','2020','2021'] and metric==@all_metrics")
        
        for r,metric in enumerate(metrics):

            year_groups = df_eval.test_group.unique()
            year_groups.sort()

            # baseline (0)
            axes[r][c].plot(
                [0.5, len(year_groups)*len(algos)+0.5],
                [0,0],
                '--',
                c = 'black',
                zorder = 0,
                label=baseline_label
            )

            # column counter
            xc = 0 
            for cy,year_group in enumerate(year_groups):

                int_year_group = int(year_group)

                perf_id = pd.read_csv(
                    os.path.join(
                        baseline_artifacts_fpath,
                        task,
                        "eval",
                        f"nn_{year_group}",
                        "by_group.csv"
                    )
                ).query("test_group==@int_year_group and CI_quantile_95=='mid' and metric==@metric")[
                    ['baseline']
                ]

                # get Median, CI_Lower, CI_Upper for all algos in year group
                temp = df_eval.query("metric==@metric and test_group==@year_group")
                data = temp.query("CI_quantile_95=='mid'").reset_index(drop=True)

                data['CI_upper'] = (
                    temp.query("CI_quantile_95=='upper'").reset_index()['delta'] - 
                    temp.query("CI_quantile_95=='mid'").reset_index()['delta']
                ).abs()

                data['CI_lower'] = (
                    temp.query("CI_quantile_95=='lower'").reset_index()['delta'] - 
                    temp.query("CI_quantile_95=='mid'").reset_index()['delta']
                ).abs()

                for icolor,algo in enumerate(algos):

                    xc+=1
                    idata = data.query("algo==@algo")
                    
                    # scatter
                    axes[r][c].scatter(
                        xc,
                        idata['delta'],
                        s = 20,
                        color = sns.color_palette("deep")[0] if algo=='erm' else sns.color_palette(cmap,n_colors=len(algos)-1)[icolor-1],
                        label = algos[algo] if cy==0 else '_nolegend_'
                    )

                    # error bar
                    axes[r][c].errorbar(
                        xc,
                        idata['delta'],
                        idata[['CI_lower','CI_upper']].values.T,
                        zorder = 0,
                        linewidth = 1.5,
                        color = sns.color_palette("deep")[0] if algo=='erm' else sns.color_palette(cmap,n_colors=len(algos)-1)[icolor-1]
                    )

               # ----------- year boundaries -------------#
                # plot rectangle box for dg in each year group
                rect = patches.Rectangle(
                    (cy*len(algos)+1.5,-1),
                    len(algos)-1,
                    20,
                    linewidth=0,
                    edgecolor=year_color,
                    facecolor='None',
                    hatch=year_hatch,
                    alpha=0.35,
                    zorder=0
                )
                axes[r][c].add_patch(rect)

                # lines
                axes[r][c].plot(
                    [cy*len(algos)+0.5,cy*len(algos)+0.5],
                    [-1,1],
                    linewidth=1,
                    color='grey',
                )
                axes[r][c].plot(
                    [cy*len(algos)+len(algos)+0.5,cy*len(algos)+len(algos)+0.5],
                    [-1,1],
                    linewidth=1,
                    color='grey',
                )
                #----------- retrained ID models ---------#
                base_value = df_base.query("metric==@metric")['baseline'].values[0]
                id_value = perf_id.values[0][0] - base_value

                axes[r][c].plot(
                    [cy*len(algos)+0.5,cy*len(algos)+len(algos)+0.5],
                    [id_value,id_value],
                    linewidth=1.5,
                    color='black',
                    linestyle='-.',
                    zorder=0,
                    label=ood_label if cy==0 else None
                )

            ## Axes settings
            axes[r][c].set_ylim(y_axis[metric]['lim'])
            axes[r][c].set_xlim([0.5,len(algos)*len(year_groups)+0.5])
            axes[r][c].yaxis.set_major_locator(MaxNLocator(nbins=4,prune='both'))
            axes[r][c].grid(which='major', linewidth=0.5, axis='y')
            if r==0:
                axes[r][c].set_title(tasks[task])

            if r==len(metrics)-1:
                axes[r][c].set_xticks(np.arange(3,len(algos)*len(year_groups),len(algos)))
                axes[r][c].set_xticklabels(year_groups)
                axes[r][c].tick_params(axis='x', length=0)
                axes[r][c].set_xlabel('Year')
            else:
                axes[r][c].set_xticklabels('')
                axes[r][c].set_xlabel('')
                axes[r][c].tick_params(axis='x', length=0)

            if c == 0:
                axes[r][c].set_ylabel(f"Relative\n{metrics[metric]}")
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
                    title='Algorithms',
                )

    #sns.despine(offset=10, trim = True)
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(save_path, dpi=save_res_dpi)
    plt.show()
    
    
def plot_calibration_curves(
    tasks,
    algos,
    ax_lims,
    ood_groups=['2019','2020','2021'],
    id_group='2009_2010_2011_2012',
    id_group_label='09-12',
    n_bins=20,
    artifacts_fpath='/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/experiments/dg/artifacts',
    baseline_artifacts_fpath='/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/experiments/baseline/artifacts',
    save_path=None,
    save_res_dpi=300,
    figsize=(12,15),
    legend_bbox_to_anchor=(0.4,-0.3),
    legend_ncols=5
    ):

    fig, axes = plt.subplots(
        nrows=len(algos),
        ncols = len(tasks),
        figsize=figsize
    )
    
    for r,algo in enumerate(algos):
        for c,task in enumerate(tasks):
            if algo=='erm':
                fpath = os.path.join(
                    baseline_artifacts_fpath,
                    task,
                    "pred_probs",
                    f"nn_{id_group}"
                )
            else:
                fpath = os.path.join(
                    artifacts_fpath,
                    task,
                    "pred_probs",
                    f"{algo}_{id_group}"
                )
            fname = [x for x in os.listdir(fpath) if 'best_model' in x]
            df_probs = pd.read_csv(f"{fpath}/{fname[0]}", dtype={'test_group':'str'}).round(3)
            id_probs = df_probs.query("test_group==[@id_group]")[['labels','pred_probs']]

            # perfect calibration
            axes[r][c].plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

            # ID
            fraction_of_positives, mean_predicted_value = calibration_curve(
                id_probs['labels'],
                id_probs['pred_probs'],
                n_bins=n_bins,
                strategy='quantile'
            )

            axes[r][c].plot(
                mean_predicted_value, 
                fraction_of_positives, 
                "o-", 
                label=f"ID [{id_group_label}]"
            )

            for g in ood_groups:
                # ood pred probs 
                ood_probs = df_probs.query("test_group==[@g]")[['labels','pred_probs']]

                fraction_of_positives, mean_predicted_value = calibration_curve(
                    ood_probs['labels'],
                    ood_probs['pred_probs'],
                    n_bins=n_bins,
                    strategy='quantile'
                )

                axes[r][c].plot(
                    mean_predicted_value, 
                    fraction_of_positives, 
                    "o-", 
                    label=f"OOD [{g}]"
                )

            ## axes settings
            if c==0:
                axes[r][c].set_ylabel(f"{algos[algo]}\nFraction of positives")


            axes[r][c].set_ylim(ax_lims[task])
            axes[r][c].set_xlim(ax_lims[task])

            if r==0:
                axes[r][c].set_title(tasks[task])
            elif r==len(algos)-1:
                axes[r][c].set_xlabel("Mean Predicted Value")
            else:
                axes[r][c].set_title('')
                axes[r][c].set_xlabel('')

            axes[r][c].set_aspect('equal')

            if c == len(tasks)-1 and r == len(algos)-1:
                axes[r][c].legend(
                    bbox_to_anchor=legend_bbox_to_anchor,
                    ncol=legend_ncols,
                    frameon=True
                )
    plt.show()
    plt.tight_layout()
    
def plot_kde_pred_probs(
    tasks,
    algos,
    outcome=0,
    ood_groups=['2019','2020','2021'],
    id_group='2009_2010_2011_2012',
    id_group_label='09-12',
    artifacts_fpath='/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/experiments/dg/artifacts',
    baseline_artifacts_fpath='/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/experiments/baseline/artifacts',
    save_path=None,
    save_res_dpi=300,
    figsize=(15,15),
    x_lim=[0,0.4],
    legend_bbox_to_anchor=(0.4,-0.3),
    legend_ncols=4
    ):

    fig, axes = plt.subplots(
        ncols = len(tasks),
        nrows = len(algos),
        figsize=figsize,
    )
    
    for r,algo in enumerate(algos):
        for c,task in enumerate(tasks):
            if algo=='erm':
                fpath = os.path.join(
                    baseline_artifacts_fpath,
                    task,
                    "pred_probs",
                    f"nn_{id_group}"
                )
            else:
                fpath = os.path.join(
                    artifacts_fpath,
                    task,
                    "pred_probs",
                    f"{algo}_{id_group}"
                )

            fname = [x for x in os.listdir(fpath) if 'best_model' in x]

            df_probs = pd.read_csv(f"{fpath}/{fname[0]}", dtype={'test_group':'str'}).round(3)
            id_probs = df_probs.query("test_group==[@id_group] and labels==@outcome")[['labels','pred_probs']].assign(group=f'ID [{id_group_label}]').sample(10000,replace=True)

            ood_probs=pd.DataFrame()
        
            for g in ood_groups:
                ood_probs = pd.concat((
                    ood_probs,
                    df_probs.query("test_group==@g and labels==[@outcome]")[
                        ['labels','pred_probs']
                    ].assign(
                        group=f'OOD [{g}]'
                    ).sample(10000,replace=True)
                ))

            data = pd.concat((id_probs,ood_probs),ignore_index=True)

            sns.kdeplot(
                data = data,
                x = 'pred_probs',
                hue='group',
                ax = axes[r][c],
                cumulative=True,
                legend=True if c==len(tasks)-1 and r==len(algos)-1 else False
            )

            ## axes setting
            axes[r][c].set_xlim(x_lim)

            if c==0:
                axes[r][c].set_ylabel(f"{algos[algo]}\nDensity")
            else:
                axes[r][c].set_ylabel('')

            if r==len(algos)-1:
                axes[r][c].set_xlabel("Predicted probability")
            elif r==0:
                axes[r][c].set_title(f"{tasks[task]}\n(y={outcome})")
            else:
                axes[r][c].set_title('')
                axes[r][c].set_xlabel('')

    plt.show()
    plt.tight_layout()