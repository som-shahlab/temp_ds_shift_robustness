import os
import argparse
import pickle
import joblib
import pdb
import re

import pandas as pd
import numpy as np

from prediction_utils.pytorch_utils.metrics import StandardEvaluator

from prediction_utils.util import str2bool

#------------------------------------
# Arg parser
#------------------------------------
parser = argparse.ArgumentParser(
    description = "Evaluate best lr model"
)

parser.add_argument(
    "--artifacts_fpath",
    type = str,
    default = "/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/clmbr/experiments/clmbr/adapter_artifacts",
    help = "path to clmbr adapter model artifacts"
)

parser.add_argument(
    "--base_artifacts_fpath",
    type = str,
    default = "/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/clmbr/experiments/baseline/artifacts",
    help = "path to count feature model artifacts"
)

parser.add_argument(
    "--count_features_fpath",
    type = str,
    default = '/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/clmbr/features/',
    help = "path to count features"
)

parser.add_argument(
    "--tasks",
    type=str,
    default="hospital_mortality/LOS_7/readmission_30/icu_admission",
    help="prediction tasks"
)

parser.add_argument(
    "--train_group", 
    type=str ,
    default = "2009/2010/2011/2012",
    help="group(s) to train on [2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]"
)

parser.add_argument(
    "--clmbr_encoder",
    type=str,
    default='gru',
    help='gru/transformer',
)

parser.add_argument(
    "--model",
    default = 'lr',
    type = str
)

parser.add_argument(
    "--seed",
    type = int,
    default = 44,
    help = "seed for deterministic training"
)

parser.add_argument(
    "--n_boot",
    type = int,
    default = 1000,
    help = "num bootstrap iterations"
)

parser.add_argument(
    "--n_jobs",
    type = int,
    default = 4,
    help = "num jobs"
)

parser.add_argument(
    "--overwrite",
    type = str2bool,
    default = "false",
    help = "whether to overwrite existing artifacts",
)

#-------------------------------------------------------------------
# helper functions
#-------------------------------------------------------------------

#-------------------------------------------------------------------
# run
#-------------------------------------------------------------------
args = parser.parse_args()
    
# set seed
np.random.seed(args.seed)

# parse tasks and train_group
args.tasks = args.tasks.split("/")
args.train_group = [int(x) for x in args.train_group.split("/")]



df_comp = pd.DataFrame()

for task in args.tasks:
    
    ## grab clmbr feature model predictions
    # set path
    fpath = os.path.join(
        args.artifacts_fpath,
        task,
        "pred_probs",
        '_'.join([
            args.clmbr_encoder,
            args.model,
            '_'.join([str(x) for x in args.train_group]),
        ])
    )

    # grab predictions by best model from path
    files = [
        x for x in os.listdir(fpath)
        if '.csv' in x and
        'best_model' in x
    ]
    
    assert len(files)==1
    
    df = pd.concat([
        pd.read_csv(f"{fpath}/{x}").assign(
            model_num=x.split('_')[-1],
        ) 
        for x in files
    ])

    df = df.query("test_group!=[2007,2008]")
    df['train_groups'] = df['train_groups'].astype(str)
    df['test_group'] = df['test_group'].astype(str)
    
    ## grab count feature model predictions
    # set path
    fpath = os.path.join(
        args.base_artifacts_fpath,
        task,
        "pred_probs",
        '_'.join([
            args.model,
            '_'.join([str(x) for x in args.train_group]),
        ])
    )

    # grab predictions by best model from path
    files = [
        x for x in os.listdir(fpath)
        if '.csv' in x and
        'best_model' in x
    ]
    
    assert len(files)==1
    
    df_base = pd.concat([
        pd.read_csv(f"{fpath}/{x}").assign(
            model_num=x.split('_')[-1],
        ) 
        for x in files
    ])

    df_base = df_base.query("test_group!=[2007,2008]")
    df_base['train_groups'] = df_base['train_groups'].astype(str)
    df_base['test_group'] = df_base['test_group'].astype(str)
    
    ## evaluate    
    # evaluate stratify by year_group
    fpath = os.path.join(
        args.artifacts_fpath,
        task,
        "eval",
        '_'.join([
            args.clmbr_encoder,
            args.model,
            '_'.join([str(x) for x in args.train_group]),
        ])
    )
    
    os.makedirs(fpath, exist_ok=True)
    
    strata_vars_dict = {
        'group':['test_group'],
    }
    
    train_group = df['train_groups'].unique()[0]
    
    evaluator = StandardEvaluator(
        metrics=['auc','auprc','auprc_c','loss_bce','ace_abs_logistic_logit'],
        **{'pi0':df.query("test_group==@train_group")['labels'].mean()} # set prior for calibrated auprc
    )
    
    for k,v in strata_vars_dict.items():
        
        if all([
            os.path.exists(f"{fpath}/{f}") for f in 
            [f'by_{k}.csv', f'by_{k}_comp_rel_ood_with_base.csv']
        ]) and not args.overwrite:

            print("Artifacts exist and args.overwrite is set to False. Skipping...")
            continue

        elif not all([
            os.path.exists(f"{fpath}/{f}") for f in 
            [f'by_{k}.csv', f'by_{k}_comp_rel_ood_with_base.csv']
        ]) or args.overwrite: 
            
            # get clmbr feature model evaluations
            df_eval_clmbr_ci, df_eval_clmbr = evaluator.bootstrap_evaluate(
                df,
                strata_vars_eval=v,
                strata_vars_boot=['labels'],
                patient_id_var='prediction_id',
                n_boot=args.n_boot,
                n_jobs=args.n_jobs,
                strata_var_experiment='test_group', 
                baseline_experiment_name=train_group,
                return_result_df=True
            )
            
            # save clmbr feature model evaluations
            df_eval_clmbr_ci.to_csv(f"{fpath}/by_{k}.csv",index=False)
            
            
            # get prediction_id for count feature models
            row_id_map = pd.read_parquet(os.path.join(
                args.count_features_fpath,
                'admissions_admission_midnight' if task!='readmission_30' else 'admissions_discharge_midnight',
                'merged_features_binary',
                'features_sparse',
                'features_row_id_map.parquet'
            ))
            
            df_base = df_base.merge(
                row_id_map, 
                left_on ='row_id',
                right_on = 'features_row_id'
            )
            
            # check that prediction_id are equal
            assert(
                len(set(df['prediction_id']) & set(df_base['prediction_id'])) 
                == 
                len(set(df['prediction_id']))
            )
            
            
            # bootstrap compare robustness between clmbr feature model robustness and count feature model 
            for i_boot in range(args.n_boot):
                
                # sample prediction_id
                ids = np.random.choice(
                    df['prediction_id'].unique(), 
                    len(df['prediction_id'].unique()),
                    replace=True
                ).tolist()
                
                sample_clmbr = df.query("prediction_id==@ids")
                sample_base = df_base.query("prediction_id==@ids")

                df_eval_clmbr = evaluator.evaluate(sample_clmbr, strata_vars= ['test_group'])
                df_eval_clmbr = df_eval_clmbr.merge(
                    df_eval_clmbr.query("test_group==@train_group")[['metric','performance']],
                    on='metric',
                    suffixes=('_clmbr','_clmbr_baseline')
                )

                df_eval_base = evaluator.evaluate(sample_base, strata_vars= ['test_group'])
                df_eval_base = df_eval_base.merge(
                    df_eval_base.query("test_group==@train_group")[['metric','performance']],
                    on='metric',
                    suffixes=('_base','_base_baseline')
                )

                df_comp = pd.concat((df_comp,pd.merge(
                    left = df_eval_clmbr.assign(
                        task=task,
                        clmbr_encoder=args.clmbr_encoder,
                        model=args.model,
                        boot_num=i_boot,
                        performance_clmbr_delta=df_eval_clmbr['performance_clmbr'] - df_eval_clmbr['performance_clmbr_baseline']
                    ),
                    right = df_eval_base.assign(
                        task=task,
                        clmbr_encoder=args.clmbr_encoder,
                        model=args.model,
                        boot_num=i_boot,
                        performance_base_delta= df_eval_base['performance_base'] - df_eval_base['performance_base_baseline']
                    ),
                    on = ['task','clmbr_encoder','model','metric','test_group','boot_num']
                )))
                
            df_comp.to_csv(f"{fpath}/by_{k}_comp_rel_ood_with_base.csv",index=False)