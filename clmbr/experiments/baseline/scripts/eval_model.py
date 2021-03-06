import os
import argparse
import pickle
import joblib
import pdb
import re

import pandas as pd
import numpy as np

from prediction_utils.pytorch_utils.metrics import StandardEvaluator
from tune_model import read_file

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
    default = "/labs/shahlab/projects/lguo/temp_ds_shift_robustness/clmbr/experiments/baseline/artifacts",
    help = "path to save artifacts"
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
    "--use_same_train_group",
    type = str2bool,
    default = "true",
    help = "whether to overwrite existing artifacts",
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

for task in args.tasks:
    # assign index_year
    if task == 'readmission_30':
        index_year = 'discharge_year'
    else:
        index_year = 'admission_year'

    # set path
    fpath = os.path.join(
        args.artifacts_fpath,
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
    
    df = pd.concat([
        pd.read_csv(f"{fpath}/{x}").assign(
            model_num=x.split('_')[-1],
        ) 
        for x in files
    ])

    df = df.query("test_group!=[2007,2008]")
    df['train_groups'] = df['train_groups'].astype(str)
    df['test_group'] = df['test_group'].astype(str)
    
    ## evaluate    
    # evaluate stratify by year_group
    fpath = os.path.join(
        args.artifacts_fpath,
        task,
        "eval",
        '_'.join([
            args.model,
            '_'.join([str(x) for x in args.train_group]),
        ])
    )
    
    os.makedirs(fpath, exist_ok=True)
    
    strata_vars_dict = {
        'group':['test_group'],
        #'group_age':['group','age_group'],
        #'group_gender':['group','gender_concept_name'],
        #'group_race_eth':['group','race_eth'],
        #'group_race_eth_raw':['group','race_eth_raw'],
        #'group_race_eth_gender':['group','race_eth_gender'],
        #'group_race_eth_age_group':['group','race_eth_age_group'],
        #'group_race_eth_gender_age_group':['group','race_eth_gender_age_group'],
    }
    
    if args.use_same_train_group:
        train_group = ['2009','2010','2011','2012']
    else:
        train_group = df['train_groups'].unique()[0]
    
    evaluator = StandardEvaluator(
        metrics=['auc','auprc','auprc_c','loss_bce','ace_abs_logistic_logit'],
        **{'pi0':df.query("test_group==@train_group")['labels'].mean()} # set prior for calibrated auprc
    )
    
    train_group = df['train_groups'].unique()[0]
    
    for k,v in strata_vars_dict.items():
        
        if all([
            os.path.exists(f"{fpath}/{f}") for f in 
            [f'by_{k}.csv']
        ]) and not args.overwrite:

            print("Artifacts exist and args.overwrite is set to False. Skipping...")
            continue

        elif not all([
            os.path.exists(f"{fpath}/{f}") for f in 
            [f'by_{k}.csv']
        ]) or args.overwrite: 
            
            df_eval = evaluator.bootstrap_evaluate(
                df,
                strata_vars_eval=v,
                strata_vars_boot=['labels'],
                patient_id_var='row_id',
                n_boot=args.n_boot,
                n_jobs=args.n_jobs,
                strata_var_experiment='test_group', 
                baseline_experiment_name=train_group
            )

            df_eval.to_csv(f"{fpath}/by_{k}.csv",index=False)
