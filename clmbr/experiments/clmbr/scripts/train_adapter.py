import os
import shutil
import argparse
import pickle
import joblib
import pdb
import re
import yaml

import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix as csr
from sklearn.linear_model import LogisticRegression as lr
from lightgbm import LGBMClassifier as gbm

from prediction_utils.pytorch_utils.metrics import StandardEvaluator
from prediction_utils.util import str2bool

from tune_adapter import (
    get_data,
    get_xy,
)

#------------------------------------
# Arg parser
#------------------------------------
parser = argparse.ArgumentParser(
    description = "Train model on selected hyperparameters"
)

parser.add_argument(
    "--clmbr_artifacts_fpath",
    type = str,
    default = "/labs/shahlab/projects/lguo/temp_ds_shift_robustness/clmbr/experiments/clmbr/clmbr_artifacts",
    help = "path to save artifacts"
)

parser.add_argument(
    "--adapter_artifacts_fpath",
    type = str,
    default = "/labs/shahlab/projects/lguo/temp_ds_shift_robustness/clmbr/experiments/clmbr/adapter_artifacts",
    help = "path to save artifacts"
)

parser.add_argument(
    "--cohort_fpath",
    type = str,
    default = "/labs/shahlab/projects/lguo/temp_ds_shift_robustness/clmbr/cohorts/",
    help = "path to save cohort"
)

parser.add_argument(
    "--cohort_id",
    type = str,
    default = "admissions",
    help = "which cohort to split"
)

parser.add_argument(
    "--tasks",
    type=str,
    default="hospital_mortality/LOS_7/readmission_30/icu_admission",
    help="prediction tasks"
)

parser.add_argument(
    "--clmbr_encoder",
    type=str,
    default='gru',
    help='gru/transformer',
)

parser.add_argument(
    "--model",
    type=str,
    default="lr"
)

parser.add_argument(
    "--train_group", 
    type=str ,
    default = "2009/2010/2011/2012",
    help="group(s) to train on [2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]"
)

parser.add_argument(
    "--n_jobs",
    type=int,
    default=4,
    help="number of threads"
)

parser.add_argument(
    "--seed",
    type = int,
    default = 44,
    help = "seed for deterministic training"
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
    
# threads
joblib.Parallel(n_jobs=args.n_jobs)

# set seed
np.random.seed(args.seed)

# parse tasks and train_group
args.tasks = args.tasks.split("/")
args.train_group = [int(x) for x in args.train_group.split("/")]

# initialize evaluator
evaluator = StandardEvaluator(metrics=['loss_bce'])

for task in args.tasks:
    
    print(f"task: {task}")

    features_fpath = os.path.join(
        args.clmbr_artifacts_fpath,
        "features",
        "_".join([str(x) for x in args.train_group]),
        args.clmbr_encoder,
    )

    best = [x for x in os.listdir(features_fpath) if 'best_model' in x][0]

    # get data
    features,labels,prediction_ids,ehr_ml_patient_ids,day_indices = get_data(
        os.path.join(
            features_fpath,
            best,
        ), 
        args.cohort_fpath, 
        args.cohort_id
    )
    
    print(f"Grabbing model performance for {task}")

    # set path
    fpath = os.path.join(
        args.adapter_artifacts_fpath,
        task,
        "models",
        '_'.join([
            args.clmbr_encoder,
            args.model,
            '_'.join([str(x) for x in args.train_group]),
        ])
    )

    # grab all files from path
    all_models = [
        x for x in os.listdir(fpath)
        if 'best' not in x
    ]

    df = pd.concat([
        pd.read_csv(f"{fpath}/{x}/val_pred_probs.csv").assign(
            model_num=x.split('_')[0],
        ) 
        for x in all_models
    ])
    
    
    # find best hparam setting based on log loss
    df_eval = evaluator.evaluate(
        df,
        strata_vars=['model_num']
    )

    df_eval = df_eval.groupby(['metric','model_num']).agg(
        mean_performance=('performance','mean')
    ).reset_index()

    model_num = float(df_eval.loc[
        df_eval.query("metric=='loss_bce'")['mean_performance'].idxmin(),
        'model_num'
    ])
    
    # grab model hparams
    hparams = yaml.load(
        open(f"{fpath}/{int(model_num)}/hparams.yml"),
        Loader=yaml.FullLoader
    )
        
    # refit model with all training data
    print(f"Refitting model number {model_num} with hparams {hparams}")
    
    # get data
    X_train,y_train,prediction_id_train = get_xy(
        task=task,
        features=features,
        labels=labels,
        prediction_ids=prediction_ids,
        combine_train_val=True
    )
    
    # save path
    folder_name = '_'.join([
        args.clmbr_encoder,
        args.model,
        '_'.join([str(x) for x in args.train_group]),
    ])

    model_name = '_'.join([
        "best_model",
        str(int(model_num)),
    ])

    fpath = os.path.join(
        args.adapter_artifacts_fpath,
        task,
        'models',
        folder_name,
        model_name
    )
    
    df = pd.DataFrame()
        
    if all([
        os.path.exists(f"{fpath}/{f}") for f in 
        [f'model.pkl','hparams.yml']
    ]) and not args.overwrite:

        print("Artifacts exist and args.overwrite is set to False. Skipping...")
        continue

    elif not all([
        os.path.exists(f"{fpath}/{f}") for f in 
        [f'model.pkl','hparams.yml']
    ]) or args.overwrite: 
        
        # remove best_model folders
        fpath_to_remove = os.path.join(
            args.adapter_artifacts_fpath,
            task,
            'models',
            folder_name,
        )
        
        for f in os.listdir(fpath_to_remove):
            if 'best_model' in f:
                shutil.rmtree(os.path.join(fpath_to_remove, f), ignore_errors=True)
                
        os.makedirs(fpath,exist_ok=True)
        
        if args.model=='lr':
            
            m = lr(
                n_jobs=args.n_jobs,
                **hparams
            )
            
        elif args.model=='gbm':
            
            m = gbm(
                n_jobs=args.n_jobs, 
                **hparams
            )
        
        m.fit(X_train,y_train)
        
        # save
        pickle.dump(
            m,
            open(f"{fpath}/model.pkl","wb")
        )
        
        yaml.dump(
            hparams,
            open(f"{fpath}/hparams.yml","w")
        )

        all_groups = [
            args.train_group,
            2009,2010,2011,2012,
            2013,2014,2015,2016,
            2017,2018,2019,2020,2021,
        ]
        
        ## get test data
        for group in all_groups:
            print(f'Obtaining model prediction in group {group}')
            
            X_test=features[task][f"test_{group}"]
            y_test=labels[task][f"test_{group}"]
            prediction_id_test=prediction_ids[task][f"test_{group}"]
            
            if type(group)==list:
                test_group='_'.join([str(x) for x in group])
            else:
                test_group=str(group)
                
            df = pd.concat((
                df,
                pd.DataFrame({
                    'pred_probs':m.predict_proba(X_test)[:,1],
                    'labels':y_test,
                    'prediction_id':prediction_id_test,
                    'task':task,
                    'train_groups':'_'.join([str(x) for x in args.train_group]),
                    'test_group':test_group,
                })
            ))

    
    # save predictions
    folder_name = '_'.join([
        args.clmbr_encoder,
        args.model,
        '_'.join([str(x) for x in args.train_group])
    ])

    file_name = '_'.join([
        "best_model",
        str(int(model_num)),
    ]) 

    fpath = os.path.join(
        args.adapter_artifacts_fpath,
        task,
        'pred_probs',
        folder_name
    )

    os.makedirs(fpath, exist_ok=True)
    
    if all([
        os.path.exists(f"{fpath}/{f}") for f in 
        [f'{file_name}.csv']
    ]) and not args.overwrite:

        print("Artifacts exist and args.overwrite is set to False. Skipping...")
        continue

    elif not all([
        os.path.exists(f"{fpath}/{f}") for f in 
        [f'{file_name}.csv']
    ]) or args.overwrite: 
        
        for f in os.listdir(fpath):
            if 'best_model' in f:
                os.remove(os.path.join(fpath,f))
        
        # add additional group info from cohort
        cohort_dir = os.path.join(
            args.cohort_fpath,
            args.cohort_id,
            "cohort",
            "cohort_split.parquet"
        )

        cohort = pd.read_parquet(cohort_dir)
        cohort = cohort.query(f"{task}_fold_id=='test'")
        
        df = df.merge(
            cohort[[
                'prediction_id',
                'age_group',
                'race_eth', 
                'gender_concept_name',
                'race_eth_raw', 
                'race_eth_gender', 
                'race_eth_age_group',
                'race_eth_gender_age_group', 
                'race_eth_raw_gender',
                'race_eth_raw_age_group', 
                'race_eth_raw_gender_age_group',
            ]],
            left_on='prediction_id',
            right_on='prediction_id'
        )

        df['test_group'] = df['test_group'].astype(str)
        
        df.reset_index(drop=True).to_csv(f"{fpath}/{file_name}.csv")