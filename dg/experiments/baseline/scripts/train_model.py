import os
import argparse
import pickle
import joblib
import pdb
import re
import yaml
import torch

import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix as csr
from sklearn.linear_model import LogisticRegression as lr

from prediction_utils.pytorch_utils.datasets import ArrayLoaderGenerator
from prediction_utils.pytorch_utils.models import FixedWidthModel
from prediction_utils.pytorch_utils.metrics import StandardEvaluator
from prediction_utils.util import str2bool

from tune_model import (
    read_file, 
    get_data,
    get_torch_data_loaders,
)
    

#------------------------------------
# Arg parser
#------------------------------------
parser = argparse.ArgumentParser(
    description = "Evaluate best lr model"
)

parser.add_argument(
    "--artifacts_fpath",
    type = str,
    default = "/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/dg/experiments/baseline/artifacts",
    help = "path to save artifacts"
)

parser.add_argument(
    "--features_fpath",
    type = str,
    default = "/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/dg/features/admissions",
    help = "path to extracted features"
)

parser.add_argument(
    "--cohort_fpath",
    type = str,
    default = "/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/dg/cohorts/",
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
    "--model",
    type=str,
    default="nn"
)

parser.add_argument(
    "--train_group", 
    type=str ,
    default = "2009/2010/2011/2012",
    help="group(s) to train on [2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]"
)

parser.add_argument(
    "--n_models",
    type=int,
    default=5
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
torch.set_num_threads(1)
joblib.Parallel(n_jobs=1)

# set seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# parse tasks and train_group
args.tasks = args.tasks.split("/")
args.train_group = [int(x) for x in args.train_group.split("/")]

# initialize evaluator
evaluator = StandardEvaluator(metrics=['loss_bce'])

for task in args.tasks:
    
    print(f"task: {task}")
    
    # assign index_year & features_fpath
    if task == 'readmission_30':
        index_year = 'discharge_year'
        features_fpath = args.features_fpath+'_discharge/merged_features_binary'
    else:
        index_year = 'admission_year'
        features_fpath = args.features_fpath+'_admission/merged_features_binary'

    # get data
    vocab, features, row_id_map = get_data(features_fpath, args.cohort_fpath, args.cohort_id)

    print(f"Grabbing model performance for {task}")

    # set path
    fpath = os.path.join(
        args.artifacts_fpath,
        task,
        "models",
        '_'.join([
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
            fold_id=''.join(
                re.findall(
                    r'\d+',
                    x.split('_')[1]
                )
            )
        ) 
        for x in all_models
    ])
    
    
    # find best hparam setting based on log loss
    df_eval = evaluator.evaluate(
        df,
        strata_vars=['model_num','fold_id']
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
        open(f"{fpath}/{int(model_num)}_fold1/hparams.yml"),
        Loader=yaml.FullLoader
    )
        
    # refit model with all training data
    print(f"Refitting model number {model_num} with hparams {hparams}")
    
    # prune features
    file_name = '_'.join([
        args.model,
        '_'.join([str(x) for x in args.train_group]),
    ])

    fpath = os.path.join(
        args.artifacts_fpath,
        task,
        'preprocessor',
    )

    vocab = pd.read_csv(f"{fpath}/{file_name}.csv")
    features = features[:,vocab.index[vocab['keep_feature']==1]]
    
    # generate torch loaders
    train_loaders = get_torch_data_loaders(
        task,
        args.train_group,
        index_year,
        row_id_map,
        features,
        val_fold='val',
        test_fold='test'
    )
    
    # save path
    folder_name = '_'.join([
        args.model,
        '_'.join([str(x) for x in args.train_group]),
    ])

    model_name = '_'.join([
        "best_model",
        str(int(model_num)),
    ])

    fpath = os.path.join(
        args.artifacts_fpath,
        task,
        'models',
        folder_name,
        model_name
    )

    os.makedirs(fpath,exist_ok=True)
    
    df = pd.DataFrame()
    for i in range(args.n_models):
        
        if all([
            os.path.exists(f"{fpath}/{f}") for f in 
            [f'model_{i}',f'model_{i}_train_scores.csv','hparams.yml']
        ]) and not args.overwrite:

            print("Artifacts exist and args.overwrite is set to False. Skipping...")
            continue

        elif not all([
            os.path.exists(f"{fpath}/{f}") for f in 
            [f'model_{i}',f'model_{i}_train_scores.csv','hparams.yml']
        ]) or args.overwrite: 
        
            m = FixedWidthModel(
                input_dim = features.shape[1], 
                **hparams
            )

            evals_epoch = m.train(train_loaders,phases=['train','val'])['performance']

            # save weights & train scores
            m.save_weights(f"{fpath}/model_{i}")

            evals_epoch.to_csv(
                f"{fpath}/model_{i}_train_scores.csv"
            )

            all_groups = [2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021]
            groups = [args.train_group] + [[y] for y in all_groups if y not in args.train_group]

            for group in groups:
                print(f'Evaluating model in group {group}')
                test_loaders = get_torch_data_loaders(
                    task,
                    group,
                    index_year,
                    row_id_map,
                    features,
                    val_fold='val',
                    test_fold='test'
                )

                idf = m.predict(test_loaders,phases=['test'])['outputs']
                idf['task'] = task
                idf['train_groups'] = '_'.join([str(x) for x in args.train_group])
                idf['test_group'] = '_'.join([str(x) for x in group])
                idf['train_iter'] = i



                df = pd.concat((df,idf))

                yaml.dump(
                    hparams,
                    open(f"{fpath}/hparams.yml","w")
                )
    
    
    # save predictions
    folder_name = '_'.join([
        args.model,
        '_'.join([str(x) for x in args.train_group])
    ])

    file_name = '_'.join([
        "best_model",
        str(int(model_num)),
    ]) 

    fpath = os.path.join(
        args.artifacts_fpath,
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
        # add additional group info from row_id_map
        df = df.merge(
            row_id_map[[
                'features_row_id',
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
            left_on='row_id',
            right_on='features_row_id'
        )

        df.reset_index(drop=True).to_csv(f"{fpath}/{file_name}.csv")