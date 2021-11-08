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

from prediction_utils.pytorch_utils.datasets import ArrayLoaderGenerator
from prediction_utils.pytorch_utils.group_fairness import group_regularized_model
from prediction_utils.pytorch_utils.robustness import group_robust_model
from prediction_utils.pytorch_utils.metrics import StandardEvaluator

from tune_model import (
    read_file, 
    get_data,
    get_torch_data_loaders,
    get_hparams
)
    

#------------------------------------
# Arg parser
#------------------------------------
parser = argparse.ArgumentParser(
    description = "train models with domain generalization"
)

parser.add_argument(
    "--artifacts_fpath",
    type = str,
    default = "/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/experiments/dg/artifacts",
    help = "path to save artifacts"
)

parser.add_argument(
    "--baseline_artifacts_fpath",
    type=str,
    default='/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/experiments/baseline/artifacts',
    help="path to model hyperparameters - same as baseline NN models"
)

parser.add_argument(
    "--algo_hparams_fpath",
    type=str,
    default='/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/experiments/dg/hyperparams',
    help="path to dg hyperparameters"
)

parser.add_argument(
    "--features_fpath",
    type = str,
    default = "/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/features/admissions",
    help = "path to extracted features"
)

parser.add_argument(
    "--cohort_fpath",
    type = str,
    default = "/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/cohorts/",
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
    "--algo",
    type=str,
    default="irm",
    help="algo to use [irm,dro,coral,adversarial]"
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
    "--n_searches",
    type=int,
    default=50
)

parser.add_argument(
    "--seed",
    type = int,
    default = 44,
    help = "seed for deterministic training"
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
    print(f"algo: {args.algo}")
    
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
    
    # grab hparams
    hparams_grid = get_hparams(task,args)
    
    for j, hparams in enumerate(hparams_grid):
        
        # refit model with all training data
        print(f"Fitting model with hparams {hparams}")

        # prune features
        file_name = '_'.join([
            "nn",
            '_'.join([str(x) for x in args.train_group]),
        ])

        fpath = os.path.join(
            args.baseline_artifacts_fpath,
            task,
            'preprocessor',
        )

        vocab = pd.read_csv(f"{fpath}/{file_name}.csv")
        sel_features = features[:,vocab.index[vocab['keep_feature']==1]]

        # generate torch loaders
        train_loaders = get_torch_data_loaders(
            task,
            args.train_group,
            index_year,
            row_id_map,
            sel_features,
            val_fold='val',
            test_fold='test',
            group_var_name=index_year
        )

        # save path
        folder_name = '_'.join([
            args.algo,
            '_'.join([str(x) for x in args.train_group]),
        ])

        model_name = '_'.join([
            "sweep",
            str(int(j)),
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

            if args.algo=='adversarial':
                hparams['output_dim_discriminator']=len(args.train_group)
                m = group_regularized_model("adversarial")

            elif args.algo=='dro':
                hparams['num_groups']=len(args.train_group)
                m = group_robust_model()

            elif args.algo=='irm':
                m = group_regularized_model("group_irm")

            elif args.algo=='coral':
                m = group_regularized_model("group_coral")

            m = m(
                input_dim = sel_features.shape[1], 
                **hparams
            )

            m.train(train_loaders,phases=['train','val'])

            m.save_weights(f"{fpath}/model_{i}")

            all_groups = [2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021]
            groups = [args.train_group] + [[y] for y in all_groups if y not in args.train_group]

            for group in groups:
                print(f'Evaluating model in group {group}')
                test_loaders = get_torch_data_loaders(
                    task,
                    group,
                    index_year,
                    row_id_map,
                    sel_features,
                    val_fold='val',
                    test_fold='test',
                    group_var_name=index_year
                )

                idf = m.predict(test_loaders,phases=['test'])['outputs']
                idf['task'] = task
                idf['train_groups'] = '_'.join([str(x) for x in args.train_group])
                idf['test_group'] = '_'.join([str(x) for x in group])
                idf['train_iter'] = i
                idf['lambda'] = hparams['lr_lambda'] if args.algo=='dro' else hparams['lambda_group_regularization']

                df = pd.concat((df,idf))

        yaml.dump(
            hparams,
            open(f"{fpath}/hparams.yml","w")
        )

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

        # save predictions
        folder_name = '_'.join([
            args.algo,
            '_'.join([str(x) for x in args.train_group])
        ])

        file_name = '_'.join([
            "sweep",
            str(int(j)),
        ]) 

        fpath = os.path.join(
            args.artifacts_fpath,
            task,
            'pred_probs',
            folder_name
        )

        os.makedirs(fpath, exist_ok=True)

        df.reset_index(drop=True).to_csv(f"{fpath}/{file_name}.csv")