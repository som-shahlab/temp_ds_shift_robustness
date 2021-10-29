import os
import argparse
import pickle
import joblib
import pdb
import yaml
import torch

import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix as csr
from sklearn.linear_model import LogisticRegression as lr
from sklearn.model_selection import ParameterGrid

from prediction_utils.pytorch_utils.datasets import ArrayLoaderGenerator
from prediction_utils.pytorch_utils.models import FixedWidthModel

#------------------------------------
# Arg parser
#------------------------------------
parser = argparse.ArgumentParser(
    description = "Train Logistic Regression or NN Models"
)

parser.add_argument(
    "--artifacts_fpath",
    type = str,
    default = "/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/experiments/baseline/artifacts",
    help = "path to save artifacts"
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
    "--hparams_fpath",
    type=str,
    default='/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/experiments/baseline/hyperparams',
    help="path to hyperparameters"
)

parser.add_argument(
    "--model",
    type=str,
    default="nn",
    help="model to train [nn,lr]"
)

parser.add_argument(
    "--train_group", 
    type=str,
    default = "2009/2010/2011/2012",
    help="group(s) to train on [2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]"
)

parser.add_argument(
    "--n_searches",
    type=int,
    default=100,
    help="number of random searches to conduct for hparam search"
)

parser.add_argument(
    "--n_fold",
    type=int,
    default=5,
    help="number of cross-validation folds"
)

parser.add_argument(
    "--prep_prune_thresh",
    type=int,
    default=25,
    help="minimum number of observations for each feature"
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
def read_file(filename, columns=None, **kwargs):
    
    load_extension = os.path.splitext(filename)[-1]
    if load_extension == ".parquet":
        return pd.read_parquet(filename, columns=columns,**kwargs)
    elif load_extension == ".csv":
        return pd.read_csv(filename, usecols=columns, **kwargs)

    
def get_data(features_fpath, cohort_fpath, cohort_id):
    """
    grab data
    """
    vocab = read_file(
        f"{features_fpath}/vocab/vocab.parquet", 
        engine="pyarrow"
    )

    row_id_map = read_file(
        f'{features_fpath}/features_sparse/features_row_id_map.parquet',
        engine='pyarrow'
    )

    features = joblib.load(
        f'{features_fpath}/features_sparse/features.gz', 
    )

    cohort = read_file(
        f'{cohort_fpath}/{cohort_id}/cohort/cohort_split.parquet',
        engine='pyarrow'
    )

    row_id_map = row_id_map.merge(
        cohort,
        left_on = 'prediction_id',
        right_on = 'prediction_id'
    )
    
    return vocab, features, row_id_map


def get_torch_data_loaders(task,train_group,index_year,row_id_map,features,val_fold,test_fold=['test']):
    
    config_dict = {
        'batch_size': 256,
        'sparse_mode': 'list',
        'row_id_col': 'features_row_id',
        'input_dim': features.shape[1],
        'label_col': task,
        'fold_id': val_fold,
        'fold_id_test':test_fold,
    }
    
    # grab only data from train_group & remove rows with "ignored" flag
    # "ignored" flags are added to readmission prediction to remove those
    # that have died in the hospital
    input_row_id_map = row_id_map.query(
        f"{index_year} == @train_group and \
        {task}_fold_id != ['ignore']"
    )
    
    input_row_id_map = input_row_id_map.assign(
        fold_id=input_row_id_map[f"{task}_fold_id"]
    )

    loader_generator = ArrayLoaderGenerator(
        cohort=input_row_id_map, features = features, **config_dict
    )
    
    return loader_generator.init_loaders()

def get_hparams(args):
    
    param_grid = yaml.load(
        open(
            os.path.join(
                args.hparams_fpath,
                f"{args.model}.yml"
            ),
            'r'
        ), 
        Loader = yaml.FullLoader
    )
    
    param_grid = list(ParameterGrid(param_grid))
    np.random.shuffle(param_grid)
    
    if args.n_searches < len(param_grid):
        param_grid=param_grid[:args.n_searches]
    
    return param_grid
#-------------------------------------------------------------------
# run
#-------------------------------------------------------------------
if __name__ == "__main__":
    
    args = parser.parse_args()
    
    # set seed
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    joblib.Parallel(n_jobs=1)
    
    # parse tasks and train_group
    args.tasks = args.tasks.split("/")
    args.train_group = [int(x) for x in args.train_group.split("/")]

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
        
        # prune features using train set
        all_train_ids = row_id_map.query(
            f"{index_year} == @args.train_group and \
            {task}_fold_id != ['test','val','ignore']"
        )
        vocab['keep_feature'] = np.array(features[all_train_ids['features_row_id']].sum(
            axis=0
        )>args.prep_prune_thresh*1)[0]
        
        # save pruned features list
        file_name = '_'.join([
            args.model,
            '_'.join([str(x) for x in args.train_group]),
        ])
        
        fpath = os.path.join(
            args.artifacts_fpath,
            task,
            'preprocessor',
        )

        os.makedirs(fpath,exist_ok=True)
        
        vocab.to_csv(f"{fpath}/{file_name}.csv")
        
        # prune features
        features = features[:,vocab.index[vocab['keep_feature']==1]]
        
        ## get model hyperparameters
        hparams_grid = get_hparams(args)
        
        for fold in range(1,args.n_fold+1):
            
            # train models
            print(f"fold number {fold}") 

            ## get data loaders
            loaders = get_torch_data_loaders(
                task,
                args.train_group,
                index_year,
                row_id_map,
                features,
                fold,
                test_fold=['test','val'],
            )

            ## loop through hyperparameter settings
            for i,hparams in enumerate(hparams_grid):

                print(hparams) 

                ## train & get outputs
                m = FixedWidthModel(
                    input_dim = features.shape[1], 
                    **hparams
                )

                evals_epoch = m.train(loaders,phases=['train','val'])['performance']

                df = m.predict(loaders,phases=['val'])['outputs']

                df['task'] = task
                df['train_groups'] = '_'.join([str(x) for x in args.train_group])

                # save model & params
                model_name = '_'.join([
                    args.model,
                    '_'.join([str(x) for x in args.train_group]),
                ])

                model_num = '_'.join([
                    str(i),
                    f"fold{fold}",
                ])

                fpath = os.path.join(
                    args.artifacts_fpath,
                    task,
                    'models',
                    model_name,
                    model_num
                )

                os.makedirs(fpath,exist_ok=True)

                m.save_weights(f"{fpath}/model")

                yaml.dump(
                    hparams,
                    open(f"{fpath}/hparams.yml","w")
                )

                evals_epoch.to_csv(
                    f"{fpath}/train_scores.csv"
                )

                df.reset_index(drop=True).to_csv(
                    f"{fpath}/val_pred_probs.csv"
                )

