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
from sklearn.model_selection import ParameterGrid

from prediction_utils.pytorch_utils.datasets import ArrayLoaderGenerator
from prediction_utils.pytorch_utils.group_fairness import group_regularized_model
from prediction_utils.pytorch_utils.robustness import group_robust_model

from prediction_utils.util import str2bool

#------------------------------------
# Arg parser
#------------------------------------
parser = argparse.ArgumentParser(
    description = "Train FCNN models with domain generalization across various hyperparameter settings"
)

parser.add_argument(
    "--artifacts_fpath",
    type = str,
    default = "/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/dg/experiments/dg/artifacts",
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
    "--baseline_artifacts_fpath",
    type=str,
    default='/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/dg/experiments/baseline/artifacts',
    help="path to model hyperparameters - same as baseline NN models"
)

parser.add_argument(
    "--algo_hparams_fpath",
    type=str,
    default='/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/dg/experiments/dg/hyperparams',
    help="path to dg hyperparameters"
)

parser.add_argument(
    "--algo",
    type=str,
    default="irm",
    help="algo to use [irm,dro,coral,adversarial]"
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
    "--prep_prune_thresh",
    type=int,
    default=25,
    help="minimum number of observations for each feature"
)

parser.add_argument(
    "--n_fold",
    type=int,
    default=5,
    help="fold_id"
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


def get_torch_data_loaders(
    task,
    train_group,
    index_year,
    row_id_map,
    features,
    val_fold,
    test_fold=['test'],
    group_var_name=None,
    ignore_fold=['ignore'],
):
    
    config_dict = {
        'batch_size': 512,
        'sparse_mode': 'list',
        'row_id_col': 'features_row_id',
        'input_dim': features.shape[1],
        'label_col': task,
        'fold_id': val_fold,
        'fold_id_test':test_fold,
        'group_var_name':group_var_name,
        'include_group_in_dataset':True if group_var_name is not None else False
    }
    
    # grab only data from train_group & remove rows with "ignored" flag
    # "ignored" flags are added to readmission prediction to remove those
    # that have died in the hospital
    input_row_id_map = row_id_map.query(
        f"{index_year} == @train_group and \
        {task}_fold_id != @ignore_fold"
    )
    
    input_row_id_map = input_row_id_map.assign(
        fold_id=input_row_id_map[f"{task}_fold_id"]
    )

    loader_generator = ArrayLoaderGenerator(
        cohort=input_row_id_map, features = features, **config_dict
    )
    
    return loader_generator.init_loaders()

def get_hparams(task,args):
    
    # load algo hparam grid
    algo_param_grid = yaml.load(
        open(
            os.path.join(
                args.algo_hparams_fpath,
                f"{args.algo}.yml"
            ),
            'r'
        ), 
        Loader = yaml.FullLoader
    )
    
    # load best model hparam setting
    fpath = os.path.join(
        args.baseline_artifacts_fpath,
        f"{task}",
        "models",
        '_'.join([
            "nn",
            '_'.join([str(x) for x in args.train_group]),
        ])
    )
    
    model_name = [x for x in os.listdir(fpath) if "best_model" in x] 
    assert(len(model_name)==1)
    
    model_param = yaml.load(
        open(
            os.path.join(
                fpath,
                model_name[0],
                "hparams.yml",
            ),
            'r'
        ), 
        Loader = yaml.FullLoader
    )
    
    param_grid = list(ParameterGrid(algo_param_grid))
    np.random.shuffle(param_grid)
    param_grid = [{**model_param, **x} for x in param_grid]
    
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
        features = features[:,vocab.index[vocab['keep_feature']==1]]
        
        ## get model hyperparameters
        hparams_grid = get_hparams(task,args)
        
        folds = row_id_map[f"{task}_fold_id"].unique().tolist()
        folds = [x for x in folds if x not in ['ignore','val','test']]
        
        for fold in folds:
        
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
                group_var_name=index_year,
                ignore_fold=['ignore','val'],
            )

            ## loop through hyperparameter settings
            for i,hparams in enumerate(hparams_grid):

                print(hparams) 
                
                ## check if path exists save model & params
                model_name = '_'.join([
                    args.algo,
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
                
                if all([
                    os.path.exists(f"{fpath}/{f}") for f in 
                    ['model','train_scores.csv','hparams.yml','val_pred_probs.csv']
                ]) and not args.overwrite:
                    
                    print("Artifacts exist and args.overwrite is set to False. Skipping...")
                    continue
                
                elif not all([
                    os.path.exists(f"{fpath}/{f}") for f in 
                    ['model','train_scores.csv','hparams.yml','val_pred_probs.csv']
                ]) or args.overwrite: 
                    
                    os.makedirs(fpath,exist_ok=True)
                    
                    ## train & get outputs
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
                        input_dim = features.shape[1], 
                        **hparams
                    )

                    evals_epoch = m.train(loaders,phases=['train','val'])['performance']

                    df = m.predict(loaders,phases=['val'])['outputs']

                    df['task'] = task
                    df['algo'] = args.algo
                    df['train_groups'] = '_'.join([str(x) for x in args.train_group])

                    ## save weights, hparams, train_scores, and predictions on validation set
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

