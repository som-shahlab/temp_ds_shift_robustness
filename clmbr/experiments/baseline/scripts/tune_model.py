import os
import argparse
import pickle
import joblib
import pdb
import yaml

import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix as csr
from sklearn.linear_model import LogisticRegression as lr
from sklearn.model_selection import ParameterGrid
from lightgbm import LGBMClassifier as gbm

from prediction_utils.util import str2bool

#------------------------------------
# Arg parser
#------------------------------------
parser = argparse.ArgumentParser(
    description = "Hyperparameter sweep"
)

parser.add_argument(
    "--artifacts_fpath",
    type = str,
    default = "/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/clmbr/experiments/baseline/artifacts",
    help = "path to save artifacts"
)

parser.add_argument(
    "--features_fpath",
    type = str,
    default = "/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/clmbr/features/admissions",
    help = "path to extracted features"
)

parser.add_argument(
    "--cohort_fpath",
    type = str,
    default = "/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/clmbr/cohorts/",
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
    default='/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/clmbr/experiments/baseline/hyperparams',
    help="path to hyperparameters"
)

parser.add_argument(
    "--model",
    type=str,
    default="lr",
    help="model to train [lr,gbm]"
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
    "--n_jobs",
    type=int,
    default=4,
    help="number of threads"
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


def get_xy(task,train_group,index_year,row_id_map,features,combine_train_val=False):
    
    features = features.asfptype()
    
    if combine_train_val:
        train_ids = row_id_map.query(
            f"{index_year} == @train_group and \
            {task}_fold_id != ['test','ignore']"
        )
        
        val_ids = row_id_map.query(
            f"{index_year} == @train_group and \
            {task}_fold_id == ['test']"
        )
    
    else:    
        train_ids = row_id_map.query(
            f"{index_year} == @train_group and \
            {task}_fold_id != ['val','test','ignore']"
        )
        val_ids = row_id_map.query(
            f"{index_year} == @train_group and \
            {task}_fold_id == ['val']"
        )
    
    X_train,y_train=features[train_ids.index,:],train_ids[task].values
    X_val,y_val=features[val_ids.index,:],val_ids[task].values
    row_id_train,row_id_val=train_ids['features_row_id'].values,val_ids['features_row_id'].values
    
    return (X_train,y_train,row_id_train,X_val,y_val,row_id_val)

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
    np.random.seed(args.seed)
    joblib.Parallel(n_jobs=args.n_jobs)
    
    # parse tasks and train_group
    args.tasks = args.tasks.split("/")
    args.train_group = [int(x) for x in args.train_group.split("/")]

    for task in args.tasks:
        
        print(f"task: {task}")
        
        # assign index_year & features_fpath
        if task == 'readmission_30':
            index_year = 'discharge_year'
            features_fpath = args.features_fpath+'_discharge_midnight/merged_features_binary'
        else:
            index_year = 'admission_year'
            features_fpath = args.features_fpath+'_admission_midnight/merged_features_binary'
        
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

        ## get data
        X_train,y_train,row_id_train,X_val,y_val,row_id_val = get_xy(
            task=task,
            train_group=args.train_group,
            index_year=index_year,
            row_id_map=row_id_map,
            features=features
        )

        ## loop through hyperparameter settings
        for i,hparams in enumerate(hparams_grid):

            print(hparams) 

            ## check if path exists save model & params
            model_name = '_'.join([
                args.model,
                '_'.join([str(x) for x in args.train_group]),
            ])

            model_num = str(i)

            fpath = os.path.join(
                args.artifacts_fpath,
                task,
                'models',
                model_name,
                model_num
            )

            os.makedirs(fpath,exist_ok=True)

            if all([
                os.path.exists(f"{fpath}/{f}") for f in 
                ['model.pkl','hparams.yml','val_pred_probs.csv']
            ]) and not args.overwrite:

                print("Artifacts exist and args.overwrite is set to False. Skipping...")
                continue

            elif not all([
                os.path.exists(f"{fpath}/{f}") for f in 
                ['model.pkl','hparams.yml','val_pred_probs.csv']
            ]) or args.overwrite: 

                ## train & get outputs
                if args.model=='lr':
                    
                    m = lr(
                        n_jobs=args.n_jobs,
                        **hparams
                    )

                    m.fit(X_train,y_train)
                
                elif args.model=='gbm':
                    
                    m = gbm(
                        n_jobs=args.n_jobs, 
                        **hparams
                    )
                    
                    eval_set = [(X_val, y_val)]
                    
                    m.fit(
                        X_train,
                        y_train,
                        eval_set=eval_set,
                        early_stopping_rounds=10,
                    )
                    
                    # get actual number of fitted trees
                    hparams['n_estimators']=m._Booster.num_trees()

                df = pd.DataFrame({
                    'pred_probs':m.predict_proba(X_val)[:,1],
                    'labels':y_val,
                    'task':task,
                    'row_id':row_id_val,
                    'train_groups':'_'.join([str(x) for x in args.train_group])
                })

                # save
                pickle.dump(
                    m,
                    open(f"{fpath}/model.pkl","wb")
                )

                yaml.dump(
                    hparams,
                    open(f"{fpath}/hparams.yml","w")
                )

                df.reset_index(drop=True).to_csv(
                    f"{fpath}/val_pred_probs.csv"
                )

