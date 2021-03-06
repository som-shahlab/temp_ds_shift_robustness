import os
import argparse
import pickle
import joblib
import pdb
import yaml
import gzip

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
    "--hparams_fpath",
    type=str,
    default='/labs/shahlab/projects/lguo/temp_ds_shift_robustness/clmbr/experiments/clmbr/hyperparams',
    help="path to hyperparameters"
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
def get_data(features_fpath, cohort_fpath, cohort_id):
    """
    grab data
    """
    
    features=pickle.load(gzip.open(os.path.join(features_fpath,"features.gz"),'rb'))
    prediction_ids=pickle.load(gzip.open(os.path.join(features_fpath,"prediction_ids.gz"),'rb'))
    labels=pickle.load(gzip.open(os.path.join(features_fpath,"labels.gz"),'rb'))
    ehr_ml_patient_ids=pickle.load(gzip.open(os.path.join(features_fpath,"ehr_ml_patient_ids.gz"),'rb'))
    day_indices=pickle.load(gzip.open(os.path.join(features_fpath,"day_indices.gz"),'rb'))
    
    return features,labels,prediction_ids,ehr_ml_patient_ids,day_indices


def get_xy(
    task,
    features,
    labels,
    prediction_ids,
    combine_train_val=False
    ):
    
    if combine_train_val:
        
        X_train=np.concatenate((
            features[task]['train'],
            features[task]['val']
        ))
        
        y_train=np.concatenate((
            labels[task]['train'],
            labels[task]['val']
        ))
        
        prediction_id_train=np.concatenate((
            prediction_ids[task]['train'],
            prediction_ids[task]['val']
        ))
        
        return (X_train,y_train,prediction_id_train)
    
    else:
        X_train=features[task]['train']
        y_train=labels[task]['train']
        X_val=features[task]['val']
        y_val=labels[task]['val']
        prediction_id_train=prediction_ids[task]['train']
        prediction_id_val=prediction_ids[task]['val']

    
        return (X_train,y_train,prediction_id_train,X_val,y_val,prediction_id_val)

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
        
        ## get model hyperparameters
        hparams_grid = get_hparams(args)

        ## get data
        X_train,y_train,prediction_id_train,X_val,y_val,prediction_id_val = get_xy(
            task=task,
            features=features,
            labels=labels,
            prediction_ids=prediction_ids
        )

        ## loop through hyperparameter settings
        for i,hparams in enumerate(hparams_grid):

            print(hparams) 

            ## check if path exists save model & params
            model_name = '_'.join([
                args.clmbr_encoder,
                args.model,
                '_'.join([str(x) for x in args.train_group]),
            ])

            model_num = str(i)

            fpath = os.path.join(
                args.adapter_artifacts_fpath,
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
                    'prediction_id':prediction_id_val,
                    'train_groups':'_'.join([str(x) for x in args.train_group]),
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