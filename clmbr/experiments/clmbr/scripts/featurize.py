import os
import torch
import bisect
import datetime
import argparse
import sys
import re
import pickle
import gzip

import numpy as np
import pandas as pd
import ehr_ml.clmbr

from ehr_ml.clmbr import convert_patient_data 
from prediction_utils.util import str2bool

#------------------------------------
# Arg parser
#------------------------------------
parser = argparse.ArgumentParser(
    description = "Hyperparameter sweep"
)

parser.add_argument(
    "--extracts_fpath",
    type = str,
    default = "/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/clmbr/experiments/clmbr/data/extracts/20210723",
    help = "path to extracts"
)

parser.add_argument(
    "--artifacts_fpath",
    type = str,
    default = "/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/clmbr/experiments/clmbr/clmbr_artifacts/",
    help = "path to clmbr artifacts including infos and models",
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
    "--train_group", 
    type=str,
    default = "2009/2012",
    help="group(s) to train on [2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]"
)

parser.add_argument(
    "--clmbr_encoder",
    type=str,
    default='gru',
    help='gru/transformer',
)

parser.add_argument(
    "--overwrite",
    type = str2bool,
    default = "false",
    help = "whether to overwrite existing artifacts",
)


#------------------------------------
# Helper Funcs
#------------------------------------
def get_best_model(models_dir):
    
    models=os.listdir(models_dir)
    
    best_loss=np.inf
    best_model=0
    
    for model in models:
        
        df=pd.read_json(
            os.path.join(models_dir,model,'config.json'),
            lines=True
        )
        
        model_losses = open(
            os.path.join(models_dir,model,'losses'),
            'r'
        )
        
        model_best_loss = min([
            float(re.sub("[^.0-9]","",x.split(' ')[-1])) 
            for x in model_losses.readlines() 
            if 'Val' in x and 'nan' not in x
        ])
        
        if model_best_loss<best_loss:
            best_loss=model_best_loss
            best_model=model
            
    return best_model
            
#-------------------------------------------------------------------
# run
#-------------------------------------------------------------------
if __name__ == "__main__":
    
    args = parser.parse_args()
    
    # parse tasks and train_group
    args.tasks = args.tasks.split("/")
    
    args.train_group = [
        x for x in 
        range(
            int(args.train_group.split("/")[0]), 
            int(args.train_group.split("/")[-1])+1,
            1
        )
    ]
    
    clmbr_model_year = f"{args.train_group[0]}_{args.train_group[-1]}"
    
    # save dir
    save_dir=os.path.join(
        args.artifacts_fpath,
        "features",
        "_".join([str(x) for x in args.train_group]),
        args.clmbr_encoder,
    )
    
    # check if files exist
    if all([
        os.path.exists(f"{save_dir}/{f}") for f in 
        ['ehr_ml_patient_ids.gz','prediction_ids.gz','day_indices.gz','labels.gz','features.gz']
    ]) and not args.overwrite:

        print("Artifacts exist and args.overwrite is set to False. Skipping...")
        sys.exit()

    elif not all([
        os.path.exists(f"{save_dir}/{f}") for f in 
        ['ehr_ml_patient_ids.gz','prediction_ids.gz','day_indices.gz','labels.gz','features.gz']
    ]) or args.overwrite: 
    
        os.makedirs(save_dir,exist_ok=True)
    
        best_model_num=get_best_model(
            os.path.join(
                args.artifacts_fpath,
                "models",
                clmbr_model_year,
                args.clmbr_encoder,
            )
        )

        # read dirs
        ehr_ml_extract_dir=args.extracts_fpath

        clmbr_model_dir = os.path.join(
            args.artifacts_fpath,
            "models",
            clmbr_model_year,
            args.clmbr_encoder,
            best_model_num,
        )

        cohort_dir = os.path.join(
            args.cohort_fpath,
            args.cohort_id,
            "cohort",
            "cohort_split.parquet"
        )

        # read cohort
        cohort = pd.read_parquet(cohort_dir)

        ehr_ml_patient_ids={}
        prediction_ids={}
        day_indices={}
        labels={}
        features={}
        
        clmbr_model = ehr_ml.clmbr.CLMBR.from_pretrained(clmbr_model_dir)
        
        for task in args.tasks:

            ehr_ml_patient_ids[task]={}
            prediction_ids[task]={}
            day_indices[task]={}
            labels[task]={}
            features[task]={}
            
            if task == 'readmission_30':
                index_year = 'discharge_year'
            else:
                index_year = 'admission_year'
            
            for fold in ['train','val','test']:
                
                print(f"Featurizing task {task} fold {fold}")
                
                if fold in ['train','val']:

                    if fold=='train':

                        df = cohort.query(
                            f"{task}_fold_id!=['test','val','ignore'] and {index_year}==@args.train_group"
                        ).reset_index()

                    elif fold=='val':

                        df = cohort.query(
                            f"{task}_fold_id==['val'] and {index_year}==@args.train_group"
                        ).reset_index()


                    ehr_ml_patient_ids[task][fold], day_indices[task][fold] = convert_patient_data( 
                        ehr_ml_extract_dir, 
                        df['person_id'], 
                        df['admit_date'].dt.date if task!='readmission_30' else df['discharge_date'].dt.date
                    )

                    labels[task][fold]=df[task]
                    prediction_ids[task][fold]=df['prediction_id']
                    
                    assert (
                        len(ehr_ml_patient_ids[task][fold]) == 
                        len(labels[task][fold]) == 
                        len(prediction_ids[task][fold])
                    )

                    features[task][fold] = clmbr_model.featurize_patients(
                        ehr_ml_extract_dir, 
                        np.array(ehr_ml_patient_ids[task][fold]), 
                        np.array(day_indices[task][fold])
                    )

                else:

                    for year in [
                        args.train_group,
                        2009,2010,2011,2012,
                        2013,2014,2015,2016,
                        2017,2018,2019,2020,
                        2021
                        ]:
                        
                        df = cohort.query(f"{task}_fold_id==['test'] and {index_year}==@year")

                        ehr_ml_patient_ids[task][f'test_{year}'], day_indices[task][f'test_{year}'] = convert_patient_data(
                            ehr_ml_extract_dir, 
                            df['person_id'], 
                            df['admit_date'].dt.date if task!='readmission_30' else df['discharge_date'].dt.date
                        )

                        labels[task][f'test_{year}'] = df[f'{task}'].to_numpy()
                        prediction_ids[task][f'test_{year}']=df['prediction_id']
                        
                        assert (
                            len(ehr_ml_patient_ids[task][f'test_{year}']) == 
                            len(labels[task][f'test_{year}']) == 
                            len(prediction_ids[task][f'test_{year}'])
                        )

                        features[task][f'test_{year}'] = clmbr_model.featurize_patients(
                            ehr_ml_extract_dir, 
                            np.array(ehr_ml_patient_ids[task][f'test_{year}']), 
                            np.array(day_indices[task][f'test_{year}'])
                        )

        # save artifacts  
        pickle.dump(
            ehr_ml_patient_ids,
            gzip.open(os.path.join(save_dir,'ehr_ml_patient_ids.gz'),'wb')
        )

        pickle.dump(
            prediction_ids,
            gzip.open(os.path.join(save_dir,'prediction_ids.gz'),'wb')
        )

        pickle.dump(
            day_indices,
            gzip.open(os.path.join(save_dir,'day_indices.gz'),'wb')
        )

        pickle.dump(
            labels,
            gzip.open(os.path.join(save_dir,'labels.gz'),'wb')
        )

        pickle.dump(
            features,
            gzip.open(os.path.join(save_dir,'features.gz'),'wb')
        )