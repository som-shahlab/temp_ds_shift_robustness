import os
import torch
import bisect
import datetime
import argparse
import sys
import re
import pickle

import numpy as np
import pandas as pd
import ehr_ml.clmbr

from ehr_ml import timeline
from prediction_utils.util import str2bool
from typing import Any, Dict, Optional, Iterable, Tuple, List, Union

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
    default = "2009/2010/2011/2012",
    help="group(s) to train on [2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]"
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

def convert_pid(
    pid: int, search_list: List[int], result_list: List[int]
) -> Tuple[int, int]:
    pid_index = bisect.bisect_left(search_list, pid)
    assert search_list[pid_index] == pid, f"patient ID {pid} not in timeline"
    return_pid = result_list[pid_index]
    return return_pid


def orig2ehr_pid(orig_pid: int, timelines: timeline.TimelineReader):
    all_original_pids = timelines.get_original_patient_ids()
    all_ehr_ml_pids = timelines.get_patient_ids()
    return convert_pid(orig_pid, all_original_pids, all_ehr_ml_pids)


def ehr2orig_pid(ehr_pid: int, timelines: timeline.TimelineReader):
    all_original_pids = timelines.get_original_patient_ids()
    all_ehr_ml_pids = timelines.get_patient_ids()
    return convert_pid(ehr_pid, all_ehr_ml_pids, all_original_pids)


def convert_patient_data(
    extract_dir: str,
    original_patient_ids: Iterable[int],
    dates: Iterable[Union[str, datetime.date]],
) -> Tuple[np.array, np.array]:
    timelines = timeline.TimelineReader(os.path.join(extract_dir, "extract.db"))

    all_original_pids = timelines.get_original_patient_ids()
    all_ehr_ml_pids = timelines.get_patient_ids()

    def get_date_index(pid: int, date_obj: datetime.date) -> int:
        patient = timelines.get_patient(pid)
        for i, day in enumerate(patient.days):
            if date_obj == day.date:
                return i
        assert 0, "should find correct date in timeline!"

    def convert_data(
        og_pid: int, date: Union[str, datetime.date]
    ) -> Tuple[int, int]:
        pid_index = bisect.bisect_left(all_original_pids, og_pid)
        assert (
            all_original_pids[pid_index] == og_pid
        ), f"original patient ID {og_pid} not in timeline"
        ehr_ml_pid = all_ehr_ml_pids[pid_index]

        date_obj = (
            datetime.date.fromisoformat(date) if type(date) == str else date
        )
        assert type(date_obj) == datetime.date
        date_index = get_date_index(ehr_ml_pid, date_obj)
        return ehr_ml_pid, date_index

    ehr_ml_patient_ids = []
    day_indices = []
    og_pids_to_drop = []
    for og_pid, date in zip(original_patient_ids, dates):
        try:
            ehr_ml_pid, date_index = convert_data(og_pid, date)
            ehr_ml_patient_ids.append(ehr_ml_pid)
            day_indices.append(date_index)
        except:
            og_pids_to_drop.append(og_pid)
        
    if len(og_pids_to_drop)>0:
        print(f"{len(og_pids_to_drop)} problematic patient IDs")

    return np.array(ehr_ml_patient_ids), np.array(day_indices), np.array(og_pids_to_drop)

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
            best_model=model
            
    return model
            
#-------------------------------------------------------------------
# run
#-------------------------------------------------------------------
if __name__ == "__main__":
    
    args = parser.parse_args()
    
    # parse tasks and train_group
    args.tasks = args.tasks.split("/")
    args.train_group = [int(x) for x in args.train_group.split("/")]
    clmbr_model_year = args.train_group[-1]
    
    # save dir
    save_dir=os.path.join(
        args.artifacts_fpath,
        "features",
        "_".join([str(x) for x in args.train_group])
    )
    
    # check if files exist
    if all([
        os.path.exists(f"{save_dir}/{f}") for f in 
        ['ehr_ml_patient_ids.pkl','prediction_ids.pkl','day_indices.pkl','labels.pkl','features.pkl']
    ]) and not args.overwrite:

        print("Artifacts exist and args.overwrite is set to False. Skipping...")
        sys.exit()

    elif not all([
        os.path.exists(f"{save_dir}/{f}") for f in 
        ['ehr_ml_patient_ids.pkl','prediction_ids.pkl','day_indices.pkl','labels.pkl','features.pkl']
    ]) or args.overwrite: 
    
        os.makedirs(save_dir,exist_ok=True)
    
        best_model_num=get_best_model(
            os.path.join(
                args.artifacts_fpath,
                "models",
                str(clmbr_model_year)
            )
        )

        # read dirs
        ehr_ml_extract_dir=args.extracts_fpath

        clmbr_model_dir = os.path.join(
            args.artifacts_fpath,
            "models",
            str(clmbr_model_year),
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


                    ehr_ml_patient_ids[task][fold], day_indices[task][fold], og_pids_to_drop = convert_patient_data( 
                        ehr_ml_extract_dir, 
                        df['person_id'], 
                        df['admit_date'].dt.date if task!='readmission_30' else df['discharge_date'].dt.date
                    )

                    if len(og_pids_to_drop)>0:
                        df=df.drop(index=df.query("person_id==@og_pids_to_drop.tolist()").index)
                        ehr_ml_patient_ids[task][fold], day_indices[task][fold], og_pids_to_drop = convert_patient_data( 
                            ehr_ml_extract_dir, 
                            df['person_id'], 
                            df['admit_date'].dt.date if task!='readmission_30' else df['discharge_date'].dt.date
                        )

                    labels[task][fold]=df[task]
                    prediction_ids[task][fold]=df['prediction_id']

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

                        ehr_ml_patient_ids[task][f'test_{year}'], day_indices[task][f'test_{year}'], og_pids_to_drop = convert_patient_data(
                            ehr_ml_extract_dir, 
                            df['person_id'], 
                            df['admit_date'].dt.date if task!='readmission_30' else df['discharge_date'].dt.date
                        )

                        df=df.drop(index=df.query("person_id==@og_pids_to_drop.tolist()").index)

                        if len(og_pids_to_drop)>0:
                            df=df.drop(index=df.query("person_id==@og_pids_to_drop.tolist()").index)
                            ehr_ml_patient_ids[task][f'test_{year}'], day_indices[task][f'test_{year}'], og_pids_to_drop = convert_patient_data(
                                ehr_ml_extract_dir, 
                                df['person_id'], 
                                df['admit_date'].dt.date if task!='readmission_30' else df['discharge_date'].dt.date
                            )

                        labels[task][f'test_{year}'] = df[f'{task}'].to_numpy()
                        prediction_ids[task][f'test_{year}']=df['prediction_id']

                        features[task][f'test_{year}'] = clmbr_model.featurize_patients(
                            ehr_ml_extract_dir, 
                            np.array(ehr_ml_patient_ids[task][f'test_{year}']), 
                            np.array(day_indices[task][f'test_{year}'])
                        )

        # save artifacts  
        pickle.dump(
            ehr_ml_patient_ids,
            open(os.path.join(save_dir,'ehr_ml_patient_ids.pkl'),'wb')
        )

        pickle.dump(
            prediction_ids,
            open(os.path.join(save_dir,'prediction_ids.pkl'),'wb')
        )

        pickle.dump(
            day_indices,
            open(os.path.join(save_dir,'day_indices.pkl'),'wb')
        )

        pickle.dump(
            labels,
            open(os.path.join(save_dir,'labels.pkl'),'wb')
        )

        pickle.dump(
            features,
            open(os.path.join(save_dir,'features.pkl'),'wb')
        )