import os
import json
import argparse
import yaml
import shutil
import pdb
import ehr_ml
import joblib
import torch

import pandas as pd
import numpy as np

from itertools import zip_longest
from subprocess import (run, Popen)
from sklearn.model_selection import ParameterGrid

from prediction_utils.util import str2bool
from ehr_ml.clmbr import convert_patient_data


#------------------------------------
# Arg parser
#------------------------------------

parser = argparse.ArgumentParser(
    description='Train end-to-end CLMBR model'
)

parser.add_argument(
    '--task',
    type=str,
    default='hospital_mortality',
    help='task for end-to-end CLMRB training [hospital_mortality, icu_admission, LOS_7, readmission_30]',
)

parser.add_argument(
    '--min_patient_count', 
    type=str,
    default="100",
)

parser.add_argument(
    '--extracts_fpath', 
    type=str,
    default="/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/clmbr/experiments/clmbr/data/extracts/20210723",
)

parser.add_argument(
    '--cohort_fpath',
    type=str,
    default="/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/clmbr/cohorts/admissions/cohort",
)

parser.add_argument(
    '--infos_fpath',
    type=str,
    default="/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/clmbr/experiments/clmbr/clmbr_artifacts/infos/"
)

parser.add_argument(
    '--train_splits_fpath',
    type=str,
    default="/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/clmbr/experiments/clmbr/clmbr_artifacts/train_splits/"
)

parser.add_argument(
    '--labels_fpath',
    type=str,
    default='/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/clmbr/experiments/clmbr/clmbr_artifacts/labels'
)

parser.add_argument(
    '--models_fpath',
    type=str,
    default="/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/clmbr/experiments/clmbr/clmbr_artifacts/models/"
)

parser.add_argument(
    '--year_range',
    type=str,
    default="2009/2012",
    help="start and end of the year range for training year group"
) 

parser.add_argument(
    '--excluded_patient_list',
    type=str,
    default="/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/clmbr/experiments/clmbr/data/held_out_patients/excluded_patient_ids.txt"
)

parser.add_argument(
    '--hparams_fpath',
    type=str,
    default="/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/clmbr/experiments/clmbr/hyperparams/"
)

parser.add_argument(
    '--encoder',
    type=str,
    default='gru',
    help='Encoder type: GRU/Transformer',
)

parser.add_argument(
    '--overwrite',
    type=str2bool,
    default='false'
)

parser.add_argument(
    '--n_gpu',
    type=int,
    default=1
)

parser.add_argument(
    '--n_jobs',
    type=int,
    default=5
)

parser.add_argument(
    '--gpu_num_start',
    type=int,
    default=1
)

#-------------------------------------------------------------------
# CLMBR model (nn.Module) and Training func
#-------------------------------------------------------------------

    
#-------------------------------------------------------------------
# run
#-------------------------------------------------------------------
if __name__ == "__main__":
    
    args = parser.parse_args()
    
    # threads
    torch.set_num_threads(1)
    
    # create train and val patient IDs
    train_splits_dir = os.path.join(
        args.train_splits_fpath,
        "_".join(args.year_range.split("/")) + '_end_to_end',
        args.task,
    )
    
    if args.overwrite and os.path.exists(train_splits_dir):
        shutil.rmtree(train_splits_dir, ignore_errors=True)
        
    os.makedirs(train_splits_dir, exist_ok=True)
    
    year_range_list = [*range(
        int(args.year_range.split("/")[0]), 
        int(args.year_range.split("/")[-1])+1, 
        1,
    )]
    
    df_cohort = pd.read_parquet(
        os.path.join(
            args.cohort_fpath,
            "cohort_split.parquet",
        )
    )
    
    if args.task == 'readmission_30':
        df_cohort = df_cohort.assign(
            date = pd.to_datetime(df_cohort['discharge_date']).dt.date
        )
        
    else:
        df_cohort = df_cohort.assign(
            date = pd.to_datetime(df_cohort['admit_date']).dt.date
        )
    
    
    train = df_cohort.query(
        f"{args.task}_fold_id!=['val','test','ignore'] and admission_year==@year_range_list"
    )
    
    
    
    val = df_cohort.query(
        f"{args.task}_fold_id==['val'] and admission_year==@year_range_list"
    )
    
    # convert patient ids and save to train_splits path
    train_person_ids, train_day_ids = convert_patient_data(
        args.extracts_fpath, 
        train['person_id'], 
        train['date']
    )
    
    with open(
        os.path.join(
            train_splits_dir,
            f"train_patients.txt"
        ), 
        "w"
    ) as f:
        
        for pid in train_person_ids:
            f.write("%d\n" % pid)
    
    
    val_person_ids, val_day_ids = convert_patient_data(
        args.extracts_fpath, 
        val['person_id'], 
        val['date']
    )
    
    with open(
        os.path.join(
            train_splits_dir,
            f"val_patients.txt"
        ), 
        "w"
    ) as f:
        
        for pid in val_person_ids:
            f.write("%d\n" % pid)
    
    
    # create info
    info_dir=os.path.join(
        args.infos_fpath,
        "_".join(args.year_range.split("/"))+'_end_to_end',
        args.task
    )
    
    train_start_date=args.year_range.split("/")[0]
    train_end_date=args.year_range.split("/")[-1]
    val_start_date=args.year_range.split("/")[0]
    val_end_date=args.year_range.split("/")[-1]
    
    if args.overwrite and os.path.exists(info_dir):
        shutil.rmtree(info_dir, ignore_errors=True)
    
    run([
        'clmbr_create_info',
        f"{args.extracts_fpath}",
        f"{info_dir}",
        f"{train_end_date}-12-31",
        f"{val_end_date}-12-31",
        "--train_start_date", f"{train_start_date}-01-01",
        "--val_start_date", f"{val_start_date}-01-01",
        "--min_patient_count", args.min_patient_count,
        "--excluded_patient_file", args.excluded_patient_list,
        "--train_patient_file", f"{train_splits_dir}/train_patients.txt",
        "--val_patient_file", f"{train_splits_dir}/val_patients.txt",
    ])
    
    # create Patient Timeline dataset
    train_labels = train[f"{args.task}"].to_numpy()
    val_labels = val[f"{args.task}"].to_numpy()
    
    train_pred_ids = train['prediction_id'].to_numpy()
    val_pred_ids = val['prediction_id'].to_numpy()
    
    assert(len(train_labels)==len(train_person_ids)==len(train_day_ids)==len(train_pred_ids))
    assert(len(val_labels)==len(val_person_ids)==len(val_day_ids)==len(val_pred_ids))
    
    train_df = pd.DataFrame({
        'labels':train_labels,
        'person_ids': train_person_ids,
        'day_ids': train_day_ids,
        'prediction_ids': train_pred_ids,
    })
    
    val_df = pd.DataFrame({
        'labels':val_labels,
        'person_ids': val_person_ids,
        'day_ids': val_day_ids,
        'prediction_ids': val_pred_ids,
    })
    
    labels_dir = os.path.join(
        args.labels_fpath,
        "_".join(args.year_range.split("/"))+'_end_to_end',
        args.task
    )
    
    os.makedirs(labels_dir,exist_ok=True)
    
    train_df.to_csv(os.path.join(labels_dir,"train.csv"),index=False)
    val_df.to_csv(os.path.join(labels_dir,"val.csv"),index=False)
    
    # get hyperparameter grid
    grid = list(
        ParameterGrid(
            yaml.load(
                open(
                    f"{os.path.join(args.hparams_fpath,args.encoder)}.yml",
                    'r'
                ),
                Loader=yaml.FullLoader
            )
        )
    )
    
    
    processes=[]
    
    # collect args
    for i,hparams in enumerate(grid):
        
        model_dir=os.path.join(
            args.models_fpath,
            "_".join(args.year_range.split("/"))+'_end_to_end',
            args.encoder,
            args.task,
            f"{i}"
        )
        
        if args.overwrite and os.path.exists(model_dir):
            shutil.rmtree(model_dir, ignore_errors=True)
            os.makedirs(model_dir, exist_ok=True)
        
        p_args = [
            'clmbr_train_ete_model',
            model_dir,
            info_dir,
            '--train_labels_path', f"{os.path.join(labels_dir, 'train.csv')}",
            '--val_labels_path', f"{os.path.join(labels_dir, 'val.csv')}",
            '--lr', f"{hparams['lr']}",
            '--encoder_type', f"{hparams['encoder_type']}",
            '--size', f"{hparams['size']}",
            '--dropout', f"{hparams['dropout']}",
            '--batch_size', f"{hparams['batch_size']}",
            '--epochs', f"50",
            '--l2', f"{hparams['l2']}",
            '--warmup_epochs', f"{hparams['warmup_epochs']}",
            '--code_dropout', f"{hparams['code_dropout'] if 'code_dropout' in hparams.keys() else 0.2}",
            '--transformer_layers', f"{hparams['transformer_layers'] if 'transformer_layers' in hparams.keys() else 6}",
            '--device', 'cuda:0',
        ]
        
        processes.append(p_args)
        
    # group processes 
    processes = [
        (
            Popen(
                p,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES = str(i%args.n_gpu+args.gpu_num_start))
            ) for i,p in enumerate(processes)
        )
    ] * args.n_jobs

    # submit n_jobs jobs at a time
    for sub_p in zip_longest(*processes): 
        for p in filter(None, sub_p):
            p.wait()
        