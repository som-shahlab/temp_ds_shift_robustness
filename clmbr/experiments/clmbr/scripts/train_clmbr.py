import os
import argparse
import yaml
import shutil
import pdb
import torch
import joblib

import pandas as pd
import numpy as np

from itertools import zip_longest
from subprocess import (run, Popen)
from prediction_utils.util import str2bool
from ehr_ml.clmbr import convert_patient_data
from sklearn.model_selection import ParameterGrid

parser = argparse.ArgumentParser(
    description='Train CLMBR model'
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
    default=2
)

parser.add_argument(
    '--n_jobs',
    type=int,
    default=12
)

parser.add_argument(
    '--gpu_num_start',
    type=int,
    default=1
)

if __name__ == "__main__":
    
    args = parser.parse_args()
    
    # threads
    torch.set_num_threads(1)
    joblib.Parallel(n_jobs=1)
    
    # create train and val patient IDs
    train_splits_dir = os.path.join(
        args.train_splits_fpath,
        "_".join(args.year_range.split("/"))
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
    
    df_cohort = df_cohort.assign(
        date = pd.to_datetime(df_cohort['admit_date']).dt.date
    )
    
    train = df_cohort.query(
        "fold_id!=['val','test','1'] and admission_year==@year_range_list"
    )
    
    val = df_cohort.query(
        "fold_id==['1'] and admission_year==@year_range_list"
    )
    
    # convert patient ids and save to train_splits path
    train_ids, _ = convert_patient_data(
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
        
        for pid in train_ids:
            f.write("%d\n" % pid)
    
    
    val_ids, _ = convert_patient_data(
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
        
        for pid in val_ids:
            f.write("%d\n" % pid)
    
    
    # create info
    info_dir=os.path.join(args.infos_fpath,"_".join(args.year_range.split("/")))
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
            "_".join(args.year_range.split("/")),
            args.encoder,
            f"{i}"
        )
        
        if args.overwrite and os.path.exists(model_dir):
            shutil.rmtree(model_dir, ignore_errors=True)
            os.makedirs(model_dir, exist_ok=True)
        
        p_args = [
            'clmbr_train_model',
            model_dir,
            info_dir,
            '--lr', f"{hparams['lr']}",
            '--encoder_type', f"{hparams['encoder_type']}",
            '--size', f"{hparams['size']}",
            '--dropout', f"{hparams['dropout']}",
            '--batch_size', f"{hparams['batch_size']}",
            '--epochs', f"{hparams['epochs']}",
            '--l2', f"{hparams['l2']}",
            '--warmup_epochs', f"{hparams['warmup_epochs']}",
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