import os
import argparse
import yaml
import shutil

from subprocess import (run, Popen)
from prediction_utils.util import str2bool
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
    '--infos_fpath',
    type=str,
    default="/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/clmbr/experiments/clmbr/clmbr_artifacts/infos/"
)

parser.add_argument(
    '--models_fpath',
    type=str,
    default="/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/clmbr/experiments/clmbr/clmbr_artifacts/models/"
)

parser.add_argument(
    '--year_end',
    type=int,
    default=2012,
    help="train up to year_end-1, use year_end for validation"
)

parser.add_argument(
    '--excluded_patient_list',
    type=str,
    default="/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/clmbr/experiments/clmbr/data/held_out_patients/excluded_patient_ids.txt"
)

parser.add_argument(
    '--hparams_fpath',
    type=str,
    default="/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/clmbr/experiments/clmbr/hyperparams/clmbr.yml"
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
    '--gpu_num_start',
    type=int,
    default=1
)

if __name__ == "__main__":
    
    args = parser.parse_args()
    
    # create info
    info_dir=os.path.join(args.infos_fpath,str(args.year_end))
    train_end_date=str(args.year_end-1)
    val_end_date=str(args.year_end)
    
    if args.overwrite and os.path.exists(info_dir):
        shutil.rmtree(info_dir, ignore_errors=True)
    
    run([
        'clmbr_create_info',
        f"{args.extracts_fpath}",
        f"{info_dir}",
        f"{train_end_date}-12-31",
        f"{val_end_date}-12-31",
        "--min_patient_count", args.min_patient_count,
        "--excluded_patient_file", args.excluded_patient_list,
    ])
    
    grid = list(
        ParameterGrid(
            yaml.load(
                open(args.hparams_fpath,'r'),
                Loader=yaml.FullLoader
            )
        )
    )
    
    processes=[]
    
    for i,hparams in enumerate(grid):
        
        model_dir=os.path.join(
            args.models_fpath,
            str(args.year_end),
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
            '--code_dropout', f"{hparams['code_dropout']}",
            '--batch_size', f"{hparams['batch_size']}",
            '--device', 'cuda:0',
        ]
        
        processes.append(
            Popen(
                p_args, 
                env=dict(os.environ, CUDA_VISIBLE_DEVICES = str(i%args.n_gpu+args.gpu_num_start))
            )
        )

    for process in processes:
        process.wait()