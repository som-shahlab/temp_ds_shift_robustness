#!/bin/bash

# conda env
source activate /local-scratch/nigam/envs/lguo/temp_ds_shift_robustness

cd /local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/clmbr/experiments/clmbr/scripts

PROJECT="som-nero-nigam-starr"
DATASET="starr_omop_cdm5_deid_20210723"
TARGET_FPATH="/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/clmbr/experiments/clmbr/data/starr_omop"

python download_bq.py \
    --project=$PROJECT \
    --dataset=$DATASET \
    --target_fpath=$TARGET_FPATH