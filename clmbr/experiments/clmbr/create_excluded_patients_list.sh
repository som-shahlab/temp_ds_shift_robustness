#!/bin/bash

# conda env
source activate /local-scratch/nigam/envs/lguo/temp_ds_shift_robustness

cd /local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/clmbr/experiments/clmbr/scripts

EXTRACTS_FPATH="/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/clmbr/experiments/clmbr/data/extracts/20210723"
COHORT_FPATH="/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/clmbr/cohorts/admissions/cohort/"
COHORT_FNAME="cohort_split.parquet"
TARGET_FPATH="/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/clmbr/experiments/clmbr/data/held_out_patients/"
TARGET_FNAME="excluded_patient_ids"

python create_excluded_patient_list.py \
    --extracts_fpath=$EXTRACTS_FPATH \
    --cohort_fpath=$COHORT_FPATH \
    --cohort_fname=$COHORT_FNAME \
    --target_fpath=$TARGET_FPATH \
    --target_fname=$TARGET_FNAME