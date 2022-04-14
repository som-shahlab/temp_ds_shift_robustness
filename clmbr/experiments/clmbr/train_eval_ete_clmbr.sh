#!/bin/bash

# conda env
source activate /local-scratch/nigam/envs/lguo/temp_ds_shift_robustness

# script dir
cd /local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/clmbr/experiments/clmbr/scripts

## -----------------------------------------------------------
## --------------------- job specification -------------------
## -----------------------------------------------------------

TASKS=("hospital_mortality" "LOS_7" "icu_admission" "readmission_30")

ENCODERS=("GRU" "transformer")
TRAIN_OVERWRITE='False'
EVAL_OVERWRITE='False'

N_GPU=2
N_JOBS=12

## -----------------------------------------------------------
## ----------------------- job pipeline ----------------------
## -----------------------------------------------------------

N_TASKS=${#TASKS[@]}
N_ENCODERS=${#ENCODERS[@]}

for (( t=0; t<$N_TASKS; t++ )); do
    for (( i=0; i<$N_ENCODERS; i++ )); do
        python -u train_ete_clmbr.py \
            --task=${TASKS[$t]} \
            --encoder=${ENCODERS[$i]} \
            --overwrite="$TRAIN_OVERWRITE" \
            --n_gpu="$N_GPU" \
            --n_jobs="$N_JOBS"
    done
done

python -u evaluate_ete_clmbr.py \
    --clmbr_encoder=${ENCODERS[$i]} \
    --overwrite="$EVAL_OVERWRITE" \
    --n_jobs=1