#!/bin/bash

# conda env
source activate /local-scratch/nigam/envs/lguo/temp_ds_shift_robustness

# script dir
cd /local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/clmbr/experiments/clmbr/scripts

## -----------------------------------------------------------
## --------------------- job specification -------------------
## -----------------------------------------------------------

mkdir -p ../logs/clmbr_featurize

YEARS=("2009/2012")

ENCODERS=("GRU" "transformer")
TRAIN_OVERWRITE='False'
FEATURIZE_OVERWRITE='False'

N_GPU=2
N_JOBS=12

## -----------------------------------------------------------
## ----------------------- job pipeline ----------------------
## -----------------------------------------------------------

N_GROUPS=${#YEARS[@]}
N_ENCODERS=${#ENCODERS[@]}

for (( t=0; t<$N_GROUPS; t++ )); do
    for (( i=0; i<$N_ENCODERS; i++ )); do
        python -u train_clmbr.py \
            --year_range=${YEARS[$t]} \
            --encoder=${ENCODERS[$i]} \
            --overwrite="$TRAIN_OVERWRITE" \
            --n_gpu="$N_GPU" \
            --n_jobs="$N_JOBS"
        
        python -u featurize.py \
            --train_group=${YEARS[$t]} \
            --clmbr_encoder=${ENCODERS[$i]} \
            --overwrite="$FEATURIZE_OVERWRITE" \
            >> "../logs/clmbr_featurize/${1:2:2}-${1: -2}-${TASKS[$t]}-$JOB_ID" 
    done
done