#!/bin/bash

# conda env
source activate /local-scratch/nigam/envs/lguo/temp_ds_shift_robustness

# script dir
cd /local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/clmbr/experiments/clmbr/scripts

## -----------------------------------------------------------
## --------------------- job specification -------------------
## -----------------------------------------------------------

YEARS=("2009/2012")

#YEARS=(
#   "2009/2012" 
#   "2010/2013" "2011/2014" "2012/2015" "2013/2016"
#   "2014/2017" "2015/2018"
#   "2018/2018" "2017/2018" "2016/2018" "2014/2018"
#   "2013/2018" "2012/2018" "2011/2018" "2010/2018"
#)


ENCODERS=("gru" "transformer")
OVERWRITE='True'
N_GPU=2
N_JOBS=15

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
            --overwrite="$OVERWRITE" \
            --n_gpu="$N_GPU" \
            --n_jobs="$N_JOBS"
    done
done