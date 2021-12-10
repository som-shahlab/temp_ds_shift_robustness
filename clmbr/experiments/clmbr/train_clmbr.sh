#!/bin/bash

# conda env
source activate /local-scratch/nigam/envs/lguo/temp_ds_shift_robustness

# script dir
cd /local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/clmbr/experiments/clmbr/scripts

## -----------------------------------------------------------
## --------------------- job specification -------------------
## -----------------------------------------------------------
YEARS=(
   "2012" "2013" "2014" "2015"
   "2016" "2017" "2018" "2019"
   "2020" "2021"
)

# whether to re-run 
OVERWRITE='False'

## -----------------------------------------------------------
## ----------------------- job pipeline ----------------------
## -----------------------------------------------------------

N_GROUPS=${#YEARS[@]}

for (( t=0; t<$N_GROUPS; t++ )); do

    python -u train_clmbr.py \
        --year_end=${YEARS[$t]} \
        --overwrite="$OVERWRITE" 

done