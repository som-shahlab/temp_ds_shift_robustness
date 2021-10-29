#!/bin/bash

# set GPU device
export CUDA_VISIBLE_DEVICES=6

# conda env
source activate /local-scratch/nigam/envs/lguo/temp_ds_shift_robustness

# script dir
cd /local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/experiments/baseline/scripts

# make log folders if not exist
mkdir -p ../logs/tune
mkdir -p ../logs/train
mkdir -p ../logs/eval

# vars
TRAIN_GROUPS=("2009/2010/2011/2012" "2013" "2014" "2015" "2016" "2017" "2018" "2019" "2020" "2021" "2010/2011/2012/2013" "2011/2012/2013/2014" "2012/2013/2014/2015" "2013/2014/2015/2016" "2014/2015/2016/2017" "2015/2016/2017/2018")
TASKS=("hospital_mortality" "LOS_7" "readmission_30" "icu_admission")
N_GROUPS=${#TRAIN_GROUPS[@]}
N_TASKS=${#TASKS[@]}
N_BOOT=1000

# number of pipes to execute in parallel
# this will exec $N_JOBS * $N_TASKS jobs in parallel
N_JOBS=2

# generate job id
JOB_ID=$(cat /proc/sys/kernel/random/uuid)

# define pipeline
function pipe {
    
    # tune model w/ 5-fold cross-validation
    # executes $N_TASK jobs in parallel
    local k=0
    for (( t=0; t<=$N_TASKS; t++ )); do
    
        python -u tune_model.py \
            --tasks=${TASKS[$t]} \
            --train_group="$1" \
            >> "../logs/tune/${1:2:2}-${TASKS[$t]}-${1: -2}-$JOB_ID" &
        
        let k+=1
        [[ $((k%N_TASKS)) -eq 0 ]] && wait
    done
    
    # train model 
    # executes $N_TASK jobs in parallel
    local k=0
    for (( t=0; t<=$N_TASKS; t++ )); do
    
        python -u train_model.py \
            --tasks=${TASKS[$t]} \
            --train_group="$1" \
            >> "../logs/train/${1:2:2}-${TASKS[$t]}-${1: -2}-$JOB_ID" &
        
        let k+=1
        [[ $((k%N_TASKS)) -eq 0 ]] && wait
    done
    
    # evaluate model
    # executes $N_TASK jobs in parallel
    local k=0
    for (( t=0; t<=$N_TASKS; t++ )); do
    
        python -u eval_model.py \
            --tasks=${TASKS[$t]} \
            --train_group="$1" \
            --n_boot=$N_BOOT \
            >> "../logs/eval/${1:2:2}-${TASKS[$t]}-${1: -2}-$JOB_ID" &
        
        let k+=1
        [[ $((k%N_TASKS)) -eq 0 ]] && wait
    done
}

# execute $N_JOBS pipes in parallel
c=0
for (( i=0; i<$N_GROUPS; i++ )); do    
    
    pipe "${TRAIN_GROUPS[$i]}" &
    
    let c+=1
    [[ $((c%N_JOBS)) -eq 0 ]] && wait
done