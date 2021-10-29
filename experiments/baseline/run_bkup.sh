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
N_GROUPS=${#TRAIN_GROUPS[@]}
N_FOLDS=5
N_BOOT=1000

# number of pipes to execute in parallel
# this will exec $N_JOBS * $N_FOLDS jobs in parallel
N_JOBS=2

# generate job id
JOB_ID=$(cat /proc/sys/kernel/random/uuid)

# define pipeline
function pipe {
    local k=0
    # tune model w/ $N_FOLDS-fold cross-validation
    # executes $N_FOLDS jobs in parallel
    for (( fold_id=1; fold_id<=$N_FOLDS; fold_id++ )); do
    
        python -u tune_model.py \
            --fold_id=$fold_id \
            --train_group="$1" \
            >> "../logs/tune/${1:2:2}-${1: -2}-$JOB_ID" &
        
        let k+=1
        [[ $((k%N_FOLDS)) -eq 0 ]] && wait
    done
    
    # train model 
    python -u train_model.py \
        --train_group="$1" \
        >> "../logs/train/${1:2:2}-${1: -2}-$JOB_ID"; 
    
    # evaluate model
    python -u eval_model.py \
        --train_group="$1" \
        --n_boot=$n_boot \
        >> "../logs/eval/${1:2:2}-${1: -2}-$JOB_ID"; 
}

# execute $N_JOBS pipes in parallel
c=0
for (( i=0; i<$N_GROUPS; i++ )); do    
    
    pipe "${TRAIN_GROUPS[$i]}" &
    
    let c+=1
    [[ $((c%N_JOBS)) -eq 0 ]] && wait
done