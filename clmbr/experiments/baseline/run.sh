#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lawrence.guo91@gmail.com
#SBATCH --time=5-00:00 # Runtime in D-HH:MM
#SBATCH --job-name=count_models_baseline
#SBATCH --nodes=1 
#SBATCH -n 16 #number of cores to reserve, default is 1
#SBATCH --mem=32000 # in MegaBytes. default is 8 GB
#SBATCH --partition=shahlab # Partition allocated for the lab
#SBATCH --error=logs/error-sbatchjob.%J.err
#SBATCH --output=logs/out-sbatchjob.%J.out

# conda env
source activate /labs/shahlab/envs/lguo/temp_ds_shift_robustness

# script dir
cd /labs/shahlab/projects/lguo/temp_ds_shift_robustness/clmbr/experiments/baseline/scripts

# make log folders if not exist
# mkdir -p ../logs/tune
# mkdir -p ../logs/train
# mkdir -p ../logs/eval

## -----------------------------------------------------------
## --------------------- job specification -------------------
## -----------------------------------------------------------
TRAIN_GROUPS=(
    "2009/2010/2011/2012" 
    "2013" "2014" "2015" "2016" "2017" 
    "2018" "2019" "2020" "2021" 
)

MODELS=("lr" "gbm")
TASKS=("hospital_mortality" "LOS_7" "readmission_30" "icu_admission")
N_BOOT=1000

# number of pipes to execute in parallel
# this will exec $N_JOBS * $N_TASKS jobs in parallel
N_JOBS=2

# whether to re-run 
TUNE_OVERWRITE='False'
TRAIN_OVERWRITE='False'
EVAL_OVERWRITE='False'

## -----------------------------------------------------------
## ----------------------- job pipeline ----------------------
## -----------------------------------------------------------

N_GROUPS=${#TRAIN_GROUPS[@]}
N_TASKS=${#TASKS[@]}
N_MODELS=${#MODELS[@]}

# generate job id
JOB_ID=$(cat /proc/sys/kernel/random/uuid)

# define pipeline
function pipe {
    
    # hyperparameter sweep
    # executes $N_TASK jobs in parallel
    local k=0
    for (( ij=0; ij<$N_MODELS; ij++ )); do
        for (( t=0; t<$N_TASKS; t++ )); do

            python -u tune_model.py \
                --tasks=${TASKS[$t]} \
                --model=${MODELS[$ij]} \
                --train_group="$1" \
                --overwrite="$TUNE_OVERWRITE" 

            let k+=1
            [[ $((k%N_TASKS)) -eq 0 ]] && wait
        
        done
    done
    
    # train model 
    # executes $N_TASK jobs in parallel
    local k=0
    for (( ij=0; ij<$N_MODELS; ij++ )); do
        for (( t=0; t<$N_TASKS; t++ )); do

            python -u train_model.py \
                --tasks=${TASKS[$t]} \
                --model=${MODELS[$ij]} \
                --train_group="$1" \
                --overwrite="$TRAIN_OVERWRITE" 

            let k+=1
            [[ $((k%N_TASKS)) -eq 0 ]] && wait
            
        done
    done
    
    # evaluate model
    # executes $N_TASK jobs in parallel
    local k=0
    for (( ij=0; ij<$N_MODELS; ij++ )); do
        for (( t=0; t<$N_TASKS; t++ )); do

            python -u eval_model.py \
                --tasks=${TASKS[$t]} \
                --model=${MODELS[$ij]} \
                --train_group="$1" \
                --n_boot=$N_BOOT \
                --overwrite="$EVAL_OVERWRITE" 

            let k+=1
            [[ $((k%N_TASKS)) -eq 0 ]] && wait
        done
    done
}

## -----------------------------------------------------------
## ----------------------- execute job -----------------------
## -----------------------------------------------------------
# execute $N_JOBS pipes in parallel
c=0
for (( i=0; i<$N_GROUPS; i++ )); do    
    
    pipe "${TRAIN_GROUPS[$i]}" &
    
    let c+=1
    [[ $((c%N_JOBS)) -eq 0 ]] && wait
done