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
cd /labs/shahlab/projects/lguo/temp_ds_shift_robustness/clmbr/experiments/clmbr/scripts

# make log folders if not exist
mkdir -p ../logs/adapter_train_sweep
mkdir -p ../logs/adapter_eval_sweep

## -----------------------------------------------------------
## --------------------- job specification -------------------
## -----------------------------------------------------------
TRAIN_GROUPS=(
   "2009/2010/2011/2012" 
)

CLMBR_ENCODERS=("transformer")
MODELS=("lr")
TASKS=("hospital_mortality" "LOS_7" "readmission_30" "icu_admission")
N_BOOT=1000

# number of pipes to execute in parallel
# this will exec $N_JOBS * $N_TASKS jobs in parallel
N_JOBS=2

# whether to re-run 
TRAIN_OVERWRITE='False'
EVAL_OVERWRITE='False'

## -----------------------------------------------------------
## ----------------------- job pipeline ----------------------
## -----------------------------------------------------------

N_GROUPS=${#TRAIN_GROUPS[@]}
N_ENCODERS=${#CLMBR_ENCODERS[@]}
N_MODELS=${#MODELS[@]}
N_TASKS=${#TASKS[@]}

# generate job id
JOB_ID=$(cat /proc/sys/kernel/random/uuid)

# define pipeline
function pipe {

    # hyperparameter sweep
    # executes $N_TASK jobs in parallel
    local k=0
    for (( ij=0; ij<$N_MODELS; ij++ )); do
        for (( t=0; t<$N_TASKS; t++ )); do

            python -u train_adapter_sweep.py \
                --tasks=${TASKS[$t]} \
                --clmbr_encoder="$2" \
                --model=${MODELS[$ij]} \
                --train_group="$1" \
                --overwrite="$TRAIN_OVERWRITE" \
                >> "../logs/adapter_train_sweep/${1:2:2}-${1: -2}-${TASKS[$t]}-$JOB_ID" &

            let k+=1
            [[ $((k%N_TASKS)) -eq 0 ]] && wait
        
        done
    done
    
    # evaluate model
    # executes $N_TASK jobs in parallel
    local k=0
    for (( ij=0; ij<$N_MODELS; ij++ )); do
        for (( t=0; t<$N_TASKS; t++ )); do

            python -u evaluate_adapter_sweep.py \
                --tasks=${TASKS[$t]} \
                --clmbr_encoder="$2" \
                --model=${MODELS[$ij]} \
                --train_group="$1" \
                --n_boot=$N_BOOT \
                --overwrite="$EVAL_OVERWRITE" \
                >> "../logs/adapter_eval_sweep/${1:2:2}-${1: -2}-${TASKS[$t]}-$JOB_ID" &

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
    for (( j=0; j<$N_ENCODERS; j++ )); do 
        pipe "${TRAIN_GROUPS[$i]}" "${CLMBR_ENCODERS[$j]}" &

        let c+=1
        [[ $((c%N_JOBS)) -eq 0 ]] && wait
    done
done
