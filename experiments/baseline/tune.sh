#!/bin/bash
export CUDA_VISIBLE_DEVICES=6

source activate /local-scratch/nigam/envs/lguo/temp_ds_shift_robustness
cd /local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/experiments/baseline/scripts

# make logs folder if not exist
mkdir -p ../logs/tune

# vars
TRAIN_GROUPS=("2009/2010/2011/2012") #"2009/2010/2011/2012/2013" "2009/2010/2011/2012/2013/2014" "2009/2010/2011/2012/2013/2014/2015" "2009/2010/2011/2012/2013/2014/2015/2016" "2009/2010/2011/2012/2013/2014/2015/2016/2017" "2009/2010/2011/2012/2013/2014/2015/2016/2017/2018")
N_GROUPS=${#TRAIN_GROUPS[@]}
N_FOLDS=5
N_JOBS=5 # max number of jobs in parallel

# tune hparams using N_FOLDS cross-validation
k=0
for (( i=0; i<$N_GROUPS; i++ )); do    
    for (( fold_id=1; fold_id<=$N_FOLDS; fold_id++ )); do
        
        python -u tune_model.py \
            --fold_id=$fold_id \
            --train_group="${TRAIN_GROUPS[$i]}" \
            >> "../logs/tune/$(date +%Y%m%d)" &
        
        let k+=1
        [[ $((k%N_JOBS)) -eq 0 ]] && wait
    done
done
