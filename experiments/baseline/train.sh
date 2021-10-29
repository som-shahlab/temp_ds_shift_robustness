#!/bin/bash
export CUDA_VISIBLE_DEVICES=6

source activate /local-scratch/nigam/envs/lguo/temp_ds_shift_robustness
cd /local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/experiments/baseline/scripts

TRAIN_GROUPS=("2009/2010/2011/2012") #"2009/2010/2011/2012/2013" "2009/2010/2011/2012/2013/2014" "2009/2010/2011/2012/2013/2014/2015" "2009/2010/2011/2012/2013/2014/2015/2016" "2009/2010/2011/2012/2013/2014/2015/2016/2017" "2009/2010/2011/2012/2013/2014/2015/2016/2017/2018")
len=${#TRAIN_GROUPS[@]}

# submit N jobs at a time
N=8;k=0
for (( i=0; i<$len; i++ )); do
    python train_model.py \
        --train_group="${TRAIN_GROUPS[$i]}" &

    let k+=1
    [[ $((k%N)) -eq 0 ]] && wait
done