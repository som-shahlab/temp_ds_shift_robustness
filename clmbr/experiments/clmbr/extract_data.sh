#!/bin/bash

# conda env
source activate /local-scratch/nigam/envs/lguo/temp_ds_shift_robustness

DATA_LOCATION="/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/clmbr/experiments/clmbr/data/starr_omop/som-nero-nigam-starr.starr_omop_cdm5_deid_20210723"
UMLS_LOCATION="/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/clmbr/experiments/clmbr/data/2020AB/META"
GEM_LOCATION="/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/clmbr/experiments/clmbr/data/gem_mappings"
RXNORM_LOCATION="/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/clmbr/experiments/clmbr/data/RxNorm"
TARGET_LOCATION="/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/clmbr/experiments/clmbr/data/extracts/20210723"

# target location should not exist
rm -rf $TARGET_LOCATION

# --use_quotes needed, otherwise throws segmentation error
ehr_ml_extract_omop \
    $DATA_LOCATION \
    $UMLS_LOCATION \
    $GEM_LOCATION \
    $RXNORM_LOCATION \
    $TARGET_LOCATION \
    --delimiter ',' \
    --use_quotes 
