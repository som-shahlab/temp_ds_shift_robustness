#!/bin/bash
#SBATCH --job-name=create_cohort_optum_1pcnt
#SBATCH --output=/home/spfohl/slurm_out/create_cohort_optum_1pcnt.out
#SBATCH --error=/home/spfohl/slurm_out/create_cohort_optum_1pcnt.err
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=20000
#SBATCH -c 5
#SBATCH -p normal,gpu

source activate prediction_utils

DATASET="optum_2017q3_cdm_53_1pcnt"
RS_DATASET="temp_dataset"

GCLOUD_PROJECT="som-nero-phi-nigam-starr"
DATASET_PROJECT="som-rit-phi-starr-prod"
RS_DATASET_PROJECT="som-nero-phi-nigam-starr"

HORIZON_LIST="1yr 5yr 10yr"
DATA_PATH="/share/pi/nigam/projects/spfohl/cohorts/ascvd/optum_1pcnt"

for horizon in $HORIZON_LIST
do 
    COHORT_NAME="ascvd_"$horizon"_optum_1pcnt"
    python -m prediction_utils.cohorts.ascvd.starr.create_cohort \
        --dataset=$DATASET \
        --rs_dataset=$RS_DATASET \
        --cohort_name=$COHORT_NAME \
        --horizon=$horizon \
        --gcloud_project=$GCLOUD_PROJECT \
        --dataset_project=$DATASET_PROJECT \
        --rs_dataset_project=$RS_DATASET_PROJECT \
        --data_path=$DATA_PATH \
        --no_has_birth_datetime
done