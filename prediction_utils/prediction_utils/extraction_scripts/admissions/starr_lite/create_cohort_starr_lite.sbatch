#!/bin/bash
#SBATCH --job-name=create_cohort_starr_lite
#SBATCH --output=/home/spfohl/slurm_out/create_cohort_starr_lite.out
#SBATCH --error=/home/spfohl/slurm_out/create_cohort_starr_lite.err
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=24000
#SBATCH -c 20
#SBATCH -p normal,gpu

source activate prediction_utils

DATASET="starr_omop_cdm5_deid_1pcent_latest"
RS_DATASET="temp_dataset"
DATA_PATH="/share/pi/nigam/projects/spfohl/cohorts/admissions/starr_1pcent_20200525"

GCLOUD_PROJECT="som-nero-phi-nigam-starr"
DATASET_PROJECT="som-rit-phi-starr-prod"
RS_DATASET_PROJECT="som-nero-phi-nigam-starr"

COHORT_NAME="starr_1pcent_20200525"
COHORT_NAME_LABELED="starr_1pcent_20200525"
COHORT_NAME_FILTERED="starr_1pcent_20200525"

python -m prediction_utils.cohorts.admissions.create_cohort \
    --dataset=$DATASET \
    --rs_dataset=$RS_DATASET \
    --cohort_name=$COHORT_NAME \
    --cohort_name_labeled=$COHORT_NAME_LABELED \
    --cohort_name_filtered=$COHORT_NAME_FILTERED \
    --gcloud_project=$GCLOUD_PROJECT \
    --dataset_project=$DATASET_PROJECT \
    --rs_dataset_project=$RS_DATASET_PROJECT \
    --data_path=$DATA_PATH \
    --has_birth_datetime