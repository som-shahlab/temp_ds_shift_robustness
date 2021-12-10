source activate /local-scratch/nigam/envs/lguo/starr

DATASET="starr_omop_cdm5_deid_20210723"
RS_DATASET="lguo_explore"
DATA_PATH="/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/dg/cohorts/admissions"
GCLOUD_PROJECT="som-nero-nigam-starr"
DATASET_PROJECT="som-nero-nigam-starr"
RS_DATASET_PROJECT="som-nero-nigam-starr"
MIN_STAY_HOUR=0

python -m prediction_utils.cohorts.admissions.create_cohort \
    --dataset=$DATASET \
    --rs_dataset=$RS_DATASET \
    --gcloud_project=$GCLOUD \
    --dataset_project=$DATASET_PROJECT \
    --rs_dataset_project=$RS_DATASET_PROJECT \
    --data_path=$DATA_PATH \
    --min_stay_hour=$MIN_STAY_HOUR