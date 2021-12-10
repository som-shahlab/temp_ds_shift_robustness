source activate /local-scratch/nigam/envs/lguo/temp_ds_shift_robustness

DATASET="starr_omop_cdm5_deid_20210723"
RS_DATASET="lguo_explore"
DATA_PATH="/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/clmbr/cohorts/admissions"
GCLOUD_PROJECT="som-nero-nigam-starr"
DATASET_PROJECT="som-nero-nigam-starr"
RS_DATASET_PROJECT="som-nero-nigam-starr"
COHORT_NAME="clmbr_admission_rollup"
COHORT_NAME_LABELED="clmbr_admission_rollup_labeled"
COHORT_NAME_FILTERED="clmbr_admission_rollup_filtered"
#FILTER_QUERY="death_date>extract(date from admit_date) AND discharge_date > admit_date_midnight"
FILTER_QUERY=""
MIN_STAY_HOUR=0

python -m prediction_utils.cohorts.admissions.create_cohort \
    --dataset=$DATASET \
    --rs_dataset=$RS_DATASET \
    --gcloud_project=$GCLOUD \
    --dataset_project=$DATASET_PROJECT \
    --rs_dataset_project=$RS_DATASET_PROJECT \
    --data_path=$DATA_PATH \
    --min_stay_hour=$MIN_STAY_HOUR \
    --cohort_name=$COHORT_NAME \
    --cohort_name_labeled=$COHORT_NAME_LABELED \
    --cohort_name_filtered=$COHORT_NAME_FILTERED \
    --filter_query="$FILTER_QUERY"