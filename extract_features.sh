source activate /local-scratch/nigam/envs/lguo/starr
DATASET="starr_omop_cdm5_deid_20210723"
RS_DATASET="lguo_explore"
COHORT_NAME="admission_rollup_filtered_temp"
DATA_PATH="/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/features/admissions"

FEATURES_DATASET="temp_dataset"
GCLOUD_PROJECT="som-nero-nigam-starr"
DATASET_PROJECT="som-nero-nigam-starr"
RS_DATASET_PROJECT="som-nero-nigam-starr"
FEATURES_PREFIX="features_"$USER
#INDEX_DATE_FIELD='admit_date'
ROW_ID_FIELD='prediction_id'
MERGED_NAME='merged_features_binary'

python -m prediction_utils.extraction_utils.extract_features \
    --data_path="/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/features/admissions_admission" \
    --features_by_analysis_path="features_by_analysis" \
    --dataset=$DATASET \
    --rs_dataset=$RS_DATASET \
    --cohort_name=$COHORT_NAME \
    --gcloud_project=$GCLOUD_PROJECT \
    --dataset_project=$DATASET_PROJECT \
    --rs_dataset_project=$RS_DATASET_PROJECT \
    --features_dataset=$FEATURES_DATASET \
    --features_prefix=$FEATURES_PREFIX \
    --index_date_field="admit_date" \
    --row_id_field=$ROW_ID_FIELD \
    --merged_name=$MERGED_NAME \
    --exclude_analysis_ids "note_nlp" "note_nlp_dt" "note_nlp_delayed" \
    --time_bins "-36500" "-30" "-7" "0" \
    --time_bins_hourly "-24" "0" \
    --binary \
    --featurize \
    --no_cloud_storage \
    --merge_features \
    --create_sparse \
    --no_create_parquet \
    --overwrite &&
    
python -m prediction_utils.extraction_utils.extract_features \
    --data_path="/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/features/admissions_discharge" \
    --features_by_analysis_path="features_by_analysis" \
    --dataset=$DATASET \
    --rs_dataset=$RS_DATASET \
    --cohort_name=$COHORT_NAME \
    --gcloud_project=$GCLOUD_PROJECT \
    --dataset_project=$DATASET_PROJECT \
    --rs_dataset_project=$RS_DATASET_PROJECT \
    --features_dataset=$FEATURES_DATASET \
    --features_prefix=$FEATURES_PREFIX \
    --index_date_field="discharge_date" \
    --row_id_field=$ROW_ID_FIELD \
    --merged_name=$MERGED_NAME \
    --exclude_analysis_ids "note_nlp" "note_nlp_dt" "note_nlp_delayed" \
    --time_bins "-36500" "-30" "-7" "0" \
    --time_bins_hourly "-24" "0" \
    --binary \
    --featurize \
    --no_cloud_storage \
    --merge_features \
    --create_sparse \
    --no_create_parquet \
    --overwrite
