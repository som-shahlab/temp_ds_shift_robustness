
DATA_PATH="/share/pi/nigam/projects/sepsis/extraction_201003"
FEATURES_PATH=$DATA_PATH"/merged_features_binary/features_sparse/features.gz"
COHORT_PATH=$DATA_PATH"/cohort/cohort_cv.parquet"
VOCAB_PATH=$DATA_PATH"/merged_features_binary/vocab/vocab.parquet"
FEATURES_ROW_ID_MAP_PATH=$DATA_PATH"/merged_features_binary/features_sparse/features_row_id_map.parquet"

python -m prediction_utils.pytorch_utils.train_model \
    --data_path=$DATA_PATH \
    --features_path=$FEATURES_PATH \
    --cohort_path=$COHORT_PATH \
    --vocab_path=$VOCAB_PATH \
    --features_row_id_map=$FEATURES_ROW_ID_MAP_PATH \
    --data_mode='array_alt' \
    --regularization_metric="group_irm" \
    --run_evaluation_group \
    --sensitive_attribute="adult_at_admission" \
    --eval_attributes="adult_at_admission" \
    --label_col='early_sepsis' \
    --num_epochs=150 \
    --num_hidden=1 \
    --hidden_dim=256 \
    --batch_size=256 \
    --early_stopping \
    --early_stopping_patience=25 \
    --drop_prob=0.85 \
    --lr=0.0001 \
    --experiment_name="scratch" \
    --fold_id="1" \
    --lambda_group_regularization=1
    

# python -m prediction_utils.pytorch_utils.train_model \
#     --data_path=$DATA_PATH \
#     --features_path=$FEATURES_PATH \
#     --cohort_path=$COHORT_PATH \
#     --vocab_path=$VOCAB_PATH \
#     --features_row_id_map=$FEATURES_ROW_ID_MAP_PATH \
#     --data_mode='array_alt' \
#     --regularization_metric="group_dro" \
#     --run_evaluation_group \
#     --sensitive_attribute="adult_at_admission" \
#     --eval_attributes="adult_at_admission" \
#     --label_col='early_sepsis' \
#     --num_epochs=100 \
#     --num_hidden=3 \
#     --hidden_dim=128 \
#     --early_stopping \
#     --early_stopping_patience=25 \
#     --drop_prob=0.75 \
#     --lr=0.0001 \
#     --lr_dro=0.01 \
#     --experiment_name="scratch" \
#     --fold_id="1" \
#     --batch_size=512

# python -m prediction_utils.pytorch_utils.train_model \
#     --regularization_metric="group_dro" \
#     --run_evaluation_group \
#     --sensitive_attribute="race_eth" \
#     --eval_attributes="race_eth" \
#     --label_col='LOS_7' \
#     --num_epochs=100 \
#     --early_stopping \
#     --early_stopping_patience=25 \
#     --drop_prob=0.5 \
#     --lr=0.0001 \
#     --lr_dro=0.01 \
#     --experiment_name="scratch"

# python -m prediction_utils.pytorch_utils.train_model \
#     --run_evaluation_group \
#     --eval_attributes="race_eth" \
#     --label_col='LOS_7' \
#     --num_epochs=100 \
#     --early_stopping \
#     --early_stopping_patience=10 \
#     --drop_prob=0.5 \
#     --lr=0.0001 \
#     --experiment_name="scratch"