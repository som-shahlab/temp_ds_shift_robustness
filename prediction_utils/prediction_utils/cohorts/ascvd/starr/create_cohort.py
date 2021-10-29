import configargparse as argparse
import os
from prediction_utils.cohorts.ascvd.cohort import (
    ASCVDCohort,
    ASCVDDemographicsCohort,
    get_transform_query_sampled,
)
from prediction_utils.util import patient_split_cv

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="starr_omop_cdm5_deid_20210130")
parser.add_argument("--rs_dataset", type=str, default="spfohl_explore")
parser.add_argument("--limit", type=int, default=0)
parser.add_argument("--gcloud_project", type=str, default="som-nero-nigam-starr")
parser.add_argument("--dataset_project", type=str, default="som-nero-nigam-starr")
parser.add_argument("--rs_dataset_project", type=str, default="som-nero-nigam-starr")
parser.add_argument(
    "--cohort_name", type=str, default="ascvd_10yr_starr_omop_cdm5_deid_20210130"
)
parser.add_argument(
    "--cohort_name_sampled",
    type=str,
    default="ascvd_10yr_starr_omop_cdm5_deid_20210130_sampled",
)
parser.add_argument("--horizon", type=str, default="10yr")

parser.add_argument(
    "--has_birth_datetime", dest="has_birth_datetime", action="store_true"
)
parser.add_argument(
    "--no_has_birth_datetime", dest="has_birth_datetime", action="store_false"
)
parser.add_argument(
    "--data_path",
    type=str,
    default="/local-scratch/nigam/projects/spfohl/nhlbi_aim_1/cohorts/ascvd/scratch",
)

parser.add_argument(
    "--google_application_credentials",
    type=str,
    default=os.path.expanduser("~/.config/gcloud/application_default_credentials.json"),
)

parser.set_defaults(has_birth_datetime=True)

label_config = {
    "1yr": {"max_index_date": "2019-12-31", "event_followup_days": 365.25},
    "5yr": {"max_index_date": "2015-12-31", "event_followup_days": 5 * 365.25},
    "10yr": {"max_index_date": "2010-12-31", "event_followup_days": 10 * 365.25},
}

if __name__ == "__main__":
    args = parser.parse_args()

    config_dict = {**args.__dict__, **label_config[args.horizon]}

    cohort = ASCVDCohort(**config_dict)
    destination_table = "{rs_dataset_project}.{rs_dataset}.{cohort_name}".format_map(
        config_dict
    )
    print(destination_table)
    cohort.db.execute_sql_to_destination_table(
        cohort.get_transform_query(), destination=destination_table
    )
    cohort_demographics = ASCVDDemographicsCohort(**config_dict)
    cohort_demographics.db.execute_sql_to_destination_table(
        cohort_demographics.get_transform_query(), destination=destination_table
    )

    destination_table_sampled = "{rs_dataset_project}.{rs_dataset}.{cohort_name_sampled}".format_map(
        config_dict
    )
    cohort.db.execute_sql_to_destination_table(
        get_transform_query_sampled(source_table=destination_table),
        destination=destination_table_sampled,
    )

    cohort_df = cohort.db.read_sql_query(
        """
        SELECT * FROM {destination_table}
    """.format(
            destination_table=destination_table_sampled
        )
    )

    cohort_df = patient_split_cv(
        cohort_df, patient_col="person_id", test_frac=0.1, nfold=10, seed=657
    )
    cohort_path = os.path.join(args.data_path, "cohort")
    os.makedirs(cohort_path, exist_ok=True)
    cohort_df.to_parquet(
        os.path.join(cohort_path, "cohort.parquet"),
        engine="pyarrow",
        index=False,
    )
