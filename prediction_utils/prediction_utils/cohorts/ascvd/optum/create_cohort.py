import configargparse as argparse
import os
from prediction_utils.cohorts.ascvd.cohort import (
    ASCVDCohort,
    ASCVDDemographicsCohort,
)
from prediction_utils.util import patient_split

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset", type=str, default="starr_omop_cdm5_deid_1pcent_lite_latest"
)
parser.add_argument("--rs_dataset", type=str, default="temp_dataset")
parser.add_argument("--limit", type=int, default=0)
parser.add_argument("--gcloud_project", type=str, default="som-nero-phi-nigam-starr")
parser.add_argument("--dataset_project", type=str, default="som-rit-phi-starr-prod")
parser.add_argument(
    "--rs_dataset_project", type=str, default="som-nero-phi-nigam-starr"
)
parser.add_argument("--cohort_name", type=str, default="ascvd_starr_20200425")

parser.add_argument("--horizon", type=str, default="1yr")

parser.add_argument(
    "--has_birth_datetime", dest="has_birth_datetime", action="store_true"
)
parser.add_argument(
    "--no_has_birth_datetime", dest="has_birth_datetime", action="store_false"
)
parser.add_argument(
    "--data_path",
    type=str,
    default="/share/pi/nigam/projects/spfohl/cohorts/admissions/scratch",
)
parser.add_argument(
    "--google_application_credentials",
    type=str,
    default=os.path.expanduser("~/.config/gcloud/application_default_credentials.json"),
)

parser.set_defaults(has_birth_datetime=True)

if __name__ == "__main__":
    args = parser.parse_args()
    label_config = {
        "1yr": {"max_index_date": "2016-12-31", "event_followup_days": 365.25},
        "5yr": {"max_index_date": "2012-12-31", "event_followup_days": 5 * 365.25},
        "10yr": {"max_index_date": "2007-12-31", "event_followup_days": 10 * 365.25},
    }
    config_dict = {**args.__dict__, **label_config[args.horizon]}
    cohort = ASCVDCohort(**config_dict)
    destination_table = "{rs_dataset_project}.{rs_dataset}.{cohort_name}".format_map(
        cohort.config_dict
    )
    print(destination_table)
    cohort.db.execute_sql_to_destination_table(
        cohort.get_transform_query_sampled(), destination=destination_table
    )
    cohort = ASCVDDemographicsCohort(**config_dict)

    cohort.db.execute_sql_to_destination_table(
        cohort.get_transform_query(), destination=destination_table
    )
    cohort_df = cohort.db.read_sql_query(
        """
        SELECT * FROM {destination_table}
    """.format(
            destination_table=destination_table
        )
    )
    cohort_df = patient_split(
        cohort_df, patient_col="person_id", test_frac=0.1, nfold=10, seed=657
    )
    cohort_path = os.path.join(args.data_path, "cohort")
    os.makedirs(cohort_path, exist_ok=True)
    cohort_df.to_parquet(
        os.path.join(cohort_path, "cohort_{}.parquet".format(args.cohort_name)),
        engine="pyarrow",
        index=False,
    )
