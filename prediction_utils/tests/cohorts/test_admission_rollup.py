import pandas as pd
from prediction_utils.cohorts.admissions.postgres.create_cohort import get_admissions
from prediction_utils.cohorts.admissions.bigquery.cohort import BQAdmissionRollupCohort

# from prediction_utils.cohorts.admissions.admission_rollup_bq import (
# admission_rollup_query,
# )
from prediction_utils.extraction_utils.database import BQDatabase
from bigquery_test_util import assert_same_rows


def df_dict_concat(the_dict):
    return (
        pd.concat(
            {key: pd.DataFrame(value) for key, value in the_dict.items()},
            join="outer",
            names=["person_id", "dummy"],
        )
        .reset_index("dummy", drop=True)
        .reset_index()
    )


def get_admission_rollup_data():
    """
    Returns the input and expected result for admission_rollup
    """
    input_df = df_dict_concat(
        {
            0: {  # Two visits - nothing happens
                "visit_start_date": pd.to_datetime(["1/1/2018", "1/3/2018"]),
                "visit_end_date": pd.to_datetime(["1/2/2018", "1/6/2018"]),
            },
            1: {  # Matching end
                "visit_start_date": pd.to_datetime(
                    ["1/1/2018", "1/3/2018", "1/10/2018"]
                ),
                "visit_end_date": pd.to_datetime(["1/3/2018", "1/6/2018", "1/10/2018"]),
            },
            2: {  # One large visit
                "visit_start_date": pd.to_datetime(
                    ["1/1/2018", "1/3/2018", "1/1/2018"]
                ),
                "visit_end_date": pd.to_datetime(["1/2/2018", "1/6/2018", "1/10/2018"]),
            },
            3: {  # Test a visit with start > end
                "visit_start_date": pd.to_datetime(["1/1/2018", "1/7/2018"]),
                "visit_end_date": pd.to_datetime(["1/2/2018", "1/6/2018"]),
            },
            4: {  # A single visit
                "visit_start_date": pd.to_datetime(["1/1/2018"]),
                "visit_end_date": pd.to_datetime(["1/2/2018"]),
            },
            5: {  # A single visit - that drops out
                "visit_start_date": pd.to_datetime(["1/1/2018"]),
                "visit_end_date": pd.to_datetime(["1/1/2018"]),
            },
        }
    )

    true_result_df = df_dict_concat(
        {
            0: {  # Two visits - nothing happens
                "admit_date": pd.to_datetime(["1/1/2018", "1/3/2018"]),
                "discharge_date": pd.to_datetime(["1/2/2018", "1/6/2018"]),
            },
            1: {  # Matching end
                "admit_date": pd.to_datetime(["1/1/2018"]),
                "discharge_date": pd.to_datetime(["1/6/2018"]),
            },
            2: {  # One large visit
                "admit_date": pd.to_datetime(["1/1/2018"]),
                "discharge_date": pd.to_datetime(["1/10/2018"]),
            },
            3: {  # Test a visit with start > end
                "admit_date": pd.to_datetime(["1/1/2018"]),
                "discharge_date": pd.to_datetime(["1/2/2018"]),
            },
            4: {  # A single visit
                "admit_date": pd.to_datetime(["1/1/2018"]),
                "discharge_date": pd.to_datetime(["1/2/2018"]),
            },
        }
    )
    return input_df, true_result_df


def test_rollup():
    """
    Tests the python implementation of admission rollup
    """
    input_df, true_result_df = get_admission_rollup_data()
    result_df = get_admissions(input_df)
    pd.testing.assert_frame_equal(result_df, true_result_df)
    assert result_df.equals(true_result_df)


def test_admission_rollup_sql():
    """
    Tests the BigQuery Standard SQL implementation of admission rollup
    """
    input_df, true_result_df = get_admission_rollup_data()
    # db = BQDatabase()
    cohort = BQAdmissionRollupCohort()
    config_dict = {
        "rs_dataset": "temp_dataset",
        "cohort_name_input": "test_admission_rollup_input",
        "cohort_name_result": "test_admission_rollup_result",
        "cohort_name_result_true": "test_admission_rollup_result_true",
    }

    cohort.db.to_sql(
        df=input_df,
        destination_table="{rs_dataset}.{cohort_name_input}".format_map(config_dict),
        date_cols=["visit_start_date", "visit_end_date"],
        mode="client",
    )

    cohort.db.to_sql(
        df=true_result_df,
        destination_table="{rs_dataset}.{cohort_name_result_true}".format_map(
            config_dict
        ),
        date_cols=["admit_date", "discharge_date"],
        mode="client",
    )

    base_query = """
    (
        SELECT * 
        FROM
        {rs_dataset}.{cohort_name_input}
        where visit_end_date > visit_start_date
    )
    """
    # formatted_query = admission_rollup_query.format_map(base_query.format_map(config_dict))
    formatted_query = cohort.get_transform_query(format_query=False).format_map(
        {**{"base_query": base_query.format_map(config_dict)}, **config_dict}
    )

    create_query = """
        CREATE OR REPLACE TABLE {rs_dataset}.{cohort_name_result} AS
        {query}
        """.format_map(
        {**config_dict, **{"query": formatted_query}}
    )

    cohort.db.execute_sql(create_query)

    assert_same_rows(
        db=cohort.db,
        table1="{rs_dataset}.{cohort_name_result}".format_map(config_dict),
        table2="{rs_dataset}.{cohort_name_result_true}".format_map(config_dict),
    )
