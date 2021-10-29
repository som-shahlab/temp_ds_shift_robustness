import pandas as pd
from test_admission_rollup import df_dict_concat
from prediction_utils.cohorts.admissions.bigquery.cohort import BQAdmissionOutcomeCohort
from bigquery_test_util import assert_same_rows


# (TODO - make test more comprehensive to check for edge cases in the base_query)
def get_readmission_30_data():
    input_df = df_dict_concat(
        {
            0: {  # First visit triggers a readmission
                "admit_date": pd.to_datetime(["1/1/2018", "1/28/2018"]),
                "discharge_date": pd.to_datetime(["1/8/2018", "2/10/2018"]),
            },
            1: {  # No readmission
                "admit_date": pd.to_datetime(["1/1/2018", "2/15/2018"]),
                "discharge_date": pd.to_datetime(["1/8/2018", "2/19/2018"]),
            },
            2: {  # Only one visit
                "admit_date": pd.to_datetime(["1/1/2018"]),
                "discharge_date": pd.to_datetime(["2/10/2018"]),
            },
            3: {  # Two readmissions
                "admit_date": pd.to_datetime(["1/1/2018", "1/28/2018", "2/15/2018"]),
                "discharge_date": pd.to_datetime(
                    ["1/8/2018", "2/10/2018", "2/20/2018"]
                ),
            },
            4: {  # Shift by a year - no readmission
                "admit_date": pd.to_datetime(["1/1/2018", "1/28/2019"]),
                "discharge_date": pd.to_datetime(["1/8/2018", "2/10/2019"]),
            },
        }
    )
    true_result_df = df_dict_concat(
        {
            0: {  # First visit triggers a readmission
                "admit_date": pd.to_datetime(["1/1/2018", "1/28/2018"]),
                "discharge_date": pd.to_datetime(["1/8/2018", "2/10/2018"]),
                "readmission_30": [1, 0],
            },
            1: {  # No readmission
                "admit_date": pd.to_datetime(["1/1/2018", "2/15/2018"]),
                "discharge_date": pd.to_datetime(["1/8/2018", "2/19/2018"]),
                "readmission_30": [0, 0],
            },
            2: {  # Only one visit
                "admit_date": pd.to_datetime(["1/1/2018"]),
                "discharge_date": pd.to_datetime(["2/10/2018"]),
                "readmission_30": [0],
            },
            3: {  # Two readmissions
                "admit_date": pd.to_datetime(["1/1/2018", "1/28/2018", "2/15/2018"]),
                "discharge_date": pd.to_datetime(
                    ["1/8/2018", "2/10/2018", "2/20/2018"]
                ),
                "readmission_30": [1, 1, 0],
            },
            4: {  # Shift by a year - no readmission
                "admit_date": pd.to_datetime(["1/1/2018", "1/28/2019"]),
                "discharge_date": pd.to_datetime(["1/8/2018", "2/10/2019"]),
                "readmission_30": [0, 0],
            },
        }
    )
    return input_df, true_result_df


def test_readmission_30():

    input_df, true_result_df = get_readmission_30_data()
    cohort = BQAdmissionOutcomeCohort()

    config_dict = {
        "rs_dataset": "temp_dataset",
        "cohort_name_input": "test_readmission_30_input",
        "cohort_name_result": "test_readmission_30_result",
        "cohort_name_result_true": "test_readmission_30_result_true",
    }

    cohort.db.to_sql(
        df=input_df,
        destination_table="{rs_dataset}.{cohort_name_input}".format_map(config_dict),
        date_cols=["admit_date", "discharge_date"],
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
        {rs_dataset}.{cohort_name_input}
    """.format_map(
        config_dict
    )

    formatted_query = cohort.get_readmission_query().format_map(
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
