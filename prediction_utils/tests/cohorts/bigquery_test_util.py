import numpy as np
import pandas as pd
import pytest


def assert_same_rows(db, table1, table2):
    """
    Utility function to assess whether table1 and table2 have the same content
    Queries the distinct rows in each table and returns false if the result is not empty.
        Args:
            db: An instance of prediction_utils.extraction_utils.database.BQDatabase
            table1: {dataset}.{table}
            table2: {dataset}.{table}
    """
    test_query = """
        (
            SELECT *, "{table1}" as source_table
            FROM
            (
              SELECT *
              FROM {table1}
              EXCEPT DISTINCT
              SELECT * from {table2}
            )
        )
        UNION ALL
        (
            SELECT *, "{table2}" as source_table 
            FROM
            (
              SELECT *
              FROM {table2} 
              EXCEPT DISTINCT
              SELECT * from {table1}
            )
        )
    """.format(
        table1=table1, table2=table2
    )
    test_df = db.read_sql_query(test_query, use_bqstorage_api=False)

    assert test_df.shape[0] == 0
