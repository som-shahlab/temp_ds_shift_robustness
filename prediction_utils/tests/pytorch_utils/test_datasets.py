import numpy as np
import scipy.sparse as sp
import pandas as pd
import random
from prediction_utils.pytorch_utils.datasets import ArrayLoaderGenerator


class TestArrayLoaderGenerator:
    def get_features_sparse(self, num_samples=100, num_features=1000, seed=10):
        return sp.random(m=num_samples, n=num_features, format="csr", random_state=seed)

    def get_cohort(
        self,
        num_samples=100,
        row_id_col="row_id",
        fold_id_test="test",
        label_col="outcome",
        attributes=["gender"],
        attribute="gender",
    ):

        return pd.DataFrame(
            {
                row_id_col: np.arange(num_samples),
                "fold_id": [
                    random.choice(["1", "2", "3", fold_id_test])
                    for _ in range(num_samples)
                ],
                label_col: np.random.randint(0, 1 + 1, size=num_samples),
                attribute: [
                    random.choice(["male", "female"]) for _ in range(num_samples)
                ],
            }
        )

    def test_get_data_dict(self):
        """
        Checks whether row_id and labels are propagated properly.
        Creates a data_dict, extracts the data back out, and compares to the original
        """
        num_samples = 100
        num_features = 1000
        features = self.get_features_sparse(
            num_samples=num_samples, num_features=num_features
        )
        cohort = self.get_cohort(num_samples=num_samples)
        loader_generator = ArrayLoaderGenerator(
            features=features,
            cohort=cohort,
            row_id_col="row_id",
            fold_id_test="test",
            label_col="outcome",
            attributes=["gender"],
            attribute="gender",
            fold_id="1",
            num_workers=0,
        )
        data_dict = loader_generator.data_dict

        data_df_dict = {
            key: pd.concat({key2: pd.Series(value2) for key2, value2 in value.items()})
            .to_frame()
            .rename(columns={0: key})
            .rename_axis(["fold_id", "dict_row_id"])
            .reset_index()
            for key, value in data_dict.items()
            if key in ["row_id", "labels"]
        }

        for i, (key, value) in enumerate(data_df_dict.items()):
            print(value)
            if i == 0:
                data_df = value
            else:
                data_df = data_df.merge(value)

        data_df = data_df.rename(columns={"labels": "outcome"})
        assert data_df.shape[0] == cohort.shape[0]
        data_df = (
            data_df[["row_id", "outcome"]].sort_values("row_id").reset_index(drop=True)
        )
        cohort = (
            cohort[["row_id", "outcome"]].sort_values("row_id").reset_index(drop=True)
        )
        print(data_df)
        print(cohort)
        assert (data_df[["row_id", "outcome"]].sort_values("row_id")).equals(
            cohort[["row_id", "outcome"]].sort_values("row_id")
        )
