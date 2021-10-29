import pandas as pd
import torch
import scipy as sp
import numpy as np
import warnings
from torch.utils.data import (
    Dataset,
    DataLoader,
    RandomSampler,
    BatchSampler,
    WeightedRandomSampler,
)
from torch.utils.data.dataloader import default_collate


class LoaderGenerator:
    """
    A class that constructs data loaders
    """

    def __init__(self, *args, **kwargs):
        self.config_dict = self.get_default_config()
        self.config_dict = self.override_config(**kwargs)

    def init_loaders(self):
        """
        Returns a dictionary of dataloaders with keys indicating phases
        """
        raise NotImplementedError

    def get_default_config(self):
        """
        Defines the default config_dict
        """
        raise NotImplementedError

    def override_config(self):
        """
        Overrides the config dict with provided kwargs
        """
        raise NotImplementedError


class ArrayLoaderGenerator(LoaderGenerator):
    def __init__(
        self,
        *args,
        dataset_dict=None,
        data_dict=None,
        cohort=None,
        features=None,
        **kwargs
    ):
        """
        A class that can be used to generate data loaders
        Arguments:
            *args: positional arguments (unused)
            data_dict: A dictionary of {phase: {'features': ...}, {'labels': ...}} that is used to construct a dataset_dict
                if None, is constructed by DataDictConstructor
            dataset_dict: A dictionary of {phase: {torch.utils.data.Dataset}},
                if not None, DataDictConstructor is not used
            features: a feature matrix (np.array or scipy.sparse.csr) where each row corresponds to a data instance
            cohort: a pandas dataframe containing metadata
            **kwargs: Additional arguments that get stored in the config dict and passed to DataDictConstructor
        """
        super().__init__(*args, **kwargs)

        self.dataset_dict = dataset_dict
        self.data_dict = data_dict

        if self.dataset_dict is None and self.data_dict is None:
            data_dict_constructor = DataDictConstructor(*args, **kwargs)
            self.data_dict = data_dict_constructor.get_data_dict(
                features=features, cohort=cohort
            )
            self.config_dict = self.override_config(**data_dict_constructor.config_dict)

        if self.dataset_dict is None:
            self.dataset_dict = self.init_datasets()

    def get_default_config(self):
        return {
            "batch_size": 256,
            "iters_per_epoch": 100,
            "include_group_in_dataset": False,
            "group_var_name": None,
            "weight_var_name": None,
            "sparse_mode": None,
            "num_workers": 0,
        }

    def override_config(self, **override_dict):
        return {**self.config_dict, **override_dict}

    def init_datasets(self):
        """
        Creates data loaders from inputs
        """
        phases = self.data_dict["row_id"].keys()
        tensor_dict_dict = {
            key: {
                "features": self.data_dict["features"][key],
                "labels": torch.tensor(self.data_dict["labels"][key], dtype=torch.long),
                "row_id": torch.tensor(self.data_dict["row_id"][key], dtype=torch.long),
            }
            for key in phases
        }
        if self.config_dict.get("include_group_in_dataset"):
            for key in phases:
                tensor_dict_dict[key]["group"] = torch.as_tensor(
                    np.copy(self.data_dict["group"][key]), dtype=torch.long
                )
        if self.config_dict.get("weight_var_name") is not None:
            for key in phases:
                tensor_dict_dict[key]["weights"] = torch.as_tensor(
                    np.copy(self.data_dict["weights"][key]),
                    dtype=torch.float
                )

        dataset_dict = {
            key: ArrayDataset(
                tensor_dict=tensor_dict_dict[key],
                sparse_mode=self.config_dict.get("sparse_mode"),
            )
            for key in phases
        }

        return dataset_dict

    def init_loaders(self, sample_keys=None):
        """
        Method that converts data and labels to instances of class torch.utils.data.DataLoader
            Returns:
                a dictionary with the same keys as data_dict and label_dict.
                    Each element of the dictionary is an instance of torch.utils.data.DataLoader
                        that yields paired elements of data and labels
        """
        # Convert the data to Dataset
        dataset_dict = self.dataset_dict

        # If the Dataset implements collate_fn, that is used. Otherwise, default_collate is used
        if hasattr(dataset_dict["train"], "collate_fn") and callable(
            getattr(dataset_dict["train"], "collate_fn")
        ):
            collate_fn = dataset_dict["train"].collate_fn
        else:
            collate_fn = default_collate

        if self.config_dict.get("iters_per_epoch") is not None:
            num_samples = (
                self.config_dict["iters_per_epoch"] * self.config_dict["batch_size"]
            )

        if sample_keys is None:
            sample_keys = ["train"]

        loaders_dict = {}
        for key in dataset_dict.keys():
            if key in sample_keys:
                if self.config_dict.get("balance_groups"):
                    random_sampler = WeightedRandomSampler(
                        weights=self.data_dict["group_weight"][key],
                        replacement=True,
                        num_samples=num_samples,
                    )
                else:
                    random_sampler = RandomSampler(
                        dataset_dict[key], replacement=True, num_samples=num_samples
                    )

                loaders_dict[key] = DataLoader(
                    dataset_dict[key],
                    batch_sampler=BatchSampler(
                        random_sampler,
                        batch_size=self.config_dict["batch_size"],
                        drop_last=False,
                    ),
                    collate_fn=collate_fn,
                    num_workers=self.config_dict["num_workers"],
                    pin_memory=False
                    if self.config_dict.get("sparse_mode") == "convert"
                    else True,
                )
            else:
                loaders_dict[key] = DataLoader(
                    dataset_dict[key],
                    batch_size=self.config_dict["batch_size"],
                    collate_fn=collate_fn,
                    num_workers=self.config_dict["num_workers"],
                    pin_memory=False
                    if self.config_dict.get("sparse_mode") == "convert"
                    else True,
                )

        return loaders_dict

    def init_loaders_predict(self, *args):
        """
        Creates data loaders from inputs - for use at prediction time
        """

        # Convert the data to Dataset
        dataset_dict = self.dataset_dict

        # If the Dataset implements collate_fn, that is used. Otherwise, default_collate is used
        if hasattr(dataset_dict["train"], "collate_fn") and callable(
            getattr(dataset_dict["train"], "collate_fn")
        ):
            collate_fn = dataset_dict["train"].collate_fn
        else:
            collate_fn = default_collate

        loaders_dict = {
            key: DataLoader(
                dataset_dict[key],
                batch_size=self.config_dict["batch_size"],
                collate_fn=collate_fn,
                num_workers=self.config_dict["num_workers"],
                pin_memory=False
                if self.config_dict.get("sparse_mode") == "convert"
                else True,
            )
            for key in dataset_dict.keys()
        }

        return loaders_dict


class DataDictConstructor:
    def __init__(self, *args, **kwargs):
        self.config_dict = self.get_default_config()
        self.config_dict = self.override_config(**kwargs)

    def get_default_config(self):
        """
        Defines the default config_dict
        """
        return {
            "fold_id_test": ["test", "eval"],
            "fold_id": None,  # the fold id corresponding to the early stopping dev/validation set
            "train_key": "train",
            "eval_key": "val",
            "group_var_name": None,
            "weight_var_name": None,
            "row_id_col": "row_id",
            "label_col": "outcome",
            "balance_groups": False,
        }

    def override_config(self, **override_dict):
        return {**self.config_dict, **override_dict}

    def append_phase_column(self, cohort):

        fold_id = (
            ""
            if self.config_dict["fold_id"] is None
            else str(self.config_dict["fold_id"])
        )
        fold_id_test = self.config_dict["fold_id_test"]

        if isinstance(fold_id_test, str):
            fold_id_test = [fold_id_test]

        cohort = cohort.assign(
            phase=lambda x: x["fold_id"].where(
                lambda y: y.apply(lambda z: z in (fold_id_test + [fold_id])),
                self.config_dict["train_key"],
            )
        ).assign(
            phase=lambda x: x.phase.where(
                lambda y: y != fold_id, self.config_dict["eval_key"]
            )
        )
        return cohort

    def compute_group_weights(self, cohort):
        group_var_name = self.config_dict["group_var_name"]
        if group_var_name is None:
            raise ValueError("Cannot compute group weights if group_var_name is None")
        if "group_weight" in cohort.columns:
            warnings.warn("group_weight is already a column in cohort")

        group_weight_df = (
            cohort.groupby(group_var_name)
            .size()
            .rename("group_weight")
            .to_frame()
            .reset_index()
            .assign(group_weight=lambda x: 1 / (x.group_weight / cohort.shape[0]))
        )
        return group_weight_df

    def create_group_variable(self, data_dict, cohort_dict, categories):
        data_dict["group"] = {
            key: pd.Categorical(
                value[self.config_dict["group_var_name"]], categories=categories
            ).codes
            for key, value in cohort_dict.items()
        }
        self.config_dict["num_groups"] = len(categories)
        self.config_dict["group_mapper"] = (
            pd.Series(categories)
            .rename(self.config_dict["group_var_name"])
            .rename_axis("group_id")
            .reset_index()
        )

        if self.config_dict["balance_groups"]:
            data_dict["group_weight"] = {
                key: np.int64(value["group_weight"].values)
                for key, value in cohort_dict.items()
            }
        return data_dict

    def get_data_dict(self, features=None, cohort=None, append_phase_column=True):
        """
        Generates a data_dict from a features array and a cohort dataframe.
        Args:
            features: The input feature matrix
            cohort: A dataframe with a column called "phase" that maps to the phases
        """

        if append_phase_column:
            cohort = self.append_phase_column(cohort)

        if (
            self.config_dict["balance_groups"]
            and self.config_dict["group_var_name"] is not None
        ):
            group_weight_df = self.compute_group_weights(
                cohort.query('phase == "train"')
            )
            cohort = cohort.merge(group_weight_df)

        cohort_dict = {
            key: cohort.query("phase == @key") for key in cohort.phase.unique()
        }
        # Ensure that each partition is sorted and not empty
        cohort_dict = {
            key: value.sort_values(self.config_dict["row_id_col"])
            for key, value in cohort_dict.items()
            if value.shape[0] > 0
        }

        # # Initialize the data_dict
        data_dict = {}
        # Save the row_id corresponding to unique predictions
        data_dict["row_id"] = {
            key: value[self.config_dict["row_id_col"]].values
            for key, value in cohort_dict.items()
        }

        # store the group_var_name
        if self.config_dict["group_var_name"] is not None:
            categories = (
                cohort[self.config_dict["group_var_name"]].sort_values().unique()
            )
            print("Parsed group variable with categories: {}".format(categories))
            data_dict = self.create_group_variable(
                data_dict=data_dict, cohort_dict=cohort_dict, categories=categories
            )

        if self.config_dict["weight_var_name"] is not None:
            data_dict["weights"] = {
                key: value[self.config_dict["weight_var_name"]].values.astype(
                    np.float32
                )
                for key, value in cohort_dict.items()
            }

        # If features should be loaded
        if features is not None:
            data_dict["features"] = {}
            for key in cohort_dict.keys():
                data_dict["features"][key] = features[data_dict["row_id"][key], :]

        data_dict["labels"] = {
            key: (value[self.config_dict["label_col"]]).values.astype(np.int64)
            for key, value in cohort_dict.items()
        }

        return data_dict


class ArrayDataset(Dataset):
    """Dataset wrapping arrays (tensor, numpy, or scipy CSR sparse)

    Each sample will be retrieved by indexing arrays along the first dimension.

    Arguments:
        tensor_dict: a dictionary of array inputs that have the same size in the first dimension
        convert_sparse: whether CSR inputs should be converted to torch.SparseTensor
    """

    def __init__(self, tensor_dict, sparse_mode=None):

        self.sparse_mode = sparse_mode
        if sparse_mode == "dict":
            tensor_dict["features"] = self.csr_to_tensor_dict(tensor_dict["features"])
        elif sparse_mode == "list":
            tensor_dict["features"] = self.csr_to_tensor_list(tensor_dict["features"])

        self.the_len = self.get_len_or_shape(list(tensor_dict.values())[0])
        assert all(
            self.the_len == self.get_len_or_shape(tensor)
            for tensor in tensor_dict.values()
        )
        self.tensor_dict = tensor_dict

    def __getitem__(self, index):
        return {
            key: tensor[index]
            if not isinstance(tensor, dict)
            else {
                tensor_key: tensor_value[index]
                for tensor_key, tensor_value in tensor.items()
            }
            for key, tensor in self.tensor_dict.items()
        }

    def __len__(self):
        return self.the_len

    def get_len_or_shape(self, x):
        if hasattr(x, "shape"):
            return x.shape[0]
        elif isinstance(x, dict):
            return self.get_len_or_shape(list(x.values())[0])
        else:
            return len(x)

    def collate_fn(self, batch):
        """
        Called by Dataloader to aggregate elements into a batch.
        Delegates to collate_helper for typed aggregation
        Arguments:
            batch: a list of dictionaries with same keys as self.tensor_dict
        """
        result = {}
        keys = batch[0].keys()
        for key in keys:
            result[key] = self.collate_helper(
                tuple(element[key] for element in batch), key=key
            )
        return result

    def collate_helper(self, batch, key=None):
        """
        Aggregates a tuple of elements of the same type
        """
        elem = batch[0]
        if isinstance(elem, sp.sparse.csr_matrix):
            return self.csr_collate(batch)
        elif (self.sparse_mode == "dict") and (isinstance(elem, dict)):
            return self.csr_dict_collate(batch)
        elif self.sparse_mode == "list" and key == "features":
            return self.csr_list_collate(batch)
        else:
            return default_collate(batch)

    def csr_dict_collate(self, batch):
        keys = batch[0].keys()
        tensor_list_dict = {
            key: tuple(element[key] for element in batch) for key in keys
        }
        return tensor_list_dict

    def csr_list_collate(self, batch):
        return batch

    def csr_collate(self, batch):
        batch_concat = sp.sparse.vstack(batch)
        if self.sparse_mode == "csr":
            return batch_concat
        elif self.sparse_mode == "convert":
            return self.csr_to_sparse_tensor(batch_concat)
        else:
            raise ValueError("sparse_mode not defined for csr matrix inputs")

    def csr_to_sparse_tensor(self, x):
        """
        Converts CSR matrix to torch.sparse.Tensor
        """
        x = x.tocoo()
        return torch.sparse.FloatTensor(
            torch.LongTensor([x.row, x.col]),
            torch.FloatTensor(x.data),
            torch.Size(x.shape),
        )

    def csr_to_tensor_list(self, x):
        """
            Converts CSR matrix to a list of tensors
            The length of the returned list is x.shape[0]
            The shape of the tensor at position i in the returned list is the number of non-zero elements in x[i, :]
            The first row of the returned tensor correspond to column indices in x[i, :]
            This class currently only supports "bag of words" inputs with no values associated with the indices
        """
        tensor_list = []
        for i in range(len(x.indptr) - 1):
            tensor_list.append(
                torch.tensor(
                    x.indices[(x.indptr[i]) : (x.indptr[i + 1])], dtype=torch.long
                )
            )
        return tensor_list

    def csr_to_tensor_dict(self, x):
        result = {"col_id": [], "data": []}
        for i in range(len(x.indptr) - 1):
            result["col_id"].append(
                torch.tensor(
                    x.indices[(x.indptr[i]) : (x.indptr[i + 1])], dtype=torch.long
                )
            )
            result["data"].append(
                torch.tensor(
                    x.data[(x.indptr[i]) : (x.indptr[i + 1])], dtype=torch.long
                )
            )
        return result
