import torch
import numpy as np
import pandas as pd
import sklearn
import logging
from prediction_utils.pytorch_utils.models import TorchModel, FixedWidthModel
from prediction_utils.pytorch_utils.datasets import (
    ArrayLoaderGenerator,
    DataDictConstructor,
    ArrayDataset,
)
from prediction_utils.pytorch_utils.metric_logging import (
    MetricLogger,
    OutputDict,
    LossDict,
)


class DiscreteTimeModel(TorchModel):
    """
    A model that uses the discrete time loss
    """

    def __init__(self, *args, transformer=None, **kwargs):
        """
        Args:
            transformer: is an object returned by create_bin_transformer
        """
        super().__init__(*args, **kwargs)
        self.transformer = transformer

    def forward_on_batch(self, the_data):
        outputs = self.model(the_data["features"])
        loss_dict_batch = {
            "loss": self.criterion(
                outputs, the_data["labels"], the_data["event_indicator"]
            )
        }
        return loss_dict_batch, outputs

    def init_loss(self):
        """
        Returns the loss function
        """
        return self.discrete_time_loss

    def get_logging_keys(self):
        result = ["pred_probs", "row_id"]
        if self.config_dict.get("weighted_evaluation"):
            result = result + ["weights"]
        if self.config_dict.get("logging_evaluate_by_group"):
            result = result + ["group"]
        return result

    def discrete_time_loss(self, outputs, labels, event_indicator):

        # The uncensored loss
        outputs_uncensored = outputs[event_indicator == 1]
        labels_uncensored = labels[event_indicator == 1]
        label_matrix_uncensored = torch.nn.functional.one_hot(
            labels_uncensored, num_classes=self.config_dict["output_dim"]
        )
        bce_matrix_uncensored = torch.nn.functional.binary_cross_entropy_with_logits(
            outputs_uncensored, label_matrix_uncensored.float(), reduction="none"
        )
        mask_uncensored = (
            torch.arange(self.config_dict["output_dim"]).unsqueeze(0).to(self.device)
            <= labels_uncensored.unsqueeze(1)
        ).to(dtype=torch.int64, device=self.device)
        assert bce_matrix_uncensored.shape == mask_uncensored.shape
        masked_loss_matrix_uncensored = bce_matrix_uncensored * mask_uncensored

        # The censored loss
        labels_censored = labels[event_indicator == 0]
        outputs_censored = outputs[event_indicator == 0]
        label_matrix_censored = torch.zeros(
            outputs_censored.shape[0],
            self.config_dict["output_dim"],
            device=self.device,
            dtype=torch.float,
        )

        bce_matrix_censored = torch.nn.functional.binary_cross_entropy_with_logits(
            outputs_censored, label_matrix_censored, reduction="none"
        )
        mask_censored = (
            torch.arange(self.config_dict["output_dim"])
            .unsqueeze(0)
            .to(device=self.device)
            <= labels_censored.unsqueeze(1)
        ).to(dtype=torch.int64, device=self.device)
        assert bce_matrix_censored.shape == mask_censored.shape
        masked_loss_matrix_censored = bce_matrix_censored * mask_censored
        loss = torch.cat(
            (
                masked_loss_matrix_uncensored.sum(dim=1),
                masked_loss_matrix_censored.sum(dim=1),
            ),
            dim=0,
        ).mean()
        return loss

    def predict_survival_function(self, inputs):
        outputs = self.model(inputs)
        return torch.exp(torch.cumsum(torch.nn.functional.logsigmoid(-outputs), dim=1))

    def predict_survival_function_at_times(self, inputs, times):
        survival_function = self.predict_survival_function(inputs)
        if not hasattr(times, "__iter__"):
            # If a constant if passed
            times = [times]
        mapped_times = torch.tensor(
            self.transformer.transform(times).codes,
            dtype=torch.long,
            device=self.device,
        )
        result = torch.gather(
            survival_function, 1, mapped_times.reshape(-1, 1)
        ).squeeze()
        return result

    def predict(self, loaders, time_horizon=None, phases=["test"], return_outputs=True):
        """
        Method that trains the model.
            Args:
                loaders: A dictionary of DataLoaders with keys corresponding to phases
                kwargs: Additional arguments to override in the config_dict
            Returns:
                result_dict: A dictionary with metrics recorded every epoch

        """

        if self.transformer is None:
            raise ValueError("Cannot predict if transformer not defined")

        metric_logger = PredictedProbabilityMetricLogger(
            phases=phases,
            metrics=self.config_dict.get("logging_metrics"),
            threshold_metrics=self.config_dict.get("logging_threshold_metrics"),
            losses=self.get_loss_names(),
            output_dict_keys=self.get_logging_keys(),
            weighted_evaluation=self.config_dict.get("weighted_evaluation"),
            evaluate_by_group=self.config_dict.get("logging_evaluate_by_group"),
            disable_metric_logging=self.config_dict.get("disable_metric_logging"),
            compute_group_min_max=self.config_dict.get("compute_group_min_max"),
        )

        self.model.train(False)
        output_dict = {}
        for phase in phases:
            logging.info("Evaluating on phase: {phase}".format(phase=phase))
            metric_logger.init_metric_dicts()
            for i, the_data in enumerate(loaders[phase]):
                the_data = self.transform_batch(
                    the_data, keys=self.get_transform_batch_keys()
                )
                times = the_data["event_times"]
                if time_horizon is not None:
                    times = np.minimum(times.numpy(), time_horizon)
                outputs = self.predict_survival_function_at_times(
                    the_data["features"], times
                )
                metric_logger.update_output_dict(pred_probs=outputs, **the_data)

            if return_outputs:
                output_dict[phase] = metric_logger.get_output_df()

        result_dict = {}
        if return_outputs:
            result_dict["outputs"] = (
                pd.concat(output_dict)
                .reset_index(level=-1, drop=True)
                .rename_axis("phase")
                .reset_index()
            )
        return result_dict


class DiscreteTimeNNet(FixedWidthModel, DiscreteTimeModel):
    """
    A discrete time neural network model
    """

    pass


class PredictedProbabilityOutputDict(OutputDict):
    """
        Accumulates outputs over an epoch
    """

    def __init__(self, keys=None):
        self.init_output_dict(keys=keys)

    def init_output_dict(self, keys=None):
        if keys is None:
            keys = ["pred_probs", "row_id"]

        if "pred_probs" not in keys:
            keys.append("pred_probs")

        self.output_dict = {key: [] for key in keys}

    def update_output_dict(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.output_dict.keys():
                self.output_dict[key].append(value.detach().cpu())


class PredictedProbabilityMetricLogger(MetricLogger):
    def init_metric_dicts(self):
        self.loss_dict = LossDict(metrics=self.losses)
        self.output_dict = PredictedProbabilityOutputDict(keys=self.output_dict_keys)

    def update_output_dict(self, *args, **kwargs):
        self.output_dict.update_output_dict(*args, **kwargs)

    def get_output_df(self):
        self.output_dict.finalize_output_dict()
        return pd.DataFrame(
            {
                key: value[:, -1] if key == "outputs" else value
                for key, value in self.output_dict.output_dict.items()
            }
        )


def create_bin_transformer(bin_times=None, num_quantiles=None):
    """
    Creates a scikit-learn transformer that maps times to bins
    Args
        bin_times: a sequence of times
        num_quantiles: The number of quantiles to use. If None, will use all unique bin_times as cut points
    Returns
        a transformer object that transforms bin_times to a pd.Categorical
    """
    if bin_times is None:
        raise ValueError("bin_times must not be None")

    if num_quantiles is not None:
        _, bins = pd.qcut(bin_times, num_quantiles, retbins=True, duplicates="drop")
    else:
        bins = np.sort(bin_times.unique())
    assert (bins < 0).sum() >= 0
    if 0 != bins[0]:
        bins = np.append(np.array(0), bins)
    if np.inf != bins[-1]:
        bins = np.append(bins, np.array(np.inf))

    transformer = sklearn.preprocessing.FunctionTransformer(
        pd.cut,
        kw_args={
            "bins": bins,
            "retbins": False,
            "right": False,
            "include_lowest": True,
        },
    )
    return transformer


class DiscreteTimeArrayLoaderGenerator(ArrayLoaderGenerator):
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
        self.config_dict = self.get_default_config()
        self.config_dict = self.override_config(**kwargs)
        # super().__init__(*args, **kwargs)

        self.dataset_dict = dataset_dict
        self.data_dict = data_dict

        if self.dataset_dict is None and self.data_dict is None:
            data_dict_constructor = DiscreteTimeDataDictConstructor(*args, **kwargs)
            self.data_dict = data_dict_constructor.get_data_dict(
                features=features, cohort=cohort
            )
            self.config_dict = self.override_config(**data_dict_constructor.config_dict)

        if self.dataset_dict is None:
            self.dataset_dict = self.init_datasets()

    def init_datasets(self):
        """
        Creates data loaders from inputs
        """
        phases = self.data_dict["row_id"].keys()
        tensor_dict_dict = {
            key: {
                "features": self.data_dict["features"][key],
                "event_times": self.data_dict["event_times"][key],
                "labels": torch.tensor(self.data_dict["labels"][key], dtype=torch.long),
                "event_indicator": torch.tensor(
                    self.data_dict["event_indicator"][key], dtype=torch.long
                ),
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
                    np.copy(self.data_dict["weights"][key]), dtype=torch.float
                )

        dataset_dict = {
            key: ArrayDataset(
                tensor_dict=tensor_dict_dict[key],
                sparse_mode=self.config_dict.get("sparse_mode"),
            )
            for key in phases
        }

        return dataset_dict


class DiscreteTimeDataDictConstructor(DataDictConstructor):
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
            "event_indicator_var_name": None,
            "num_bins": None,
        }

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

        if self.config_dict.get("num_bins") is None:
            raise ValueError("num_bins must be specified")

        self.config_dict["bin_transformer"] = create_bin_transformer(
            cohort_dict["train"].query(
                "{} == 1".format(self.config_dict["event_indicator_var_name"])
            )[self.config_dict["label_col"]],
            num_quantiles=self.config_dict.get("num_bins"),
        )

        data_dict["event_times"] = {
            key: value[self.config_dict["label_col"]].values
            for key, value in cohort_dict.items()
        }

        data_dict["labels"] = {
            key: self.config_dict["bin_transformer"]
            .transform(value[self.config_dict["label_col"]].values)
            .codes.astype(np.int64)
            for key, value in cohort_dict.items()
        }

        data_dict["event_indicator"] = {
            key: (value[self.config_dict["event_indicator_var_name"]]).values.astype(
                np.int64
            )
            for key, value in cohort_dict.items()
        }

        return data_dict
