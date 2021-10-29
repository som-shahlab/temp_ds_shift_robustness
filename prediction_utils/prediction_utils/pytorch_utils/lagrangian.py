import torch
import logging

from prediction_utils.pytorch_utils.models import TorchModel
from prediction_utils.pytorch_utils.layers import FeedforwardNet
from prediction_utils.pytorch_utils.pytorch_metrics import (
    roc_auc_score_surrogate,
    tpr_surrogate,
    fpr_surrogate,
    positive_rate_surrogate,
    precision_surrogate,
    MetricUndefinedError,
    logistic_surrogate,
    hinge_surrogate,
    indicator,
    weighted_cross_entropy_loss,
)

# Structure:
# group_lagrangian_model -> function returning a model class based on a provided key
# LagrangianModel -> a class that implements the "practical" proxy-Lagrangian
#   LagrangianAUCModel
#   LagrangianThresholdRateModel
#   LagrangianTPRModel
#   LagrangianFPRModel
#   LagrangianPositiveRateModel
#   LagrangianPrecisionModel
#   MultiLagrangianThresholdRateModel


def group_lagrangian_model(model_type="loss"):
    """
    A function that returns an instance of GroupRegularizedModel
    """
    class_dict = {
        "loss": LagrangianModel,
        "auc": LagrangianAUCModel,
        "tpr": LagrangianTPRModel,
        "fpr": LagrangianFPRModel,
        "positive_rate": LagrangianPositiveRateModel,
        "precision": LagrangianPrecisionModel,
        "multi": MultiLagrangianThresholdRateModel,
        "grad_norm": LagrangianGradNormModel,
    }
    the_class = class_dict.get(model_type, None)
    if the_class is None:
        raise ValueError("model_type not defined in group_regularized_model")
    return the_class


class LagrangianModel(TorchModel):
    """
        A model that implements a practical proxy-Lagrangian algorithm to learn with constraints
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.lagrange_multipliers = self.init_lagrange_multipliers(
            num_groups=self.config_dict.get("num_groups"), num_constraints_per_group=2
        )

    def init_lagrange_multipliers(self, num_groups=None, num_constraints_per_group=2):
        """
        Initializes the langrange multipliers
        """
        if num_groups is None:
            raise ValueError("num_groups must be provided")

        return torch.ones(
            num_constraints_per_group * num_groups,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        ) / (num_constraints_per_group * num_groups)

    def get_default_config(self):
        config_dict = super().get_default_config()
        update_dict = {
            "num_hidden": 1,
            "hidden_dim": 128,
            "drop_prob": 0.0,
            "normalize": False,
            "sparse": True,
            "sparse_mode": "csr",  # alternatively, "convert"
            "resnet": False,
            "lr_lambda": 1,
            "num_groups": 2,
            "num_constraints_per_group": 2,
            "multiplier_bound": 1,
            "constraint_slack": 0.05,
            "use_exact_constraints": True,
            "update_lambda_on_val": False,
            "surrogate": "logistic",
            "project_small_lambda": True,
        }
        return {**config_dict, **update_dict}

    def init_model(self):
        model = FeedforwardNet(
            in_features=self.config_dict["input_dim"],
            hidden_dim_list=self.config_dict["num_hidden"]
            * [self.config_dict["hidden_dim"]],
            output_dim=self.config_dict["output_dim"],
            drop_prob=self.config_dict["drop_prob"],
            normalize=self.config_dict["normalize"],
            sparse=self.config_dict["sparse"],
            sparse_mode=self.config_dict["sparse_mode"],
            resnet=self.config_dict["resnet"],
        )
        return model

    def constraint_metric(self, outputs, labels, sample_weight=None):
        """
            Defines a differentiable metric that will be used to construct the constraints during training
        """
        return weighted_cross_entropy_loss(outputs, labels, sample_weight=sample_weight)

    def constraint_metric_exact(self, outputs, labels, sample_weight=None):
        """
            Defines the metric used to update the Lagrange multipliers
            Does not necessarily need to be differentiable
        """
        return weighted_cross_entropy_loss(outputs, labels, sample_weight=sample_weight)

    def get_surrogate_fn(self):
        """
        (TODO) replicate this logic in the robustness.py and group_fairness.py
        """
        if self.config_dict.get("surrogate") is None:
            raise ValueError("No surrogate defined in config_dict")
        elif self.config_dict.get("surrogate") == "logistic":
            return logistic_surrogate
        elif self.config_dict.get("surrogate") == "hinge":
            return hinge_surrogate
        else:
            raise ValueError("Surrogate not defined")

    def get_loss_names(self):
        return ["loss", "supervised"]

    def forward_on_batch(self, the_data):
        loss_dict_batch = {}

        inputs, labels, group = (
            the_data["features"],
            the_data["labels"],
            the_data["group"],
        )

        if self.config_dict.get("weighted_loss"):
            sample_weight = the_data.get("weights")
            if sample_weight is None:
                raise ValueError("weighted_loss is True, but no weights provided")
        else:
            sample_weight = None

        outputs = self.model(inputs)
        # Compute the supervised loss
        if self.config_dict.get("weighted_loss"):
            loss_dict_batch["supervised"] = self.criterion(
                outputs, labels, sample_weight=sample_weight
            )
        else:
            loss_dict_batch["supervised"] = self.criterion(outputs, labels)

        constraint_values = self.compute_constraints(
            outputs=outputs,
            labels=labels,
            group=group,
            sample_weight=sample_weight,
            exact_constraints=False,
        )

        if self.config_dict.get("use_exact_constraints"):
            constraint_values_exact = self.compute_constraints(
                outputs=outputs,
                labels=labels,
                group=group,
                sample_weight=sample_weight,
                exact_constraints=True,
            )

        # Update the Lagrange multipliers
        if (
            self.is_training() and not self.config_dict.get("update_lambda_on_val")
        ) or (not self.is_training() and self.config_dict.get("update_lambda_on_val")):
            if self.config_dict.get("use_exact_constraints"):
                self.update_lagrange_multipliers(
                    constraint_values_exact,
                    additive_update=self.config_dict.get("additive_update"),
                )
            else:
                self.update_lagrange_multipliers(
                    constraint_values,
                    additive_update=self.config_dict.get("additive_update"),
                )

        if self.config_dict.get("additive_update"):
            langrange_multipliers_exp = torch.exp(self.lagrange_multipliers)
            multiplier_sum = langrange_multipliers_exp.sum()
            if self.config_dict.get("project_small_lambda") or (
                multiplier_sum > self.config_dict.get("multiplier_bound")
            ):
                # Apply the update from Agarwal et al 2018
                lagrange_multipliers_normalized = (
                    self.config_dict.get("multiplier_bound")
                    * langrange_multipliers_exp
                    / (1 + langrange_multipliers_exp.sum())
                )
            else:
                lagrange_multipliers_normalized = langrange_multipliers_normalized

            loss_dict_batch["loss"] = loss_dict_batch["supervised"] + torch.dot(
                lagrange_multipliers_normalized, constraint_values
            )
        else:
            loss_dict_batch["loss"] = loss_dict_batch["supervised"] + torch.dot(
                self.lagrange_multipliers, constraint_values
            )

        return loss_dict_batch, outputs

    def compute_constraints(
        self, outputs, labels, group, sample_weight=None, exact_constraints=False
    ):
        """
            For a problem subject to constraints satisfying Ax <= b,
                this method returns Ax - b
        """
        if exact_constraints:
            constraint_metric_fn = self.constraint_metric_exact
        else:
            constraint_metric_fn = self.constraint_metric

        # Compute the value of the constraint metric on the marginal population (over all groups)

        # Initialize a tensor to hold the value of the constraint metric for each group
        constraint_metric_value_group = torch.zeros(
            self.config_dict.get("num_groups")
        ).to(self.device)

        # Initialize a tensor to hold the value of the constraints
        constraint_values = torch.zeros((self.config_dict.get("num_groups"), 2)).to(
            self.device
        )

        try:
            constraint_metric_value_overall = constraint_metric_fn(
                outputs, labels, sample_weight=sample_weight
            )
        except MetricUndefinedError:
            # if self.config_dict.get("print_debug"):
            logging.debug("Warning: metric undefined")
            return constraint_values.reshape(-1)

        for the_group in group.unique():
            the_group = int(the_group.item())

            # Subset the outputs and labels
            outputs_group = outputs[group == the_group]
            labels_group = labels[group == the_group]

            if self.config_dict.get("weighted_loss"):
                sample_weight_group = sample_weight[group == the_group]
            else:
                sample_weight_group = None

            # Compute the value of the constraint metric for each group
            try:
                constraint_metric_value_group[the_group] = constraint_metric_fn(
                    outputs_group, labels_group, sample_weight=sample_weight_group
                )

                logging.debug(
                    "Constraint metric values for group {}, exact {}: {}".format(
                        the_group,
                        exact_constraints,
                        constraint_metric_value_group[the_group],
                    )
                )

                # Compute the first side of the inequality constraint
                constraint_values[the_group][0] = (
                    constraint_metric_value_group[the_group]
                    - constraint_metric_value_overall
                    - self.config_dict["constraint_slack"]
                )

                # Compute the second side of the inequality constraint
                constraint_values[the_group][1] = (
                    constraint_metric_value_overall
                    - constraint_metric_value_group[the_group]
                    - self.config_dict["constraint_slack"]
                )
            except MetricUndefinedError:
                logging.debug("Warning: metric undefined")
                continue
        constraint_values = constraint_values.reshape(-1)

        return constraint_values

    def update_lagrange_multipliers(self, constraint_values, additive_update=False):
        """
            Updates the Lagrange multipliers
        """
        if self.config_dict.get("additive_update"):
            self.lagrange_multipliers = self.lagrange_multipliers + (
                self.config_dict.get("lr_lambda") * constraint_values
            )
        else:
            self.lagrange_multipliers = self.lagrange_multipliers * torch.exp(
                self.config_dict.get("lr_lambda") * constraint_values.detach()
            )
            multiplier_sum = self.lagrange_multipliers.sum()
            if self.config_dict.get("project_small_lambda") or (
                multiplier_sum > self.config_dict.get("multiplier_bound")
            ):
                self.lagrange_multipliers = (
                    self.config_dict.get("multiplier_bound", 1)
                    * self.lagrange_multipliers
                    / multiplier_sum
                )
            else:
                pass

            logging.debug("Training Lambda: {}".format(self.lagrange_multipliers))
            logging.debug("Lambda sum: {}".format(multiplier_sum))


class LagrangianAUCModel(LagrangianModel):
    def get_default_config(self):
        """
        Default parameters
        """
        config_dict = super().get_default_config()
        update_dict = {
            "use_exact_constraints": True,
        }
        return {**config_dict, **update_dict}

    def compute_metric(self, outputs, labels, sample_weight=None, surrogate_fn=None):
        return roc_auc_score_surrogate(
            outputs=outputs,
            labels=labels,
            sample_weight=sample_weight,
            surrogate_fn=surrogate_fn,
        )

    def constraint_metric(self, outputs, labels, sample_weight=None):

        return self.compute_metric(
            outputs=outputs,
            labels=labels,
            sample_weight=sample_weight,
            surrogate_fn=self.get_surrogate_fn(),
        )

    def constraint_metric_exact(self, outputs, labels, sample_weight=None):
        return self.compute_metric(
            outputs=outputs,
            labels=labels,
            sample_weight=sample_weight,
            surrogate_fn=indicator,
        )


class LagrangianGradNormModel(LagrangianModel):
    """
    Caveats:
        * Support for backwards through sparse matrix grads is poor
            * When computing gradient norms, this class skips the input layer
    """

    def get_default_config(self):
        """
        Default parameters
        """
        config_dict = super().get_default_config()
        update_dict = {"use_exact_constraints": False, "skip_input_grad": True}
        return {**config_dict, **update_dict}

    def compute_metric(self, outputs, labels, sample_weight=None):
        loss = weighted_cross_entropy_loss(outputs, labels, sample_weight=sample_weight)

        return self.compute_grad_norm(loss)

    def constraint_metric(self, outputs, labels, sample_weight=None):
        return self.compute_metric(
            outputs=outputs, labels=labels, sample_weight=sample_weight,
        )

    def constraint_metric_exact(self, outputs, labels, sample_weight=None):
        return self.constraint_metric(
            outputs=outputs, labels=labels, sample_weight=sample_weight,
        )


class LagrangianThresholdRateModel(LagrangianModel):
    def get_default_config(self):
        config_dict = super().get_default_config()
        update_dict = {"use_exact_constraints": True, "threshold": 0.5}
        return {**config_dict, **update_dict}

    def compute_metric(self, outputs, labels, sample_weight=None):
        raise NotImplementedError

    def constraint_metric(self, outputs, labels, sample_weight=None):
        return self.compute_metric(
            outputs=outputs,
            labels=labels,
            sample_weight=sample_weight,
            threshold=self.config_dict.get("threshold"),
            surrogate_fn=self.get_surrogate_fn(),
        )

    def constraint_metric_exact(self, outputs, labels, sample_weight=None):
        return self.compute_metric(
            outputs,
            labels,
            sample_weight=sample_weight,
            threshold=self.config_dict.get("threshold"),
            surrogate_fn=indicator,
        )


class LagrangianTPRModel(LagrangianThresholdRateModel):
    def compute_metric(
        self, outputs, labels, sample_weight=None, threshold=0.5, surrogate_fn=None
    ):
        return tpr_surrogate(
            outputs=outputs,
            labels=labels,
            sample_weight=sample_weight,
            threshold=threshold,
            surrogate_fn=surrogate_fn,
        )


class LagrangianFPRModel(LagrangianThresholdRateModel):
    def compute_metric(
        self, outputs, labels, sample_weight=None, threshold=0.5, surrogate_fn=None
    ):
        return fpr_surrogate(
            outputs=outputs,
            labels=labels,
            sample_weight=sample_weight,
            threshold=threshold,
            surrogate_fn=surrogate_fn,
        )


class LagrangianPositiveRateModel(LagrangianThresholdRateModel):
    def compute_metric(
        self, outputs, labels, sample_weight=None, threshold=0.5, surrogate_fn=None
    ):
        return positive_rate_surrogate(
            outputs=outputs,
            labels=labels,
            sample_weight=sample_weight,
            threshold=threshold,
            surrogate_fn=surrogate_fn,
        )


class LagrangianPrecisionModel(LagrangianThresholdRateModel):
    def compute_metric(
        self, outputs, labels, sample_weight=None, threshold=0.5, surrogate_fn=None
    ):
        return precision_surrogate(
            outputs=outputs,
            labels=labels,
            sample_weight=sample_weight,
            threshold=threshold,
            surrogate_fn=surrogate_fn,
        )


class MultiLagrangianThresholdRateModel(LagrangianModel):
    """
        An alternative implementation of the LagrangianModel that can impose constraints over multiple threshold-based metrics
        (TODO) generalize implementation to allow constraints over any combination of metrics
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        num_constraints_per_group = (
            2
            * len(self.config_dict.get("thresholds"))
            * len(self.config_dict.get("constraint_metrics"))
        )
        self.config_dict["num_constraints_per_group"] = num_constraints_per_group

        self.lagrange_multipliers = self.init_lagrange_multipliers(
            num_groups=self.config_dict.get("num_groups"),
            num_constraints_per_group=num_constraints_per_group,
        )
        self.constraint_metric_fn_dict = self.get_constraint_metric_fn_dict(
            metrics=self.config_dict.get("constraint_metrics")
        )

    def get_default_config(self):
        config_dict = super().get_default_config()
        update_dict = {
            "use_exact_constraints": True,
            "thresholds": [0.5],
            "constraint_metrics": ["tpr", "fpr"],
        }

        return {**config_dict, **update_dict}

    def get_constraint_metric_fn_dict(self, metrics=None):
        result = {
            "tpr": tpr_surrogate,
            "fpr": fpr_surrogate,
            "positive_rate": positive_rate_surrogate,
            "precision": precision_surrogate,
        }
        if metrics is None:
            raise ValueError("Must provide metrics to get_constraint_metric_fn_dict")
        else:
            result = {key: result[key] for key in metrics if key in result.keys()}
            if len(result) == 0:
                raise ValueError("No valid constraint metrics provided")
            return result

    def compute_constraints(
        self, outputs, labels, group, sample_weight=None, exact_constraints=False
    ):

        if exact_constraints:
            surrogate_fn = indicator
        else:
            surrogate_fn = self.get_surrogate_fn()

        # Initialize a tensor to hold the value of the constraints
        constraint_values = torch.zeros(
            (
                self.config_dict.get("num_groups"),
                len(self.config_dict["constraint_metrics"]),
                len(self.config_dict["thresholds"]),
                2,
            )
        ).to(self.device)

        for i, constraint_metric_fn in enumerate(
            self.constraint_metric_fn_dict.values()
        ):
            for j, threshold in enumerate(self.config_dict["thresholds"]):

                try:
                    constraint_metric_value_overall = constraint_metric_fn(
                        outputs=outputs,
                        labels=labels,
                        threshold=threshold,
                        sample_weight=sample_weight,
                        surrogate_fn=surrogate_fn,
                    )
                except MetricUndefinedError:
                    # if self.config_dict.get("print_debug"):
                    logging.debug("Warning: metric undefined")
                    continue

                # Initialize a tensor to hold the value of the constraint metric for each group
                constraint_metric_value_group = torch.zeros(
                    self.config_dict.get("num_groups")
                ).to(self.device)

                for the_group in group.unique():
                    the_group = int(the_group.item())

                    # Subset the outputs and labels
                    outputs_group = outputs[group == the_group]
                    labels_group = labels[group == the_group]

                    if self.config_dict.get("weighted_loss"):
                        sample_weight_group = sample_weight[group == the_group]
                    else:
                        sample_weight_group = None
                    try:
                        # Compute the value of the constraint metric for each group
                        constraint_metric_value_group[the_group] = constraint_metric_fn(
                            outputs=outputs_group,
                            labels=labels_group,
                            threshold=threshold,
                            sample_weight=sample_weight_group,
                            surrogate_fn=surrogate_fn,
                        )
                        # if self.config_dict.get("print_debug"):
                        logging.debug(
                            "Constraint metric values for group {}, exact {}: {}".format(
                                the_group,
                                exact_constraints,
                                constraint_metric_value_group[the_group],
                            )
                        )

                        # Compute the first side of the inequality constraint
                        constraint_values[the_group][i][j][0] = (
                            constraint_metric_value_group[the_group]
                            - constraint_metric_value_overall
                            - self.config_dict["constraint_slack"]
                        )

                        # Compute the second side of the inequality constraint
                        constraint_values[the_group][i][j][1] = (
                            constraint_metric_value_overall
                            - constraint_metric_value_group[the_group]
                            - self.config_dict["constraint_slack"]
                        )

                    except MetricUndefinedError:
                        # if self.config_dict.get("print_debug"):
                        logging.debug("Warning: metric undefined")

        constraint_values = constraint_values.reshape(-1)
        # if self.config_dict.get("print_debug"):
        logging.debug(
            "Constraint values, exact {}, {}".format(
                exact_constraints, constraint_values
            )
        )
        assert constraint_values.shape == self.lagrange_multipliers.shape

        return constraint_values
