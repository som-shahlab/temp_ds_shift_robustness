# pytorch_utils

This is the primary module with code for training and evaluating models in Pytorch.

It further contains utilities for training models that satisfy fairness and robustness goals over subpopulations.

Description of files

    * datasets.py
        * Functionality for generating data loaders
    * layers.py
        * Definitions of neural network layers and model architectures
    * models.py
        * Wrappers around models that encapsulate training code
    * metrics.py
        * Evaluators and metrics. Not dependent on torch and can be applied to any set of predictions and labels
    * group_fairness.py
        * Defines regularized objectives for group fairness
    * robustness.py
        * Defines distributionally robust objectives to optimize for worst-group performance
    * lagrangian.py
        * Defines approximately constrained objectives for constraints on data- and model- dependent quantities
    * pytorch_metrics.py
        * Differentiable metrics that can be used in loss functions in pytorch
    * metric_logging.py
        * Utilities to handle logging metrics during training
    * util.py
        * Miscellaneous functions
