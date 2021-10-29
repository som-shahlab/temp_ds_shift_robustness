## Prediction Utils

This library is composed of the following modules:
* `extraction_utils`: API for connecting to databases and extracting features
* `cohorts`: Definition for cohorts defined on the OMOP CDM
* `extraction_scripts`: Runtime scripts for defining cohorts and extracting features
* `pytorch_utils`: Pytorch models and training loops for supervised learning
* `vignettes`: Example workflows to help get started

### Installation
1. Clone the repository
2. Install with pip
    * Option 1: `pip install .` from within the directory to use install with a static copy of the code
    * Option 2: `pip install -e .` for within the directory to install in "editable" mode

### Getting Started
* See the `vignettes` directory for examples of basic usage

### Modules

#### extraction_utils
* Connect to databases using the BigQuery client library or through the python DBAPI via SqlAlchemy
* Extract clinical features for machine learning using custom code or with the OHDSI feature extractors

#### cohorts
* The following cohorts are implemented
    * Inpatient admissions rolled up to continuous episodes of care
        * Labeling functions for this cohort
            * Hospital mortality
            * Length of stay
            * 30-day readmission
    * Atherosclerotic Cardiovascular Disease (ASCVD)

#### pytorch_utils
* Several pipelines are implemented
    * Dataloaders
        * Suport for sparse data (e.g. in scipy.sparse.csr_matrix format)
    * Layers
        * Input layers that efficiently handle sparse inputs
        * Feedforward networks
    * Training pipelines
        * Supervised learning for binary outcomes
        * Group Fairness
            * Regularized, adversarial, approximate constraints with the practical proxy-Lagrangian
        * Distributional robustness to subpopulation shift
    * Comprehensive evaluation procedures for performance and fairness metrics on groups
