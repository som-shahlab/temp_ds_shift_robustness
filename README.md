# Model Robustness to Temporal Distribution Shift 

This repository contains the code to reproduce "EHR Foundation Models Improve Model Robustness in the Presence of Temporal Distribution Shift"


#### Repo Structure
- prediction_utils:
    - Library that defines cohort definition, feature extraction, and model evaluation pipelines.
- ehr_ml:
    - CLMBR (see [paper](https://pubmed.ncbi.nlm.nih.gov/33290879/) and [original repo](https://github.com/som-shahlab/ehr_ml)) library used in this study.
- clmbr:
    - This codebase calls prediction_utils and ehr_ml to run the experiments in the paper "EHR Foundation Models Improve Model Robustness in the Presence of Temporal Distribution Shift"
- dg:
    - This codebase calls prediction_utils to run experiments that evaluate domain generalization on improving model robustness under temporal distribution shift in clinical medicine.