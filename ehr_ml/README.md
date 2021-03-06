# ehr_ml

ehr_ml is a python package for building models using EHR data. As part of the model building process, it offers the ability to learn clinical language model based representations (CLMBR) as described in Steinberg et al at https://pubmed.ncbi.nlm.nih.gov/33290879/.

There are four main groups of functionality in ehr_ml. The ability to:
1. Convert EHR and claims data into a common schema
2. Apply labeling functions on that schema in order to derive labels
3. Apply featurization schemes on those patients to obtain feature matrices
4. Perform other common tasks necessary for research with EHR data

https://ehr-ml.readthedocs.io/en/latest/ has the full documentation, including setup instructions and a tutorial using SynPuf data. 
