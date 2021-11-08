import os
import argparse
import pickle
import joblib
import pdb

import pandas as pd
import numpy as np

from sklearn.model_selection import KFold

#------------------------------------
# Arg parser
#------------------------------------
parser = argparse.ArgumentParser(
    description = "Split cohort stratified by admission year and task labels"
)

parser.add_argument(
    "--cohort_fpath",
    type = str,
    default = "/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/cohorts/",
    help = "path to save cohort"
)

parser.add_argument(
    "--cohort_id",
    type = str,
    default = "admissions",
    help = "which cohort to split"
)

parser.add_argument(
    "--tasks",
    nargs='+',
    type=str ,
    default=['hospital_mortality','LOS_7','readmission_30','icu_admission'],
    help="regularization parameter C"
)

parser.add_argument(
    "--seed",
    type = int,
    default = 44,
    help = "seed for deterministic training"
)

#------------------------------------
# Helper funs
#------------------------------------
def read_file(filename, columns=None, **kwargs):
    print(filename)
    load_extension = os.path.splitext(filename)[-1]
    if load_extension == ".parquet":
        return pd.read_parquet(filename, columns=columns,**kwargs)
    elif load_extension == ".csv":
        return pd.read_csv(filename, usecols=columns, **kwargs)
    
def split_cohort_by_year(
        df,
        seed,
        patient_col='person_id',
        index_year='admission_year',
        tasks=['hospital_mortality','LOS_7','readmission_30','icu_admission'],
        val_frac=0.15,
        test_frac=0.15,
        nfold=5
    ):
    
    assert (test_frac > 0.0) & (test_frac < 1.0) & (val_frac < 1.0)
    
    # Get admission year
    df['admission_year']=df['admit_date'].dt.year
    df['discharge_year']=df['discharge_date'].dt.year
    
    # Split into train, val, and test
    test = df.groupby(['admission_year']).sample(
        frac=val_frac+test_frac,
        random_state = seed
    ).assign(**{
        f"fold_id":'test'
    })
    
    val = test.groupby(['admission_year']).sample(
        frac=val_frac/(val_frac+test_frac),
        random_state = seed
    ).assign(**{
        f"fold_id":'val'
    })
    
    test = test.drop(index=val.index)
    test = test.append(val)

    train = df.drop(index=test.index)
    
    # split train into kfolds
    kf = KFold(
        n_splits=nfold,
        shuffle=True,
        random_state=seed
    )
    
    years = df['admission_year'].unique()
    
    for year in years:
        itrain = train.query(f"admission_year==@year")
        c=0
        
        for _, val_ids in kf.split(itrain[patient_col]):
            c+=1

            test = test.append(
                itrain.iloc[val_ids,:].assign(**{
                    f"fold_id":str(c)
                })
            )
    
    for task in tasks:
        assert(task in test.columns)
        
        test[f"{task}_fold_id"]=test['fold_id']
        
        if task == 'readmission_30':
            test.loc[test['hospital_mortality']==1,f'{task}_fold_id']='ignore'
            test.loc[test['hospital_mortality']==1,f'{task}']= np.nan
        
    return test.sort_index()
    
#-------------------------------------------------------------------
# run
#-------------------------------------------------------------------
if __name__ == "__main__":
    
    args = parser.parse_args()
    np.random.seed(args.seed)
    
    cohort = read_file(
        os.path.join(
            args.cohort_fpath,
            args.cohort_id,
            "cohort/cohort.parquet"
        ),
        engine='pyarrow'
    )
    
    # split cohort
    cohort = split_cohort_by_year(
        cohort,
        args.seed
    )

    # save splitted cohort
    cohort.to_parquet(
        os.path.join(
            args.cohort_fpath,
            args.cohort_id,
            "cohort/cohort_split.parquet"
        ),
        engine="pyarrow",
    )