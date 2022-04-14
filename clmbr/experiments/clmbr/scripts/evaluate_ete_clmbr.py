import os
import argparse
import pickle
import joblib
import pdb
import re

import pandas as pd
import numpy as np

from prediction_utils.pytorch_utils.metrics import StandardEvaluator

from prediction_utils.util import str2bool

from ehr_ml.clmbr import convert_patient_data
from ehr_ml.clmbr import PatientTimelineDataset
from ehr_ml.clmbr.dataset import DataLoader
from ehr_ml.clmbr.prediction_model import BinaryLinearCLMBRClassifier
from ehr_ml.clmbr.utils import read_config, read_info

#------------------------------------
# Arg parser
#------------------------------------
parser = argparse.ArgumentParser(
    description = "Evaluate best lr model"
)

parser.add_argument(
    "--artifacts_fpath",
    type = str,
    default = "/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/clmbr/experiments/clmbr/clmbr_artifacts",
    help = "path to clmbr adapter model artifacts"
)

parser.add_argument(
    "--base_artifacts_fpath",
    type = str,
    default = "/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/clmbr/experiments/baseline/artifacts",
    help = "path to count feature model artifacts"
)

parser.add_argument(
    '--cohort_fpath',
    type=str,
    default="/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/clmbr/cohorts/admissions/cohort",
)

parser.add_argument(
    "--count_features_fpath",
    type = str,
    default = '/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/clmbr/features/',
    help = "path to count features"
)

parser.add_argument(
    '--extracts_fpath', 
    type=str,
    default="/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/clmbr/experiments/clmbr/data/extracts/20210723",
)


parser.add_argument(
    "--tasks",
    type=str,
    default="hospital_mortality/LOS_7/readmission_30/icu_admission",
    help="prediction tasks"
)

parser.add_argument(
    "--train_group", 
    type=str ,
    default = "2009/2010/2011/2012",
    help="group(s) to train on [2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]"
)

parser.add_argument(
    "--clmbr_encoder",
    type=str,
    default='gru',
    help='gru/transformer',
)

parser.add_argument(
    "--seed",
    type = int,
    default = 44,
    help = "seed for deterministic training"
)

parser.add_argument(
    "--n_boot",
    type = int,
    default = 1000,
    help = "num bootstrap iterations"
)

parser.add_argument(
    "--n_jobs",
    type = int,
    default = 4,
    help = "num jobs"
)

parser.add_argument(
    "--overwrite",
    type = str2bool,
    default = "false",
    help = "whether to overwrite existing artifacts",
)

#-------------------------------------------------------------------
# helper functions
#-------------------------------------------------------------------
def get_best_model(fpath):
    
    evaluator=StandardEvaluator(metrics=['loss_bce'])
    
    all_models = [
        x for x in os.listdir(fpath)
        if 'best' not in x
    ]
    
    df = pd.DataFrame()
    for x in all_models:
    
        try:
            df = pd.concat((
                df,
                pd.read_csv(f"{fpath}/{x}/val_pred_probs.csv").assign(
                    model_num=x,
                ) 
            ))
        except:
            print(f"{fpath}/{x} missing val_pred_probs.csv")
    
    
    # find best hparam setting based on log loss
    df_eval = evaluator.evaluate(
        df,
        strata_vars=['model_num']
    )
    
    df_eval = df_eval.groupby(['metric','model_num']).agg(
        mean_performance=('performance','mean')
    ).reset_index()
    
    model_num = int(df_eval.loc[
        df_eval.query("metric=='loss_bce'")['mean_performance'].idxmin(),
        'model_num'
    ])
    
    # instantiate model and load weights
    model_dir = os.path.join(fpath,str(model_num))
    
    model = BinaryLinearCLMBRClassifier(
        read_config(os.path.join(model_dir,'config.json')),
        read_info(os.path.join(model_dir,'info.json')),
    )
    
    model.load_weights(model_dir)
    
    return model

    
#-------------------------------------------------------------------
# run
#-------------------------------------------------------------------
args = parser.parse_args()
    
# set seed
np.random.seed(args.seed)

# parse tasks and train_group
args.tasks = args.tasks.split("/")
args.train_group = [int(x) for x in args.train_group.split("/")]



df_comp = pd.DataFrame()

for task in args.tasks:
    
    #-------------- selection of end-to-end model -----------------#
    fpath = os.path.join(
        args.artifacts_fpath,
        "models",
        "_".join([str(args.train_group[0]), str(args.train_group[-1]), 'end_to_end']),
        args.clmbr_encoder,
        task,
    )
    
    
    model=get_best_model(fpath)
    
    #-------------- get model preds on test set -----------------#
    df_cohort = pd.read_parquet(
        os.path.join(
            args.cohort_fpath,
            "cohort_split.parquet",
        )
    )
    
    if task == 'readmission_30':
        df_cohort = df_cohort.assign(
            date = pd.to_datetime(df_cohort['discharge_date']).dt.date
        )
        
        index_year='discharge_year'
    
    else:
        df_cohort = df_cohort.assign(
            date = pd.to_datetime(df_cohort['admit_date']).dt.date
        )
        
        index_year='admission_year'
        
    train = df_cohort.query(
            f"{task}_fold_id!=['val','test','ignore'] and admission_year==@args.train_group"
        )
    
    test = df_cohort.query(
            f"{task}_fold_id==['test']"
        )
    
    test_person_ids, test_day_ids = convert_patient_data(
        args.extracts_fpath, 
        test['person_id'], 
        test['date']
    )
    
    train_person_ids, train_day_ids = convert_patient_data(
        args.extracts_fpath, 
        train['person_id'], 
        train['date']
    )
    
    train_labels = train[f"{task}"].to_numpy()
    test_labels = test[f"{task}"].to_numpy()
    
    train_pred_ids = train['prediction_id'].to_numpy()
    test_pred_ids = test['prediction_id'].to_numpy()
    
    assert(len(train_labels)==len(train_person_ids)==len(train_day_ids)==len(train_pred_ids)==len(train_pred_ids))
    assert(len(test_labels)==len(test_person_ids)==len(test_day_ids)==len(test_pred_ids)==len(test_pred_ids))
    
    test_df = pd.DataFrame({
        'pid':test_person_ids,
        'day_ids':test_day_ids,
        'labels':test_labels,
        'prediction_id':test_pred_ids,
    })
    
    test_df=test_df[['pid','prediction_id','day_ids']].merge(df_cohort, how='inner', on='prediction_id')

    #get dataset and dataloader
    info = read_info(os.path.join(model.config['model_dir'],'info.json'))
    
    dataset = PatientTimelineDataset(
        os.path.join(info["extract_dir"], "extract.db"),
        os.path.join(info["extract_dir"], "ontology.db"),
        os.path.join(model.config['model_dir'], "info.json"),
        (train_labels, train_person_ids, train_day_ids),
        (test_labels, test_person_ids, test_day_ids),
    )
    
    dataloader = DataLoader(
        dataset,
        threshold=model.config['num_first'],
        is_val=True
    )
    
    df_preds = model.predict(dataloader)
    
    df_preds = df_preds.merge(test_df,on='pid')
    df_preds['test_group'] = df_preds[index_year]
    df_preds['train_groups'] = '_'.join([str(x) for x in args.train_group])
    
    df = df_preds[[
        'pid',
        'prediction_id',
        'labels',
        'pred_probs',
        'train_groups',
        'test_group',
        'age_group',
        'race_eth',
        'gender_concept_name'
    ]].query("test_group!=[2007,2008]")
    
    df['train_groups'] = df['train_groups'].astype(str)
    df['test_group'] = df['test_group'].astype(str)
    
    
    #------------------------- evaluate -------------------- # 
    # evaluate stratify by year_group
    fpath = os.path.join(
        args.artifacts_fpath,
        "eval_end_to_end",
        task,
        '_'.join([
            args.clmbr_encoder,
            '_'.join([str(x) for x in args.train_group]),
        ])
    )
    
    os.makedirs(fpath, exist_ok=True)
    
    strata_vars_dict = {
        'group':['test_group'],
    }
    
    train_group = df['train_groups'].unique()[0]
    
    df = df.replace({
        '2009':'2009_2010_2011_2012',
        '2010':'2009_2010_2011_2012',
        '2011':'2009_2010_2011_2012',
        '2012':'2009_2010_2011_2012',
    })
    
    evaluator = StandardEvaluator(
        metrics=['auc','auprc','auprc_c','loss_bce','ace_abs_logistic_logit'],
        **{'pi0':df.query("test_group==@train_group")['labels'].mean()} # set prior for calibrated auprc
    )
    
    for k,v in strata_vars_dict.items():
        
        if all([
            os.path.exists(f"{fpath}/{f}") for f in 
            [f'by_{k}.csv',f'by_{k}_all_results.csv']
        ]) and not args.overwrite:

            print("Artifacts exist and args.overwrite is set to False. Skipping...")
            continue

        elif not all([
            os.path.exists(f"{fpath}/{f}") for f in 
            [f'by_{k}.csv',f'by_{k}_all_results.csv']
        ]) or args.overwrite: 
            
            df2 = df.query("test_group==['2013','2014','2015','2016','2017','2018','2019','2020','2021']")
            df2 = df2.replace({
                '2013':'2013_2014_2015_2016',
                '2014':'2013_2014_2015_2016',
                '2015':'2013_2014_2015_2016',
                '2016':'2013_2014_2015_2016',
                '2017':'2017_2018_2019_2020_2021',
                '2018':'2017_2018_2019_2020_2021',
                '2019':'2017_2018_2019_2020_2021',
                '2020':'2017_2018_2019_2020_2021',
                '2021':'2017_2018_2019_2020_2021',
            })
            
            df = pd.concat((df, df2))
            
            # get clmbr feature model evaluations
            df_eval_clmbr_ci, df_eval_clmbr = evaluator.bootstrap_evaluate(
                df,
                strata_vars_eval=v,
                strata_vars_boot=['labels'],
                patient_id_var='prediction_id',
                n_boot=args.n_boot,
                n_jobs=args.n_jobs,
                strata_var_experiment='test_group', 
                baseline_experiment_name=train_group,
                return_result_df=True
            )
            
            # save clmbr feature model evaluations
            df_eval_clmbr_ci.to_csv(f"{fpath}/by_{k}.csv",index=False)
            df_eval_clmbr.to_csv(f"{fpath}/by_{k}_all_results.csv",index=False)
            df.to_csv(f"{fpath}/pred_probs.csv")