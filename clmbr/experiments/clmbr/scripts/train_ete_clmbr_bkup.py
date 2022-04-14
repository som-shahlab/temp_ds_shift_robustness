import os
import json
import argparse
import yaml
import shutil
import pdb
import logging
import ehr_ml
import joblib
import torch

import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from subprocess import run
from tqdm import tqdm
from joblib import Parallel, delayed 
from sklearn.model_selection import ParameterGrid

from prediction_utils.util import str2bool
from ehr_ml.clmbr import convert_patient_data
from ehr_ml.clmbr import PatientTimelineDataset
from ehr_ml.clmbr import Trainer
from ehr_ml.clmbr.rnn_model import PatientRNN
from ehr_ml.clmbr.utils import read_info
from ehr_ml.clmbr.dataset import DataLoader
from ehr_ml.utils import set_up_logging


#------------------------------------
# Arg parser
#------------------------------------

parser = argparse.ArgumentParser(
    description='Train end-to-end CLMBR model'
)

parser.add_argument(
    '--task',
    type=str,
    default='hospital_mortality',
    help='task for end-to-end CLMRB training [hospital_mortality, icu_admission, LOS_7, readmission_30]',
)

parser.add_argument(
    '--min_patient_count', 
    type=str,
    default="100",
)

parser.add_argument(
    '--extracts_fpath', 
    type=str,
    default="/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/clmbr/experiments/clmbr/data/extracts/20210723",
)

parser.add_argument(
    '--cohort_fpath',
    type=str,
    default="/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/clmbr/cohorts/admissions/cohort",
)

parser.add_argument(
    '--infos_fpath',
    type=str,
    default="/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/clmbr/experiments/clmbr/clmbr_artifacts/infos/"
)

parser.add_argument(
    '--train_splits_fpath',
    type=str,
    default="/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/clmbr/experiments/clmbr/clmbr_artifacts/train_splits/"
)

parser.add_argument(
    '--models_fpath',
    type=str,
    default="/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/clmbr/experiments/clmbr/clmbr_artifacts/models/"
)

parser.add_argument(
    '--year_range',
    type=str,
    default="2009/2012",
    help="start and end of the year range for training year group"
) 

parser.add_argument(
    '--excluded_patient_list',
    type=str,
    default="/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/clmbr/experiments/clmbr/data/held_out_patients/excluded_patient_ids.txt"
)

parser.add_argument(
    '--hparams_fpath',
    type=str,
    default="/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/clmbr/experiments/clmbr/hyperparams/"
)

parser.add_argument(
    '--encoder',
    type=str,
    default='gru',
    help='Encoder type: GRU/Transformer',
)

parser.add_argument(
    '--overwrite',
    type=str2bool,
    default='false'
)

parser.add_argument(
    '--n_gpu',
    type=int,
    default=1
)

parser.add_argument(
    '--n_jobs',
    type=int,
    default=5
)

parser.add_argument(
    '--gpu_num_start',
    type=int,
    default=1
)

#-------------------------------------------------------------------
# CLMBR model (nn.Module) and Training func
#-------------------------------------------------------------------
class BinaryLinearCLMBRClassifier(nn.Module):
            
    def __init__(self, config, info):
        super().__init__()
        self.config=config
        self.info=info
        self.timeline_model = PatientRNN(config,info)
        self.linear = nn.Linear(config["size"], 1)
        self.device = torch.device(config['device'])
        self.criterion = nn.Sigmoid()
        self=self.to(self.device)

    def forward(self, batch):
        outputs = dict()
        
        #pdb.set_trace()
        
        embedding = self.timeline_model(batch["rnn"])

        label_indices, label_values = batch["label"]

        flat_embeddings = embedding.view((-1, embedding.shape[-1]))
        
        target_embeddings = F.embedding(label_indices, flat_embeddings) 
        
        logits = self.linear(target_embeddings).flatten()
        
        outputs['pids']=batch['pid']
        outputs['pred_probs'] = self.criterion(logits)
        outputs['labels'] = label_values
        outputs['loss'] = F.binary_cross_entropy_with_logits(
            logits, label_values.float(), reduction="sum"
        )
    
        
        return outputs
    
    def predict(self, dataloader):
        
        self.eval()
        
        pred_probs = []
        labels = []
        pids = []
        
        pbar = tqdm(total=dataloader.num_batches)
        with torch.no_grad():
            for batch in dataloader:
                outputs = self.forward(batch)
                pred_probs.extend(list(outputs['pred_probs'].cpu().numpy()))
                labels.extend(outputs['labels'].cpu().numpy())
                pids.extend(outputs['pids'])
                pbar.update(1)
                
        return pd.DataFrame({
            'pid': pids,
            'labels': labels,
            'pred_probs': pred_probs
        })
        
    
    def load_weights(self,model_dir):
        
        model_data = torch.load(
            os.path.join(model_dir,"best")
        )
        
        self.load_state_dict(model_data)
        
        return self
    

def TrainCLMBR(args,i,params,first_too_small_index):
    
    gpu_num = i % args.n_gpu + args.gpu_num_start
    
    model_dir=os.path.join(
        args.models_fpath,
        "_".join(args.year_range.split("/")) + '_end_to_end',
        args.encoder,
        args.task,
        f"{i}"
    )

    config = {
        'lr':params['lr'],
        'encoder_type':params['encoder_type'],
        "num_first": first_too_small_index,
        "num_second": len(info["valid_code_map"]) - first_too_small_index,
        'size':params['size'],
        'dropout':params['dropout'],
        'l2':params['l2'],
        'batch_size':params['batch_size'],
        'eval_batch_size':params['batch_size'],
        'epochs_per_cycle':50,
        'tied_weights':True,
        'warmup_epochs':params['warmup_epochs'],
        "b1": 0.9,
        "b2": 0.999,
        "e": 1e-8,
        'model_dir':model_dir,
        'rnn_layers':1,
        'day_dropout':0.2,
        'code_dropout':params['code_dropout'] if 'code_dropout' in params.keys() else 0.2,
        'transformer_layers': params['transformer_layers'] if 'transformer_layers' in params.keys() else 6,
        "device": f'cuda:{gpu_num}'
    }

    if args.overwrite or not os.path.exists(model_dir):
        shutil.rmtree(model_dir, ignore_errors=True)
        os.makedirs(model_dir, exist_ok=True)
        
        # setup logging
        ehr_ml.utils.set_up_logging(os.path.join(config['model_dir'],"train.log"))
        logging.info("Args: %s", str(config))

        # save model config
        with open(os.path.join(model_dir,'config.json'),"w") as f:
            json.dump(config,f)

        # instantiate & train model
        model = BinaryLinearCLMBRClassifier(config, info)

        trainer = Trainer(model)

        trainer.train(dataset, use_pbar=False)

        # get model predictions for validation set
        dataloader = DataLoader(
            dataset, 
            threshold = config['num_first'], 
            is_val=True
        )

        outputs = model.predict(dataloader)

        # save predictions 
        outputs.to_csv(os.path.join(model_dir, 'val_pred_probs.csv'),index=False)
        
    else:
        print("model dir exists and overwrite is set to False, skipping...")
        

    

    
#-------------------------------------------------------------------
# run
#-------------------------------------------------------------------
if __name__ == "__main__":
    
    args = parser.parse_args()
    
    # threads
    torch.set_num_threads(1)
    
    # create train and val patient IDs
    train_splits_dir = os.path.join(
        args.train_splits_fpath,
        "_".join(args.year_range.split("/")) + '_end_to_end',
        args.task,
    )
    
    if args.overwrite and os.path.exists(train_splits_dir):
        shutil.rmtree(train_splits_dir, ignore_errors=True)
        
    os.makedirs(train_splits_dir, exist_ok=True)
    
    year_range_list = [*range(
        int(args.year_range.split("/")[0]), 
        int(args.year_range.split("/")[-1])+1, 
        1,
    )]
    
    df_cohort = pd.read_parquet(
        os.path.join(
            args.cohort_fpath,
            "cohort_split.parquet",
        )
    )
    
    if args.task == 'readmission_30':
        df_cohort = df_cohort.assign(
            date = pd.to_datetime(df_cohort['discharge_date']).dt.date
        )
        
    else:
        df_cohort = df_cohort.assign(
            date = pd.to_datetime(df_cohort['admit_date']).dt.date
        )
    
    
    train = df_cohort.query(
        f"{args.task}_fold_id!=['val','test','ignore'] and admission_year==@year_range_list"
    )
    
    
    
    val = df_cohort.query(
        f"{args.task}_fold_id==['val'] and admission_year==@year_range_list"
    )
    
    # convert patient ids and save to train_splits path
    train_person_ids, train_day_ids = convert_patient_data(
        args.extracts_fpath, 
        train['person_id'], 
        train['date']
    )
    
    with open(
        os.path.join(
            train_splits_dir,
            f"train_patients.txt"
        ), 
        "w"
    ) as f:
        
        for pid in train_person_ids:
            f.write("%d\n" % pid)
    
    
    val_person_ids, val_day_ids = convert_patient_data(
        args.extracts_fpath, 
        val['person_id'], 
        val['date']
    )
    
    with open(
        os.path.join(
            train_splits_dir,
            f"val_patients.txt"
        ), 
        "w"
    ) as f:
        
        for pid in val_person_ids:
            f.write("%d\n" % pid)
    
    
    # create info
    info_dir=os.path.join(
        args.infos_fpath,
        "_".join(args.year_range.split("/"))+'_end_to_end',
        args.task
    )
    
    train_start_date=args.year_range.split("/")[0]
    train_end_date=args.year_range.split("/")[-1]
    val_start_date=args.year_range.split("/")[0]
    val_end_date=args.year_range.split("/")[-1]
    
    if args.overwrite and os.path.exists(info_dir):
        shutil.rmtree(info_dir, ignore_errors=True)
    
    run([
        'clmbr_create_info',
        f"{args.extracts_fpath}",
        f"{info_dir}",
        f"{train_end_date}-12-31",
        f"{val_end_date}-12-31",
        "--train_start_date", f"{train_start_date}-01-01",
        "--val_start_date", f"{val_start_date}-01-01",
        "--min_patient_count", args.min_patient_count,
        "--excluded_patient_file", args.excluded_patient_list,
        "--train_patient_file", f"{train_splits_dir}/train_patients.txt",
        "--val_patient_file", f"{train_splits_dir}/val_patients.txt",
    ])
    
    info=read_info(os.path.join(info_dir, 'info.json'))
    
    # create Patient Timeline dataset
    train_labels = train[f"{args.task}"].to_numpy()
    val_labels = val[f"{args.task}"].to_numpy()
    
    assert(len(train_labels)==len(train_person_ids)==len(train_day_ids))
    assert(len(val_labels)==len(val_person_ids)==len(val_day_ids))
    
    train_data = (train_labels, np.array(train_person_ids), np.array(train_day_ids))
    val_data = (val_labels, np.array(val_person_ids), np.array(val_day_ids))
    
    dataset = PatientTimelineDataset(
        os.path.join(args.extracts_fpath, "extract.db"), 
        os.path.join(args.extracts_fpath, "ontology.db"),
        os.path.join(info_dir, "info.json"),
        train_data,
        val_data
    )
    
    # get hyperparameter grid
    grid = list(
        ParameterGrid(
            yaml.load(
                open(
                    f"{os.path.join(args.hparams_fpath,args.encoder)}.yml",
                    'r'
                ),
                Loader=yaml.FullLoader
            )
        )
    )
    
    
    # get first_too_small_index
    first_too_small_index = float("inf")
    for code, index in info["valid_code_map"].items():
        if info["code_counts"][code] < 10 * info["min_patient_count"]:
            first_too_small_index = min(first_too_small_index, index)

    
    # train clmbr models
    Parallel(n_jobs=args.n_jobs, backend="threading")(
        delayed(TrainCLMBR)(args,i,params,first_too_small_index) 
        for i,params in enumerate(grid)
    )
        