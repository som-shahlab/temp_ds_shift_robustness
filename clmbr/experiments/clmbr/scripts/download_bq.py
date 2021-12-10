import argparse
import os
import threading

from google.cloud import bigquery
from google.cloud import storage

from functools import partial

parser = argparse.ArgumentParser(description='Download a bigquery dataset')

parser.add_argument(
    '--project', 
    type=str,
    default="som-nero-nigam-starr",
    help='The project with the dataset',
)

parser.add_argument(
    '--dataset', 
    type=str,
    default="starr_omop_cdm5_deid_20210723",
    help='The name of the dataset to download',
)

parser.add_argument(
    '--target_fpath',
    type=str,
    default="/local-scratch/nigam/projects/lguo/temp_ds_shift_robustness/clmbr/experiments/clmbr/data/starr_omop",
    help='target directory to store the dataset',
)


if __name__ == "__main__":
    

    args = parser.parse_args()

    # Construct a BigQuery client object.
    client = bigquery.Client(project=args.project)

    storage_client = storage.Client(project=args.project)

    dataset_id = f'{args.project}.{args.dataset}'

    bucket_name = f'{args.project}-temp-extract-01'
    
    print(dataset_id)
    os.makedirs(f"{args.target_fpath}{dataset_id}",exist_ok=True)
    
    # create temporary storage bucket
    bucket = storage_client.bucket(bucket_name)
    bucket = storage_client.create_bucket(bucket, location="us-west2")

    tables = client.list_tables(dataset_id)

    extract_config = bigquery.job.ExtractJobConfig(
        compression=bigquery.job.Compression.GZIP,
        destination_format=bigquery.job.DestinationFormat.CSV,
        field_delimiter=','
    )

    sem = threading.Semaphore(value=0)
    needed = 0

    def download(table_id, f):
        print('Downloading to ', table_id, f)
        blobs = storage_client.list_blobs(bucket, prefix=table_id + '/')
        target_folder = os.path.join(args.target_fpath, dataset_id, table_id)
        os.makedirs(target_folder,exist_ok=True)
        for blob in blobs:
            blob.download_to_filename(os.path.join(target_folder, blob.name.split('/')[-1]))
            blob.delete()
        sem.release()

    print("Tables contained in '{}':".format(dataset_id))
    for table in tables:
        if table.table_id in ('note', 'note_nlp', 'observation'):
            continue

        print("{}.{}.{}".format(table.project, table.dataset_id, table.table_id))
        target_path = f'gs://{bucket_name}/{table.table_id}/*.csv.gz'
        extract_job = client.extract_table(table.reference, target_path, job_config=extract_config)
        needed += 1

        blob = bucket.blob(f'{table.table_id}')
        target_filename = f'{dataset_id}/{table.table_id}.csv.gz'

        extract_job.add_done_callback(partial(download, table.table_id))

    for i in range(needed):
        print(f'{i} out of {needed}')
        sem.acquire()

    bucket.delete()