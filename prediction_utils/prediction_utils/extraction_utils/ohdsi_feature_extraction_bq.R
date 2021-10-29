library(optparse)
library(FeatureExtraction)
library(SqlRender)
library(ffbase)
library(stringr)
library(tidyverse)
library(getopt)

option_list <- list(
    make_option(c("--data_path"), type = "character", default = "/share/pi/nigam/spfohl/cohorts/scratch"),
    make_option(c("--database"), type = "character", default = "som-nero-phi-nigam-starr"),
    make_option(c("--schema"), type = "character", default = "starr_omop_cdm5_deid_20200404"),
    make_option(c("--rs_schema"), type = "character", default = "plp_cohort_tables"),
    make_option(c("--cohort_name"), type = "character", default = "admission_rollup_20200404_with_labels"),
    make_option(c("--cohort_id"), type = "numeric", default = -1),
    make_option(c("--row_id_field"), type = "character", default = "prediction_id"),
    make_option(c("--db_config"), type = "character", default = "~/.pg.cnf"),
    make_option(c("--covariate_settings"), type = "character", default = "drug"),
    make_option(c("--limit"), type = "numeric", default = 0),
    make_option(
        c("--fftempdir"),
        type = "character",
        default = "/share/pi/nigam/spfohl/fftemp"
        ),
    make_option(
        c("--path_to_driver"),
        type = "character",
        # default = "/share/sw/open/simba/SimbaJDBCDriverforGoogleBigQuery42_1.2.1.1001"
        default = "/home/spfohl/jdbc"
        ),
    make_option(
        c("--credentials_path"),
        type = "character",
        default = "/home/spfohl/.config/gcloud/application_default_credentials.json"
        ),
    make_option(
        c("--email"),
        type = 'character',
        default = "spfohl@stanford.edu"
        )
    )

arg_parser <- OptionParser(option_list = option_list)
args <- parse_args(arg_parser)
print(args)
options(fftempdir = args$fftempdir)

# Source the covariate settings file
source(file.path(dirname(getopt::get_Rscript_filename()), "ohdsi_covariate_settings.R"))

write_path <- args$data_path
covariates_path <- file.path(write_path, "covariates_ff")
csv_path <- file.path(write_path, "covariates_csv")

connection_string=sprintf(
    "jdbc:bigquery://https://www.googleapis.com/bigquery/v2:443;ProjectId=%s;OAuthType=0;OAuthServiceAcctEmail=%s;OAuthPvtKeyPath=%s;Timeout=600;DefaultDataset=%s;EnableHighThroughputAPI=1",
    args['database'],
    args['email'],
    args['credentials_path'],
    'temp_dataset'
)

Sys.setenv(GOOGLE_APPLICATION_CREDENTIALS=args['credentials_path'])

connectionDetails <- DatabaseConnector::createConnectionDetails(
    dbms = "bigquery",
    connectionString=connection_string,
    user="",
    password="",
    pathToDriver=args$path_to_driver
)

settings <- covariateSettingList[args$covariate_settings]

if (args$limit != 0) {
    limit_str = sprintf("LIMIT %s", args$limit)
} else {
    limit_str = ""
}

conn <- connect(connectionDetails)

print('Connected')
covariateData <- getDbCovariateData(
    connection = conn,
    cdmDatabaseSchema = args$schema,
    cohortDatabaseSchema = args$rs_schema,
    cohortTable=args$cohort_name,
    cohortId = -1,
    rowIdField = args$row_id_field,
    covariateSettings = settings
)

print('Success')

#Save the data as both ff and as csv
if (dir.exists(covariates_path)) {
    unlink(covariates_path, recursive = TRUE)
}
saveCovariateData(covariateData, covariates_path)
if (dir.exists(csv_path)) {
    unlink(csv_path, recursive = TRUE)
}
dir.create(csv_path)
covariateData$covariates %>%
    as.ffdf() %>%
    write.csv.ffdf(file.path(csv_path, "covariates.csv"))
covariateData$analysisRef %>%
    as.ffdf() %>%
    write.csv.ffdf(file.path(csv_path, "analysisRef.csv"))
covariateData$covariateRef %>%
    as.ffdf() %>%
    write.csv.ffdf(file.path(csv_path, "covariateRef.csv"))
