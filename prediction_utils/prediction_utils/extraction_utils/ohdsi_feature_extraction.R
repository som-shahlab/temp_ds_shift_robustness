library(optparse)
library(FeatureExtraction)
library(SqlRender)
library(ffbase)
library(stringr)
library(tidyverse)
library(getopt)

option_list <- list(
    make_option(c("--data_path"), type = "character", default = "/labs/shahlab/projects/spfohl/cohorts/scratch"),
    make_option(c("--database"), type = "character", default = "cdm_stride8_20191214"),
    make_option(c("--schema"), type = "character", default = "public"),
    make_option(c("--rs_schema"), type = "character", default = "plp_cohort_tables"),
    make_option(c("--cohort_name"), type = "character", default = "admission_rollup_with_labels"),
    make_option(c("--cohort_id"), type = "numeric", default = -1),
    make_option(c("--row_id_field"), type = "character", default = "prediction_id"),
    make_option(c("--db_config"), type = "character", default = "~/.pg.cnf"),
    make_option(c("--covariate_settings"), type = "character", default = "drug"),
    make_option(c("--limit"), type = "numeric", default = 0),
    make_option(
        c("--fftempdir"),
        type = "character",
        default = "/labs/shahlab/projects/spfohl/fftemp/"
        )
    )

arg_parser <- OptionParser(option_list = option_list)
args <- parse_args(arg_parser)
options(fftempdir = args$fftempdir)

# Source the covariate settings file
source(file.path(dirname(getopt::get_Rscript_filename()), "ohdsi_covariate_settings.R"))

# Read the login details
config <- read.table(args$db_config,
    sep = "=",
    skip = 1,
    col.names = c("param", "value"),
    stringsAsFactors = FALSE
) %>%
    mutate(param = trimws(param), value = trimws(value))

login_details <- config$value %>% setNames(config$param)

# write_path <- file.path(args$data_path, args$schema, args$covariate_settings)
write_path <- args$data_path
covariates_path <- file.path(write_path, "covariates_ff")
csv_path <- file.path(write_path, "covariates_csv")

connectionDetails <- DatabaseConnector::createConnectionDetails(
    dbms = "postgresql",
    server = paste0(login_details["host"], "/", args$database),
    user = login_details["user"],
    password = login_details["password"],
    port = login_details["port"]
)
settings <- covariateSettingList[args$covariate_settings]

if (args$limit != 0) {
    limit_str = sprintf("LIMIT %s", args$limit)
} else {
    limit_str = ""
}

conn <- connect(connectionDetails)
query <- render("
    DROP TABLE IF EXISTS temp_cohort_table;
    CREATE TEMP TABLE temp_cohort_table AS
    SELECT subject_id, @row_id_field, cohort_start_date, 0 as cohort_definition_id
    FROM @rs_schema.@cohort_name
    @limit
    ",
    row_id_field = args$row_id_field, rs_schema = args$rs_schema, cohort_name = args$cohort_name, limit = limit_str
)
print(query)
dbExecute(conn, query)

index_query <- "CREATE INDEX idx_subject_id ON temp_cohort_table (subject_id ASC)"
dbExecute(conn, index_query)

covariateData <- getDbCovariateData(
    connection = conn,
    cdmDatabaseSchema = args$schema,
    cohortDatabaseSchema = args$rs_schema,
    cohortTable = "temp_cohort_table",
    cohortTableIsTemp = TRUE,
    cohortId = args$cohort_id,
    rowIdField = args$row_id_field,
    covariateSettings = settings
)

# Save the data as both ff and as csv
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
