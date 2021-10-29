library(tidyverse)
library(PatientLevelPrediction)
library(ParallelLogger)

read_db_config <- function(db_config_path, skip = 1, sep = "=") {
    # Reads a file containing database credentials
    # We expect one row per parameter, with a separator character
    # Example:
    # <header>
    # user = <username>
    # password = <password>
    # host = <hostname>
    # database = <database>
    # port = <port>
    config <- read.table(db_config_path,
        sep = sep,
        skip = skip,
        col.names = c("param", "value"),
        stringsAsFactors = FALSE
    ) %>% mutate(param = trimws(param), value = trimws(value))
    login_details <- config$value %>% setNames(config$param)
    return(login_details)
}

get_installed_packages <- function(write_path = "./installed_packages.csv") {
    # Gets the set of installed packages and their versions as a dataframe
    # See: https://www.r-bloggers.com/list-of-user-installed-r-packages-and-their-versions/
    ip <- as.data.frame(installed.packages()[, c(1, 3:4)])
    rownames(ip) <- NULL
    ip <- ip[is.na(ip$Priority), 1:2, drop = FALSE]
    if (write_path != NULL) {
        write.csv(ip, write_path, row.names = FALSE)
    }
    return(ip)
}

personSplitterCustom <- function(population, test = 0.3, train = NULL, nfold = 3, seed = NULL) {
    ## Splits data based on patients, not rowId. To be used as a replacement for subjectSplitter for older versions of PatientLevelPrediction

    # Check logger
    if (length(ParallelLogger::getLoggers()) == 0) {
        logger <- ParallelLogger::createLogger(
            name = "SIMPLE",
            threshold = "INFO",
            appenders = list(ParallelLogger::createConsoleAppender(layout = ParallelLogger::layoutTimestamp))
        )
        ParallelLogger::registerLogger(logger)
    }
    if (!is.null(seed)) {
        set.seed(seed)
    }

    if (!class(nfold) %in% c("numeric", "integer") | nfold < 1) {
        stop("nfold must be an integer 1 or greater")
    }

    if (!class(test) %in% c("numeric", "integer") | test <= 0 | test >= 1) {
        stop("test must be between 0 and 1")
    }

    if (is.null(train)) {
        train <- 1 - test
    }

    if (train + test > 1) {
        stop("train + test must be less than 1")
    }

    leftover <- max(0, 1 - train - test)


    if (length(table(population$outcomeCount)) <= 1 | sum(population$outcomeCount > 0) < 10) {
        stop("Outcome only occurs in fewer than 10 people or only one class")
    }

    if (floor(sum(population$outcomeCount > 0) * test / nfold) == 0) {
        stop("Insufficient outcomes for choosen nfold value, please reduce")
    }

    ParallelLogger::logInfo(paste0(
        "Creating a ",
        test * 100,
        "% test and ",
        train * 100,
        "% train (into ",
        nfold,
        " folds) stratified split by person"
    ))

    # Get the unique set of subjects
    subject_df <- population %>%
        select(subjectId) %>%
        distinct()
    # Shuffle the subjects
    subject_df <- subject_df %>% sample_n(nrow(subject_df), replace = FALSE)

    # Record the number of samples in each split
    num_test <- floor(test * nrow(subject_df))
    num_leftover <- max(c(0, floor(leftover * nrow(subject_df))))
    num_train <- nrow(subject_df) - num_test - num_leftover
    # Get the subjects in the test set
    test_subject_df <- subject_df %>%
        slice(1:num_test) %>%
        mutate(index = -1)
    # Get the subjects that are in neither train nor test
    if (num_leftover == 0) {
        # We take a dataframe with zero rows if num_leftover==0
        leftover_subject_df <- subject_df %>% slice(0)
    } else {
        leftover_subject_df <- subject_df %>%
            anti_join(test_subject_df) %>%
            slice(1:num_leftover) %>%
            mutate(index = 0)
    }
    # Assign fold ids to the training set
    train_subject_df <- subject_df %>%
        anti_join(rbind(test_subject_df, leftover_subject_df)) %>%
        mutate(index = rep(seq.int(from = 1, to = nfold, by = 1),
            times = floor(num_train / nfold),
            length.out = num_train
        ))

    # Combine the sets into a dataframe with columns (subjectId, index)
    subject_df <- rbind(test_subject_df, leftover_subject_df, train_subject_df)
    # Map the results to the population data frame
    split <- population %>% inner_join(subject_df)
    # Produce a summary
    summary_df <- split %>%
        group_by(index) %>%
        summarise(subject_count = length(unique(subjectId)), row_count = n())
    ParallelLogger::logInfo(paste0(
        "Data split into ",
        select(filter(summary_df, index == -1), subject_count), " test subjects, ",
        select(filter(summary_df, index == -1), row_count), " test rows, ",
        select(filter(summary_df, index > 0), subject_count), " train subjects, ",
        select(filter(summary_df, index > 0), row_count), " train rows "
    ))

    split <- split %>%
        select(rowId, index) %>%
        arrange(rowId)
    return(split)
}
