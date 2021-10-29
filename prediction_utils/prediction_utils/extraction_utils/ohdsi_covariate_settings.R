### Collection of preset covariate settings for OHDSI feature extraction library

library(FeatureExtraction)

covariateSettingList <- list()

covariateSettingList[["all_covariates"]] <- createCovariateSettings(
    useDemographicsGender = TRUE,
    useDemographicsAgeGroup = TRUE,
    useDemographicsRace = TRUE,
    useDemographicsEthnicity = TRUE,
    useDemographicsIndexYear = TRUE,
    useDemographicsIndexMonth = TRUE,
    useConditionOccurrenceAnyTimePrior = TRUE,
    useConditionOccurrenceLongTerm = TRUE,
    useConditionOccurrenceShortTerm = TRUE,
    useConditionOccurrenceMediumTerm = TRUE,
    useDrugExposureAnyTimePrior = TRUE,
    useDrugExposureLongTerm = TRUE,
    useDrugExposureShortTerm = TRUE,
    useDrugExposureMediumTerm = TRUE,
    useProcedureOccurrenceAnyTimePrior = TRUE,
    useProcedureOccurrenceLongTerm = TRUE,
    useProcedureOccurrenceShortTerm = TRUE,
    useProcedureOccurrenceMediumTerm = TRUE,
    useDeviceExposureAnyTimePrior = TRUE,
    useDeviceExposureLongTerm = TRUE,
    useDeviceExposureShortTerm = TRUE,
    useDeviceExposureMediumTerm = TRUE,
    useMeasurementAnyTimePrior = TRUE,
    useMeasurementLongTerm = TRUE,
    useMeasurementShortTerm = TRUE,
    useMeasurementMediumTerm = TRUE,
    useMeasurementRangeGroupLongTerm = TRUE,
    useMeasurementRangeGroupShortTerm = TRUE,
    useMeasurementRangeGroupMediumTerm = TRUE,
    useObservationAnyTimePrior = TRUE,
    useObservationLongTerm = TRUE,
    useObservationShortTerm = TRUE,
    useObservationMediumTerm = TRUE,
    useVisitCountLongTerm = TRUE,
    useVisitCountShortTerm = TRUE,
    useVisitCountMediumTerm = TRUE,
    useVisitConceptCountLongTerm = TRUE,
    useVisitConceptCountShortTerm = TRUE,
    useVisitConceptCountMediumTerm = TRUE,
    longTermStartDays = -365,
    mediumTermStartDays = -180,
    shortTermStartDays = -30,
    endDays = -1, # Do not use data on prediction day
    includedCovariateConceptIds = c(),
    addDescendantsToInclude = FALSE,
    excludedCovariateConceptIds = c(),
    addDescendantsToExclude = FALSE,
    includedCovariateIds = c()
)
covariateSettingList[["no_measurement_observation"]] <- createCovariateSettings(
    useDemographicsGender = TRUE,
    useDemographicsAgeGroup = TRUE,
    useDemographicsRace = TRUE,
    useDemographicsEthnicity = TRUE,
    useDemographicsIndexYear = TRUE,
    useDemographicsIndexMonth = TRUE,
    useConditionOccurrenceAnyTimePrior = TRUE,
    useConditionOccurrenceLongTerm = TRUE,
    useConditionOccurrenceShortTerm = TRUE,
    useConditionOccurrenceMediumTerm = TRUE,
    useDrugExposureAnyTimePrior = TRUE,
    useDrugExposureLongTerm = TRUE,
    useDrugExposureShortTerm = TRUE,
    useDrugExposureMediumTerm = TRUE,
    useProcedureOccurrenceAnyTimePrior = TRUE,
    useProcedureOccurrenceLongTerm = TRUE,
    useProcedureOccurrenceShortTerm = TRUE,
    useProcedureOccurrenceMediumTerm = TRUE,
    useDeviceExposureAnyTimePrior = TRUE,
    useDeviceExposureLongTerm = TRUE,
    useDeviceExposureShortTerm = TRUE,
    useDeviceExposureMediumTerm = TRUE,
    useVisitCountLongTerm = TRUE,
    useVisitCountShortTerm = TRUE,
    useVisitCountMediumTerm = TRUE,
    useVisitConceptCountLongTerm = TRUE,
    useVisitConceptCountShortTerm = TRUE,
    useVisitConceptCountMediumTerm = TRUE,
    longTermStartDays = -365,
    mediumTermStartDays = -180,
    shortTermStartDays = -30,
    endDays = -1, # Do not use data on prediction day
    includedCovariateConceptIds = c(),
    addDescendantsToInclude = FALSE,
    excludedCovariateConceptIds = c(),
    addDescendantsToExclude = FALSE,
    includedCovariateIds = c()
)
covariateSettingList[["no_observation"]] <- createCovariateSettings(
    useDemographicsGender = TRUE,
    useDemographicsAgeGroup = TRUE,
    useDemographicsRace = TRUE,
    useDemographicsEthnicity = TRUE,
    useDemographicsIndexYear = TRUE,
    useDemographicsIndexMonth = TRUE,
    useConditionOccurrenceAnyTimePrior = TRUE,
    useConditionOccurrenceLongTerm = TRUE,
    useConditionOccurrenceShortTerm = TRUE,
    useConditionOccurrenceMediumTerm = TRUE,
    useDrugExposureAnyTimePrior = TRUE,
    useDrugExposureLongTerm = TRUE,
    useDrugExposureShortTerm = TRUE,
    useDrugExposureMediumTerm = TRUE,
    useProcedureOccurrenceAnyTimePrior = TRUE,
    useProcedureOccurrenceLongTerm = TRUE,
    useProcedureOccurrenceShortTerm = TRUE,
    useProcedureOccurrenceMediumTerm = TRUE,
    useDeviceExposureAnyTimePrior = TRUE,
    useDeviceExposureLongTerm = TRUE,
    useDeviceExposureShortTerm = TRUE,
    useDeviceExposureMediumTerm = TRUE,
    useMeasurementAnyTimePrior = TRUE,
    useMeasurementLongTerm = TRUE,
    useMeasurementShortTerm = TRUE,
    useMeasurementMediumTerm = TRUE,
    useMeasurementRangeGroupLongTerm = TRUE,
    useMeasurementRangeGroupShortTerm = TRUE,
    useMeasurementRangeGroupMediumTerm = TRUE,
    useVisitCountLongTerm = TRUE,
    useVisitCountShortTerm = TRUE,
    useVisitCountMediumTerm = TRUE,
    useVisitConceptCountLongTerm = TRUE,
    useVisitConceptCountShortTerm = TRUE,
    useVisitConceptCountMediumTerm = TRUE,
    longTermStartDays = -365,
    mediumTermStartDays = -180,
    shortTermStartDays = -30,
    endDays = -1, # Do not use data on prediction day
    includedCovariateConceptIds = c(),
    addDescendantsToInclude = FALSE,
    excludedCovariateConceptIds = c(),
    addDescendantsToExclude = FALSE,
    includedCovariateIds = c()
)
covariateSettingList[["binary_history"]] <- createCovariateSettings(
    useDemographicsGender = TRUE,
    useDemographicsAge = FALSE,
    useDemographicsAgeGroup = TRUE,
    useDemographicsRace = TRUE,
    useDemographicsEthnicity = TRUE,
    useDemographicsIndexYear = TRUE,
    useDemographicsIndexMonth = TRUE,
    useConditionOccurrenceAnyTimePrior = TRUE,
    useConditionGroupEraAnyTimePrior = TRUE,
    useDrugExposureAnyTimePrior = TRUE,
    useDrugGroupEraAnyTimePrior = TRUE,
    useProcedureOccurrenceAnyTimePrior = TRUE,
    useDeviceExposureAnyTimePrior = TRUE,
    useMeasurementAnyTimePrior = TRUE,
    useObservationAnyTimePrior = TRUE,
    longTermStartDays = -365,
    mediumTermStartDays = -180,
    shortTermStartDays = -30,
    endDays = -1, # Do not use data on prediction day
    includedCovariateConceptIds = c(),
    addDescendantsToInclude = FALSE,
    excludedCovariateConceptIds = c(),
    addDescendantsToExclude = FALSE,
    includedCovariateIds = c()
)

covariateSettingList[["demographics"]] <- createCovariateSettings(
    useDemographicsGender = TRUE,
    useDemographicsAge = FALSE,
    useDemographicsAgeGroup = TRUE,
    useDemographicsRace = TRUE,
    useDemographicsEthnicity = TRUE,
    useDemographicsIndexYear = TRUE,
    useDemographicsIndexMonth = TRUE,
    longTermStartDays = -365,
    mediumTermStartDays = -180,
    shortTermStartDays = -30,
    endDays = -1, # Do not use data on prediction day
    includedCovariateConceptIds = c(),
    addDescendantsToInclude = FALSE,
    excludedCovariateConceptIds = c(),
    addDescendantsToExclude = FALSE,
    includedCovariateIds = c()
)

covariateSettingList[["condition_occurrence"]] <- createCovariateSettings(
    useConditionOccurrenceAnyTimePrior = TRUE,
    useConditionOccurrenceLongTerm = TRUE,
    useConditionOccurrenceShortTerm = TRUE,
    useConditionOccurrenceMediumTerm = TRUE,
    longTermStartDays = -365,
    mediumTermStartDays = -180,
    shortTermStartDays = -30,
    endDays = -1, # Do not use data on prediction day
    includedCovariateConceptIds = c(),
    addDescendantsToInclude = FALSE,
    excludedCovariateConceptIds = c(),
    addDescendantsToExclude = FALSE,
    includedCovariateIds = c()
)

covariateSettingList[["condition_group_era"]] <- createCovariateSettings(
    useConditionGroupEraAnyTimePrior = TRUE,
    useConditionGroupEraLongTerm = TRUE,
    useConditionGroupEraMediumTerm = TRUE,
    useConditionGroupEraShortTerm = TRUE,
    longTermStartDays = -365,
    mediumTermStartDays = -180,
    shortTermStartDays = -30,
    endDays = -1, # Do not use data on prediction day
    includedCovariateConceptIds = c(),
    addDescendantsToInclude = FALSE,
    excludedCovariateConceptIds = c(),
    addDescendantsToExclude = FALSE,
    includedCovariateIds = c()
)

covariateSettingList[["drug_exposure"]] <- createCovariateSettings(
    useDrugExposureAnyTimePrior = TRUE,
    useDrugExposureLongTerm = TRUE,
    useDrugExposureMediumTerm = TRUE,
    useDrugExposureShortTerm = TRUE,
    longTermStartDays = -365,
    mediumTermStartDays = -180,
    shortTermStartDays = -30,
    endDays = -1, # Do not use data on prediction day
    includedCovariateConceptIds = c(),
    addDescendantsToInclude = FALSE,
    excludedCovariateConceptIds = c(),
    addDescendantsToExclude = FALSE,
    includedCovariateIds = c()
)

covariateSettingList[["drug_group_era"]] <- createCovariateSettings(
    useDrugGroupEraAnyTimePrior = TRUE,
    useDrugGroupEraLongTerm = TRUE,
    useDrugGroupEraMediumTerm = TRUE,
    useDrugGroupEraShortTerm = TRUE,
    longTermStartDays = -365,
    mediumTermStartDays = -180,
    shortTermStartDays = -30,
    endDays = -1, # Do not use data on prediction day
    includedCovariateConceptIds = c(),
    addDescendantsToInclude = FALSE,
    excludedCovariateConceptIds = c(),
    addDescendantsToExclude = FALSE,
    includedCovariateIds = c()
)

covariateSettingList[["procedure"]] <- createCovariateSettings(
    useProcedureOccurrenceAnyTimePrior = TRUE,
    useProcedureOccurrenceLongTerm = TRUE,
    useProcedureOccurrenceMediumTerm = TRUE,
    useProcedureOccurrenceShortTerm = TRUE,
    longTermStartDays = -365,
    mediumTermStartDays = -180,
    shortTermStartDays = -30,
    endDays = -1, # Do not use data on prediction day
    includedCovariateConceptIds = c(),
    addDescendantsToInclude = FALSE,
    excludedCovariateConceptIds = c(),
    addDescendantsToExclude = FALSE,
    includedCovariateIds = c()
)

covariateSettingList[["device"]] <- createCovariateSettings(
    useDeviceExposureAnyTimePrior = TRUE,
    useDeviceExposureLongTerm = TRUE,
    useDeviceExposureMediumTerm = TRUE,
    useDeviceExposureShortTerm = TRUE,
    longTermStartDays = -365,
    mediumTermStartDays = -180,
    shortTermStartDays = -30,
    endDays = -1, # Do not use data on prediction day
    includedCovariateConceptIds = c(),
    addDescendantsToInclude = FALSE,
    excludedCovariateConceptIds = c(),
    addDescendantsToExclude = FALSE,
    includedCovariateIds = c()
)

covariateSettingList[["measurement"]] <- createCovariateSettings(
    useMeasurementAnyTimePrior = TRUE,
    useMeasurementLongTerm = TRUE,
    useMeasurementMediumTerm = TRUE,
    useMeasurementShortTerm = TRUE,
    longTermStartDays = -365,
    mediumTermStartDays = -180,
    shortTermStartDays = -30,
    endDays = -1, # Do not use data on prediction day
    includedCovariateConceptIds = c(),
    addDescendantsToInclude = FALSE,
    excludedCovariateConceptIds = c(),
    addDescendantsToExclude = FALSE,
    includedCovariateIds = c()
)

covariateSettingList[["measurement_range_group"]] <- createCovariateSettings(
    useMeasurementRangeGroupAnyTimePrior = TRUE,
    useMeasurementRangeGroupLongTerm = TRUE,
    useMeasurementRangeGroupMediumTerm = TRUE,
    useMeasurementRangeGroupShortTerm = TRUE,
    longTermStartDays = -365,
    mediumTermStartDays = -180,
    shortTermStartDays = -30,
    endDays = -1, # Do not use data on prediction day
    includedCovariateConceptIds = c(),
    addDescendantsToInclude = FALSE,
    excludedCovariateConceptIds = c(),
    addDescendantsToExclude = FALSE,
    includedCovariateIds = c()
)

covariateSettingList[["observation"]] <- createCovariateSettings(
    useObservationAnyTimePrior = TRUE,
    useObservationLongTerm = TRUE,
    useObservationMediumTerm = TRUE,
    useObservationShortTerm = TRUE,
    longTermStartDays = -365,
    mediumTermStartDays = -180,
    shortTermStartDays = -30,
    endDays = -1, # Do not use data on prediction day
    includedCovariateConceptIds = c(),
    addDescendantsToInclude = FALSE,
    excludedCovariateConceptIds = c(),
    addDescendantsToExclude = FALSE,
    includedCovariateIds = c()
)
