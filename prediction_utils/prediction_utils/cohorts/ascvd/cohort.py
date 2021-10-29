from prediction_utils.cohorts.cohort import BQCohort


class ASCVDCohort(BQCohort):
    """
    Cohort that defines binary ASCVD labels
    """

    def get_defaults(self):
        return {
            **super().get_defaults(),
            **{
                "years_history_required": 1,
                "max_observation_period_end_date": "2021-12-31",
                "min_age_in_years": 18.0,
                "has_birth_datetime": True,
            },
        }

    def get_config_dict(self, **kwargs):
        config_dict = super().get_config_dict(**kwargs)
        if config_dict["has_birth_datetime"]:
            config_dict[
                "age_in_years"
            ] = "CAST(DATE_DIFF(CAST(visit_start_date AS DATE), CAST(birth_datetime AS DATE), DAY) AS FLOAT64) / 365.25"
        else:
            config_dict[
                "age_in_years"
            ] = "EXTRACT(YEAR FROM visit_start_date) - year_of_birth"
        return config_dict

    def get_base_query(self, format_query=True):
        query = """
            WITH visits AS (
                SELECT t1.person_id, visit_occurrence_id,
                    birth_datetime, visit_start_date, 
                    {age_in_years} as age_in_years,
                FROM {dataset_project}.{dataset}.visit_occurrence t1
                INNER JOIN {dataset_project}.{dataset}.person as t2
                    ON t1.person_id = t2.person_id
                WHERE 
                    visit_concept_id in (9202, 581477)
            ), 
            observation_periods AS (
                SELECT t1.person_id, 
                    MIN(observation_period_start_date) as observation_period_start_date, 
                    MAX(observation_period_end_date) as observation_period_end_date,
                FROM visits t1
                INNER JOIN {dataset_project}.{dataset}.observation_period as t2
                    ON t1.person_id = t2.person_id
                GROUP BY t1.person_id
            ),
            relative_endpoints AS (
                SELECT *,
                    CAST(DATE_DIFF(CAST(visit_start_date AS DATE), CAST(observation_period_start_date AS DATE), DAY) AS FLOAT64) / 365.25 as years_since_start,
                    CAST(DATE_DIFF(CAST(observation_period_end_date AS DATE), CAST(visit_start_date AS DATE), DAY) AS FLOAT64) / 365.25 as years_until_end
                FROM visits
                INNER JOIN observation_periods USING (person_id)
            )
            SELECT *
            FROM relative_endpoints
            WHERE years_since_start >= {years_history_required}
                AND observation_period_end_date <= "{max_observation_period_end_date}"
                AND years_until_end >= 0.0
                AND age_in_years >= {min_age_in_years}
                AND visit_start_date <= "{max_index_date}"
            """.format_map(
            self.config_dict
        )
        if not format_query:
            return query
        else:
            return query.format_map(self.config_dict)

    def get_transform_query(self, format_query=True):
        query = """
            WITH base_query as (
                {base_query}
            ),
            cvd_concepts as (
                {cvd_concepts}
            ),
            mi_stroke_concepts as (
                {mi_stroke_concepts}
            ),
            chd_concepts as (
                {chd_concepts}
            ),
            statin_concepts as (
                {statin_concepts}
            ),
            cvd_history AS (
                SELECT DISTINCT t3.person_id, t3.visit_occurrence_id, t3.visit_start_date, 1 as has_cvd_history
                FROM {dataset_project}.{dataset}.condition_occurrence t1
                INNER JOIN cvd_concepts as t2 ON
                    t1.condition_concept_id = t2.concept_id
                INNER JOIN base_query as t3 ON
                    t1.person_id = t3.person_id
                WHERE condition_start_date < visit_start_date
            ),
            statin_history AS (
                SELECT DISTINCT t3.person_id, t3.visit_occurrence_id, t3.visit_start_date, 1 as has_statin_history
                FROM {dataset_project}.{dataset}.drug_exposure t1
                INNER JOIN statin_concepts as t2 ON
                    t1.drug_concept_id = t2.concept_id
                INNER JOIN base_query as t3 ON
                    t1.person_id = t3.person_id
                WHERE drug_exposure_start_date < visit_start_date
            ),
            mi_stroke_outcomes AS (
                SELECT DISTINCT t3.person_id, t1.condition_start_date as outcome_date
                FROM {dataset_project}.{dataset}.condition_occurrence t1
                INNER JOIN mi_stroke_concepts as t2 ON
                    t1.condition_concept_id = t2.concept_id
                INNER JOIN base_query as t3 ON
                    t1.person_id = t3.person_id
                WHERE t1.condition_start_date > t3.visit_start_date
            ), 
            chd_death_outcomes AS (
                SELECT DISTINCT t1.person_id, t1.condition_start_date as outcome_date
                FROM {dataset_project}.{dataset}.condition_occurrence t1
                INNER JOIN chd_concepts as t2 ON
                    t1.condition_concept_id = t2.concept_id
                INNER JOIN base_query as t3 ON
                    t1.person_id = t3.person_id
                INNER JOIN {dataset_project}.{dataset}.death as t4 ON
                    t1.person_id = t4.person_id
                WHERE t1.condition_start_date > t3.visit_start_date 
                    AND DATE_DIFF(t4.death_date, t1.condition_start_date, DAY) > 0
                    AND DATE_DIFF(t4.death_date, t1.condition_start_date, DAY) <= 365
            ), 
            all_outcome_times AS (
                SELECT * FROM mi_stroke_outcomes
                UNION ALL
                SELECT * FROM chd_death_outcomes
            ), 
            min_outcome_times_per_index_date AS (
                SELECT person_id, visit_occurrence_id, visit_start_date, MIN(outcome_date) as outcome_date
                FROM base_query
                INNER JOIN all_outcome_times USING (person_id)
                WHERE outcome_date > visit_start_date
                GROUP BY person_id, visit_occurrence_id, visit_start_date
            ),
            ascvd_outcomes AS (
                SELECT *, DATE_DIFF(outcome_date, visit_start_date, DAY) as days_until_outcome
                FROM min_outcome_times_per_index_date
            ),
            statin_censoring_times AS (
                SELECT DISTINCT t3.person_id, t3.visit_occurrence_id, t3.visit_start_date, MIN(drug_exposure_start_date) as statin_censoring_date
                FROM {dataset_project}.{dataset}.drug_exposure t1
                INNER JOIN statin_concepts as t2 ON
                    t1.drug_concept_id = t2.concept_id
                INNER JOIN base_query as t3 ON
                    t1.person_id = t3.person_id
                WHERE drug_exposure_start_date >= visit_start_date
                GROUP BY t3.person_id, t3.visit_occurrence_id, t3.visit_start_date
            ),
            death_times AS (
                SELECT t1.person_id, t1.visit_occurrence_id, t1.visit_start_date, death_date
                FROM base_query t1
                INNER JOIN {dataset_project}.{dataset}.death as t2 ON
                    t1.person_id = t2.person_id
            ),
            cohort_with_censoring_1 AS (
                SELECT *, 
                    LEAST(
                        IFNULL(death_date, observation_period_end_date),
                        IFNULL(statin_censoring_date, observation_period_end_date),
                        observation_period_end_date
                    ) as censoring_date, --includes statin initiation as a censoring event
                    LEAST(
                        IFNULL(death_date, observation_period_end_date),
                        observation_period_end_date
                    ) as endpoint_censoring_date -- not considering statin time as a censoring event
                FROM base_query
                LEFT JOIN statin_censoring_times USING (person_id, visit_occurrence_id, visit_start_date)
                LEFT JOIN death_times USING (person_id, visit_occurrence_id, visit_start_date)
            ),
            cohort_with_censoring_2 AS (
                SELECT *, 
                DATE_DIFF(censoring_date, visit_start_date, DAY) as days_until_censoring,
                DATE_DIFF(endpoint_censoring_date, visit_start_date, DAY) as days_until_endpoint_censoring
                FROM cohort_with_censoring_1
            ),
            cohort_with_censoring_3 AS (
                SELECT *,
                        CAST(
                            (days_until_censoring < {event_followup_days}) AND
                            (days_until_censoring < IFNULL(days_until_censoring, {event_followup_days}))
                            AS INT64
                        ) as censored_binary,
                        CAST(
                            (days_until_endpoint_censoring < {event_followup_days}) AND
                            (days_until_endpoint_censoring < IFNULL(days_until_endpoint_censoring, {event_followup_days}))
                            AS INT64
                        ) as endpoint_censored_binary
                FROM cohort_with_censoring_2
            ),
            cohort_with_history_and_outcomes AS (
                SELECT * EXCEPT(has_statin_history, has_cvd_history),
                        IFNULL(has_statin_history, 0) as has_statin_history,
                        IFNULL(has_cvd_history, 0) as has_cvd_history,
                        CASE
                            WHEN days_until_outcome is NULL THEN days_until_censoring
                            WHEN days_until_outcome <= days_until_censoring THEN days_until_outcome
                            WHEN days_until_outcome > days_until_censoring THEN days_until_censoring
                        END AS days_until_event,
                        CASE
                            WHEN days_until_outcome is NULL THEN 0
                            WHEN days_until_outcome <= days_until_censoring THEN 1
                            WHEN days_until_outcome > days_until_censoring THEN 0
                        END AS event_indicator,
                        CASE
                            WHEN days_until_outcome is NULL THEN days_until_endpoint_censoring
                            WHEN days_until_outcome <= days_until_endpoint_censoring THEN days_until_outcome
                            WHEN days_until_outcome > days_until_endpoint_censoring THEN days_until_endpoint_censoring
                        END AS days_until_endpoint_censored_event,
                        CASE
                            WHEN days_until_outcome is NULL THEN 0
                            WHEN days_until_outcome <= days_until_endpoint_censoring THEN 1
                            WHEN days_until_outcome > days_until_endpoint_censoring THEN 0
                        END AS endpoint_censored_event_indicator
                FROM cohort_with_censoring_3
                LEFT JOIN statin_history USING (person_id, visit_occurrence_id, visit_start_date)
                LEFT JOIN cvd_history USING (person_id, visit_occurrence_id, visit_start_date)
                LEFT JOIN ascvd_outcomes USING (person_id, visit_occurrence_id, visit_start_date)
            ),
            cohort_with_binary_outcomes AS (
                SELECT * EXCEPT(visit_start_date), 
                    visit_start_date as index_date,
                    CASE
                        WHEN 
                            (days_until_outcome is NOT NULL) AND
                            (days_until_outcome BETWEEN 0 AND {event_followup_days}) AND 
                            (days_until_outcome < days_until_censoring)
                            THEN 1
                        WHEN 
                            (IFNULL(days_until_outcome, days_until_censoring) > {event_followup_days}) AND
                            (days_until_censoring > {event_followup_days}) 
                            THEN 0
                        WHEN
                            censored_binary = 1
                            THEN NULL
                    END AS ascvd_binary,
                    CASE
                        WHEN 
                            (days_until_outcome is NOT NULL) AND
                            (days_until_outcome BETWEEN 0 AND {event_followup_days}) AND 
                            (days_until_outcome < days_until_endpoint_censoring)
                            THEN 1
                        WHEN 
                            (IFNULL(days_until_outcome, days_until_endpoint_censoring) > {event_followup_days}) AND
                            (days_until_endpoint_censoring > {event_followup_days})
                            THEN 0
                        WHEN
                            endpoint_censored_binary = 1
                            THEN NULL
                    END AS ascvd_binary_endpoint_censored
                FROM cohort_with_history_and_outcomes
            )
            SELECT *,
            FROM cohort_with_binary_outcomes
        """.format_map(
            {
                **self.config_dict,
                **{
                    "base_query": self.get_base_query(),
                    "cvd_concepts": self.get_cvd_concepts(),
                    "mi_stroke_concepts": self.get_mi_stroke_concepts(),
                    "chd_concepts": self.get_chd_concepts(),
                    "statin_concepts": self.get_statin_concepts(),
                },
            }
        )
        return query

    def get_transform_query_sampled(self, source_table=None):
        return """
            SELECT * EXCEPT (rnd, pos), 
            FARM_FINGERPRINT(GENERATE_UUID()) as prediction_id
            FROM (
                SELECT *, ROW_NUMBER() OVER(PARTITION BY person_id ORDER BY rnd) AS pos
                FROM (
                    SELECT 
                        *,
                        FARM_FINGERPRINT(CONCAT(CAST(person_id AS STRING), CAST(visit_occurrence_id AS STRING))) as rnd
                    FROM ({base_query})
                )
            )
            WHERE pos = 1
        """.format(
            base_query=self.get_transform_query()
            if source_table is None
            else "SELECT * FROM {}".format(source_table)
        )

    def get_concept_rollup_query(self):
        return """
            WITH source_concepts AS (
                {base_query}
            ), 
            standard_concepts as (
                SELECT DISTINCT concept_id_2 as concept_id
                FROM source_concepts t1
                INNER JOIN {dataset_project}.{dataset}.concept_relationship as t2 
                    ON t1.concept_id = t2.concept_id_1
                WHERE relationship_id = 'Maps to'
            ),
            concept_descendants as (
                SELECT descendant_concept_id as concept_id
                FROM standard_concepts t1
                INNER JOIN {dataset_project}.{dataset}.concept_ancestor as t2
                    ON t1.concept_id = t2.ancestor_concept_id
            )
            SELECT * FROM concept_descendants
                UNION DISTINCT 
            SELECT * FROM standard_concepts
        """

    def get_source_concepts_cvd(self, format_query=True):
        query = """
            SELECT concept_id, concept_name, concept_code 
            FROM {dataset_project}.{dataset}.concept
            WHERE vocabulary_id = 'ICD9CM'
                AND (
                    concept_code LIKE '410%'
                    OR concept_code LIKE '411%' 
                    OR concept_code LIKE '413%' 
                    OR concept_code LIKE '414%' 
                    OR concept_code LIKE '430%' 
                    OR concept_code LIKE '431%' 
                    OR concept_code LIKE '432%' 
                    OR concept_code LIKE '433%'
                    OR concept_code LIKE '434%' 
                    OR concept_code LIKE '436%' 
                    OR concept_code = '427.31' 
                    OR concept_code LIKE '428%'
                )
        """
        if not format_query:
            return query
        else:
            return query.format_map(self.config_dict)

    def get_cvd_concepts(self):
        return self.get_concept_rollup_query().format_map(
            {
                **self.config_dict,
                **{"base_query": self.get_source_concepts_cvd(format_query=True)},
            }
        )

    def get_source_concepts_mi_stroke(self, format_query=True):
        query = """
            SELECT concept_id 
            FROM {dataset_project}.{dataset}.concept
            WHERE vocabulary_id = 'ICD9CM'
            AND (
                    ( -- MI
                        concept_code LIKE '410%'
                    )
                OR 
                    ( -- Stroke
                        concept_code LIKE "430%" OR 
                        concept_code LIKE "431%" OR 
                        concept_code LIKE "432%" OR
                        (concept_code LIKE "433%" AND concept_code NOT LIKE "433._0") OR 
                        (concept_code LIKE "434%" AND concept_code NOT LIKE "434._0") OR 
                        concept_code LIKE "436%"
                )
            )
        """
        if not format_query:
            return query
        else:
            return query.format_map(self.config_dict)

    def get_mi_stroke_concepts(self):
        return self.get_concept_rollup_query().format_map(
            {
                **self.config_dict,
                **{"base_query": self.get_source_concepts_mi_stroke(format_query=True)},
            }
        )

    def get_source_concepts_chd(self, format_query=True):
        query = """
            SELECT concept_id, concept_name, concept_code 
            FROM {dataset_project}.{dataset}.concept
            WHERE vocabulary_id = 'ICD9CM'
                AND (
                    concept_code LIKE '411%' 
                    OR concept_code LIKE '413%' 
                    OR concept_code LIKE '414%' 
                )
        """
        if not format_query:
            return query
        else:
            return query.format_map(self.config_dict)

    def get_chd_concepts(self):
        return self.get_concept_rollup_query().format_map(
            {
                **self.config_dict,
                **{"base_query": self.get_source_concepts_chd(format_query=True)},
            }
        )

    def get_source_concepts_statin(self, format_query=True):
        query = """
            SELECT concept_id 
            FROM {dataset_project}.{dataset}.concept
            WHERE vocabulary_id = 'ATC'
                AND (
                    concept_code IN (
                        'C10AA01',
                        'C10AA07',
                        'C10AA02',
                        'C10AA08',
                        'C10AA04',
                        'C10AA03',
                        'C10BA05',
                        'C10AA06',
                        'C10BX03',
                        'C10BA01',
                        'C10BA02',
                        'C10BA04'
                    )
                )
        """
        if not format_query:
            return query
        else:
            return query.format_map(self.config_dict)

    def get_statin_concepts(self):
        return self.get_concept_rollup_query().format_map(
            {
                **self.config_dict,
                **{"base_query": self.get_source_concepts_statin(format_query=True)},
            }
        )


class ASCVDDemographicsCohort(BQCohort):
    """
    Cohort class that defines the demographic variables
    """

    def get_base_query(self, format_query=True):

        query = "{rs_dataset_project}.{rs_dataset}.{cohort_name}"
        if not format_query:
            return query
        else:
            return query.format_map(self.config_dict)

    def get_transform_query(self, format_query=True):
        query = """
            WITH age_labels AS (
                {age_query}
            ),
            demographics AS (
                {demographics_query}
            )
            SELECT *
            FROM {base_query}
            LEFT JOIN age_labels USING (person_id, age_in_years)
            LEFT JOIN demographics USING (person_id)
            WHERE gender_concept_name != "No matching concept"
        """

        if not format_query:
            return query
        else:
            return query.format(
                base_query=self.get_base_query(), **self.get_label_query_dict()
            )

    def get_label_query_dict(self):
        query_dict = {
            "age_query": self.get_age_query(),
            "demographics_query": self.get_demographics_query(),
        }
        base_query = self.get_base_query()
        return {
            key: value.format_map({**{"base_query": base_query}, **self.config_dict})
            for key, value in query_dict.items()
        }

    def get_age_query(self):
        return """
            WITH temp AS (
                SELECT t1.person_id, age_in_years,
                FROM {base_query} t1
                INNER JOIN {dataset_project}.{dataset}.person t2
                ON t1.person_id = t2.person_id
            )
            SELECT *,
            CASE 
                WHEN age_in_years >= 18.0 and age_in_years < 30.0 THEN '[18-30)'
                WHEN age_in_years >= 30.0 and age_in_years < 45.0 THEN '[30-45)'
                WHEN age_in_years >= 45.0 and age_in_years < 55.0 THEN '[45-55)'
                WHEN age_in_years >= 55.0 and age_in_years < 65.0 THEN '[55-65)'
                WHEN age_in_years >= 65.0 and age_in_years < 75.0 THEN '[65-75)'
                WHEN age_in_years >= 75.0 and age_in_years < 91.0 THEN '[75-91)'
                ELSE '<18'
            END as age_group
            FROM temp
        """

    def get_demographics_query(self):
        return """
            SELECT DISTINCT t1.person_id,
                CASE
                    WHEN t5.concept_name = "Hispanic or Latino" THEN "Hispanic or Latino"
                    WHEN t4.concept_name = "Other, Hispanic" THEN "Hispanic or Latino"
                    WHEN t4.concept_name in (
                        "Native Hawaiian or Other Pacific Islander",
                        "American Indian or Alaska Native",
                        "No matching concept",
                        "Other, non-Hispanic",
                        "Race and Ethnicity Unknown",
                        "Unknown",
                        "Declines to State", 
                        "Patient Refused"
                    ) THEN "Other"
                    ELSE t4.concept_name
                END as race_eth,
                t3.concept_name as gender_concept_name
                FROM {base_query} t1
                INNER JOIN {dataset_project}.{dataset}.person AS t2
                    ON t1.person_id = t2.person_id
                INNER JOIN {dataset_project}.{dataset}.concept as t3
                    ON t2.gender_concept_id = t3.concept_id
                INNER JOIN {dataset_project}.{dataset}.concept as t4
                    ON t2.race_concept_id = t4.concept_id
                INNER JOIN {dataset_project}.{dataset}.concept as t5
                    ON t2.ethnicity_concept_id = t5.concept_id
        """


def get_transform_query_sampled(base_query=None, source_table=None):
    if base_query is None and source_table is None:
        raise ValueError("Both base_query and source_table can not be None")

    if base_query is not None and source_table is not None:
        raise ValueError("Both base_query and source_table can not be not None")

    return """
        SELECT * EXCEPT (rnd, pos), 
        FARM_FINGERPRINT(GENERATE_UUID()) as prediction_id
        FROM (
            SELECT *, ROW_NUMBER() OVER(PARTITION BY person_id ORDER BY rnd) AS pos
            FROM (
                SELECT 
                    *,
                    FARM_FINGERPRINT(CONCAT(CAST(person_id AS STRING), CAST(visit_occurrence_id AS STRING))) as rnd
                FROM ({base_query})
            )
        )
        WHERE pos = 1
    """.format(
        base_query=base_query
        if source_table is None
        else "SELECT * FROM {}".format(source_table)
    )
