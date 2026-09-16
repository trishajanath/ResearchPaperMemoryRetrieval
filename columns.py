"""Shared SG-QMS column groupings for the diabetes memory bank.

Single source of truth so baseline_rag.py, sacr.py, and experiments.py cannot
drift out of sync on which fields count as state vs. context vs. experience.
"""

STATE_COLS = [
    "age",
    "gender",
    "race",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

CONTEXT_COLS = [
    "admission_type_id",
    "admission_source_id",
    "time_in_hospital",
    "medical_specialty",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
]

EXPERIENCE_COLS = [
    "insulin",
    "change",
    "diabetesMed",
]

CATEGORICAL_COLS = {
    "age",
    "gender",
    "race",
    "admission_type_id",
    "admission_source_id",
    "medical_specialty",
    "insulin",
    "change",
    "diabetesMed",
}
