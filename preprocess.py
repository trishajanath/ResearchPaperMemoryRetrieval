from pathlib import Path

import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo

DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_CACHE_PATH = DATA_DIR / "diabetic_data_raw.pkl"

if RAW_CACHE_PATH.exists():
    df = pd.read_pickle(RAW_CACHE_PATH)
else:
    dataset = fetch_ucirepo(id=296)

    X = dataset.data.features
    y = dataset.data.targets
    df = X.copy()

    # Add the outcome
    df["readmitted"] = y["readmitted"].values

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_pickle(RAW_CACHE_PATH)

print(df.shape)
print(df.head())

#define SG QMS Components

state_cols = [
    "age",
    "gender",
    "race",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses"
]

context_cols = [
    "admission_type_id",
    "admission_source_id",
    "time_in_hospital",
    "medical_specialty",
    "num_lab_procedures",
    "num_procedures",
    "num_medications"
]

experience_cols = [
    "insulin",
    "change",
    "diabetesMed"
]

outcome_col = "readmitted"

#missing values
df["race"] = df["race"].fillna("Unknown")
df["medical_specialty"] = df["medical_specialty"].fillna("Unknown")

#verify the columns
print(df[state_cols].isnull().sum())
print(df[context_cols].isnull().sum())
print(df[experience_cols].isnull().sum())
print(df["readmitted"].value_counts())

feedback_map = {
    "NO": 1.0,
    ">30": 0.0,
    "<30": -1.0
}

df["feedback"] = df["readmitted"].map(feedback_map)
print(df[["readmitted", "feedback"]].drop_duplicates())

memory_df = df[
    state_cols +
    context_cols +
    experience_cols +
    ["readmitted", "feedback"]
].copy()

memory_df["memory_id"] = [
    f"M{i:06d}" for i in range(len(memory_df))
]

print(memory_df.shape)
print(memory_df.head())
print(memory_df.columns.tolist())

# OGMM will later update this score using environmental feedback:
# Q_new = clip(Q_old + alpha * F_env, 0, 1)
memory_df["quality_score"] = 0.5

print(memory_df[["memory_id", "readmitted", "feedback", "quality_score"]].head(10))
print(memory_df["quality_score"].value_counts())
print(memory_df.shape)

memory_df["quarantined"] = False

print(memory_df[[
    "memory_id",
    "readmitted",
    "feedback",
    "quality_score",
    "quarantined"
]].head(10))

print(memory_df["quarantined"].value_counts())

print(memory_df.shape)