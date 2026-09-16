"""SACR - State-Aligned Contextual Retrieval.

Algorithm 1 mapping:
    Current Query P      -> current_patient
    Memory Bank M        -> memory_df
    Active Rules R_active -> active_rules
    Top-k aligned records -> sacr_retrieve return value

This module uses only state, context, experience, and semantic similarity for
retrieval. Outcome and governance columns are displayed after retrieval only.
"""

import numpy as np
import pandas as pd

from baseline_rag import embeddings, memory_df, model, state_neutral_text
from baseline_rag import semantic_retrieve
from columns import CATEGORICAL_COLS, CONTEXT_COLS, EXPERIENCE_COLS, STATE_COLS


# Initial experimental values; they are not claimed to be optimal.
STATE_ALIGNMENT_THRESHOLD = 0.70
LAMBDA_SEMANTIC = 0.60
LAMBDA_CONTEXT = 0.25
LAMBDA_EXPERIENCE = 0.15
CANDIDATE_K = 100

active_rules = {
    "min_state_alignment": STATE_ALIGNMENT_THRESHOLD,
}

current_patient = {
    "age": "[70-80)",
    "gender": "Female",
    "race": "Caucasian",
    "number_outpatient": 2,
    "number_emergency": 1,
    "number_inpatient": 3,
    "number_diagnoses": 8,
    "admission_type_id": 1,
    "admission_source_id": 7,
    "time_in_hospital": 5,
    "medical_specialty": "InternalMedicine",
    "num_lab_procedures": 45,
    "num_procedures": 2,
    "num_medications": 18,
    "insulin": "Steady",
    "change": "Ch",
    "diabetesMed": "Yes",
}

if not np.isclose(
    LAMBDA_SEMANTIC + LAMBDA_CONTEXT + LAMBDA_EXPERIENCE,
    1.0,
):
    raise ValueError("SACR ranking weights must sum to 1")


def _is_missing(value):
    return value is None or bool(pd.isna(value))


def variable_similarity(current_value, memory_value, column, memory_df):
    """Return categorical match or normalized numeric similarity in [0, 1]."""
    if _is_missing(current_value) or _is_missing(memory_value):
        return 0.0

    if column in CATEGORICAL_COLS:
        return float(current_value == memory_value)

    numeric_values = pd.to_numeric(memory_df[column], errors="coerce").dropna()
    column_min = numeric_values.min()
    column_max = numeric_values.max()
    if column_max == column_min:
        return float(float(current_value) == column_min)

    similarity = 1.0 - abs(float(current_value) - float(memory_value)) / (
        column_max - column_min
    )
    return float(np.clip(similarity, 0.0, 1.0))


def calculate_state_alignment(current_patient, memory_row, memory_df):
    """Calculate mean state similarity for the hard eligibility constraint."""
    scores = [
        variable_similarity(
            current_patient[column], memory_row[column], column, memory_df
        )
        for column in STATE_COLS
    ]
    return float(np.mean(scores))


def calculate_context_alignment(current_patient, memory_row, memory_df):
    """Calculate mean context similarity for final ranking."""
    scores = [
        variable_similarity(
            current_patient[column], memory_row[column], column, memory_df
        )
        for column in CONTEXT_COLS
    ]
    return float(np.mean(scores))


def calculate_experience_alignment(current_patient, memory_row):
    """Calculate mean categorical similarity for observed experience."""
    scores = [
        variable_similarity(current_patient[column], memory_row[column], column, memory_df)
        for column in EXPERIENCE_COLS
    ]
    return float(np.mean(scores))


def patient_to_text(current_patient):
    """Use the same state-neutral (context+experience only) text as the baseline."""
    return state_neutral_text(current_patient)


def check_active_rules(current_patient, memory_row, active_rules, memory_bank=None):
    """Apply active-rule eligibility checks.

    State eligibility (``active_rules["min_state_alignment"]``) is always
    enforced -- this is RQ1's original, unchanged behavior. Context
    eligibility (``active_rules.get("min_context_alignment")``) is enforced
    only when that key is present in ``active_rules``; RQ1's own
    ``active_rules`` dict never sets it, so RQ1 behavior is unaffected. This
    optional second gate is what RQ2 uses to jointly require state AND
    context alignment.
    """
    bank = memory_df if memory_bank is None else memory_bank
    state_threshold = active_rules["min_state_alignment"]
    state_alignment = calculate_state_alignment(
        current_patient, memory_row, bank
    )
    if state_alignment < state_threshold:
        return False

    context_threshold = active_rules.get("min_context_alignment")
    if context_threshold is not None:
        context_alignment = calculate_context_alignment(
            current_patient, memory_row, bank
        )
        if context_alignment < context_threshold:
            return False

    return True


def get_semantic_candidates(current_patient, candidate_k=CANDIDATE_K, memory_bank=None):
    """Return (candidate_indices, semantic_scores) for the top semantic
    candidates, ranked purely by cosine similarity on state-neutral text.

    Pulled out of ``sacr_retrieve`` so other code (e.g. RQ2's context-
    threshold analysis) can reuse the exact same candidate-selection path
    without duplicating it. Behavior is unchanged: this is the same
    computation ``sacr_retrieve`` always did inline.
    """
    bank = memory_df if memory_bank is None else memory_bank
    query_embedding = model.encode(
        [patient_to_text(current_patient)],
        normalize_embeddings=True,
    )[0]
    semantic_scores = embeddings @ np.asarray(query_embedding, dtype=np.float32)
    candidate_count = min(candidate_k, len(bank))
    candidate_indices = np.argsort(semantic_scores)[::-1][:candidate_count]
    return candidate_indices, semantic_scores


def sacr_retrieve(
    current_patient, active_rules, top_k=5, candidate_k=CANDIDATE_K,
    memory_bank=None, excluded_memory_ids=None,
):
    """Retrieve top-k records after semantic candidates and hard rule filtering.

    ``memory_bank`` optionally overrides the DataFrame used for state/context
    lookups and the returned rows (used by the state-shuffle experiment).
    Semantic candidate selection always uses the module-level ``embeddings``,
    which are unaffected by a STATE_COLS-only shuffle since they are built
    from state-neutral text.

    ``excluded_memory_ids`` optionally names memory_ids (e.g. quarantined or
    deleted by RQ3's governance layer) to exclude from the final ranked
    output. It is applied strictly AFTER the state/context eligibility check,
    so ``number_passing_*`` diagnostics keep their original RQ1/RQ2 meaning
    (pure alignment, independent of governance); a separate
    ``number_excluded_by_governance`` diagnostic counts this extra filter.
    Default is None (no exclusions), which reproduces RQ1/RQ2 behavior
    exactly.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if candidate_k < top_k:
        raise ValueError("candidate_k must be at least top_k")
    bank = memory_df if memory_bank is None else memory_bank
    excluded = excluded_memory_ids if excluded_memory_ids is not None else frozenset()

    candidate_indices, semantic_scores = get_semantic_candidates(
        current_patient, candidate_k, bank
    )
    candidate_count = len(candidate_indices)

    state_threshold = active_rules["min_state_alignment"]
    context_threshold = active_rules.get("min_context_alignment")
    scored_rows = []
    state_passing_count = 0
    context_passing_count = 0
    state_only_passing_count = 0
    context_only_passing_count = 0
    both_passing_count = 0
    active_rule_passing_count = 0
    excluded_by_governance_count = 0

    for index in candidate_indices:
        memory_row = bank.iloc[index]
        state_alignment = calculate_state_alignment(
            current_patient, memory_row, bank
        )
        context_alignment = calculate_context_alignment(
            current_patient, memory_row, bank
        )
        state_ok = state_alignment >= state_threshold
        context_ok = True if context_threshold is None else context_alignment >= context_threshold

        if state_ok:
            state_passing_count += 1
        if context_threshold is not None:
            if context_ok:
                context_passing_count += 1
            if state_ok and context_ok:
                both_passing_count += 1
            elif state_ok and not context_ok:
                state_only_passing_count += 1
            elif context_ok and not state_ok:
                context_only_passing_count += 1

        if not (state_ok and context_ok):
            continue
        active_rule_passing_count += 1

        if memory_row["memory_id"] in excluded:
            excluded_by_governance_count += 1
            continue

        experience_alignment = calculate_experience_alignment(
            current_patient, memory_row
        )
        final_ranking_score = (
            LAMBDA_SEMANTIC * semantic_scores[index]
            + LAMBDA_CONTEXT * context_alignment
            + LAMBDA_EXPERIENCE * experience_alignment
        )
        scored_rows.append(
            {
                "memory_index": index,
                "semantic_similarity": float(semantic_scores[index]),
                "state_alignment": state_alignment,
                "context_alignment": context_alignment,
                "experience_alignment": experience_alignment,
                "final_ranking_score": float(final_ranking_score),
            }
        )

    scored_rows.sort(key=lambda row: row["final_ranking_score"], reverse=True)
    selected_rows = scored_rows[:top_k]
    selected_indices = [row["memory_index"] for row in selected_rows]
    results = bank.iloc[selected_indices].copy().reset_index(drop=True)

    score_columns = [
        "semantic_similarity",
        "state_alignment",
        "context_alignment",
        "experience_alignment",
        "final_ranking_score",
    ]
    if selected_rows:
        score_values = np.array(
            [[row[column] for column in score_columns] for row in selected_rows],
            dtype=float,
        )
    else:
        score_values = np.empty((0, len(score_columns)), dtype=float)
    for position, column in enumerate(score_columns):
        results.insert(position + 1, column, score_values[:, position])

    results.attrs["number_semantic_candidates"] = candidate_count
    results.attrs["number_passing_state_alignment"] = state_passing_count
    results.attrs["number_passing_active_rules"] = active_rule_passing_count
    results.attrs["context_alignment_threshold"] = context_threshold
    results.attrs["number_passing_context_alignment"] = context_passing_count
    results.attrs["number_passing_state_only"] = state_only_passing_count
    results.attrs["number_passing_context_only"] = context_only_passing_count
    results.attrs["number_passing_both"] = both_passing_count
    results.attrs["number_excluded_by_governance"] = excluded_by_governance_count
    return results


def print_baseline_results(results):
    for rank, (_, result) in enumerate(results.iterrows(), start=1):
        print(f"\nRank: {rank}")
        print(f"Memory ID: {result['memory_id']}")
        print(f"Semantic Similarity: {result['similarity_score']:.6f}")
        print(f"Number of inpatient visits: {result['number_inpatient']}")
        print(f"Number of emergency visits: {result['number_emergency']}")
        print(f"Number of diagnoses: {result['number_diagnoses']}")
        print(f"Insulin: {result['insulin']}")
        print(f"DiabetesMed: {result['diabetesMed']}")
        print(f"Observed readmission outcome: {result['readmitted']}")


def print_sacr_results(results):
    for rank, (_, result) in enumerate(results.iterrows(), start=1):
        print(f"\nRank: {rank}")
        print(f"Memory ID: {result['memory_id']}")
        print(f"Semantic Similarity: {result['semantic_similarity']:.6f}")
        print(f"State Alignment: {result['state_alignment']:.6f}")
        print(f"Context Alignment: {result['context_alignment']:.6f}")
        print(f"Experience Alignment: {result['experience_alignment']:.6f}")
        print(f"Final Ranking Score: {result['final_ranking_score']:.6f}")
        print(f"Number of inpatient visits: {result['number_inpatient']}")
        print(f"Number of emergency visits: {result['number_emergency']}")
        print(f"Number of diagnoses: {result['number_diagnoses']}")
        print(f"Insulin: {result['insulin']}")
        print(f"DiabetesMed: {result['diabetesMed']}")
        print(f"Observed readmission outcome: {result['readmitted']}")


if __name__ == "__main__":
    baseline_results = semantic_retrieve(patient_to_text(current_patient), top_k=5)
    sacr_results = sacr_retrieve(
        current_patient,
        active_rules,
        top_k=5,
        candidate_k=CANDIDATE_K,
    )

    print("\n================ BASELINE SEMANTIC RETRIEVAL ================")
    print_baseline_results(baseline_results)
    print("\n================ SACR RETRIEVAL ================")
    print_sacr_results(sacr_results)
    print(f"\nNumber of semantic candidates: {sacr_results.attrs['number_semantic_candidates']}")
    print(
        "Number passing state alignment: "
        f"{sacr_results.attrs['number_passing_state_alignment']}"
    )
    print(
        "Number passing active rules: "
        f"{sacr_results.attrs['number_passing_active_rules']}"
    )

    baseline_state_scores = [
        calculate_state_alignment(current_patient, row, memory_df)
        for _, row in baseline_results.iterrows()
    ]
    sacr_state_scores = sacr_results["state_alignment"].to_numpy()
    baseline_misaligned_count = sum(
        score < STATE_ALIGNMENT_THRESHOLD for score in baseline_state_scores
    )
    sacr_misaligned_count = sum(
        score < active_rules["min_state_alignment"] for score in sacr_state_scores
    )
    print(
        f"\nAverage baseline top-5 state alignment: "
        f"{np.mean(baseline_state_scores):.6f}"
    )
    print(
        f"Average SACR top-5 state alignment: "
        f"{np.mean(sacr_state_scores):.6f}"
    )
    print(f"Baseline misaligned count: {baseline_misaligned_count}")
    print(f"SACR misaligned count: {sacr_misaligned_count}")
