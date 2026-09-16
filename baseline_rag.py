"""BASELINE: STANDARD SEMANTIC RETRIEVAL.

This module intentionally uses ordinary semantic similarity only. It does not
apply SG-QMS state/context governance, quality scores, quarantine flags,
feedback, policy rules, or any retrieval-time reranking.
"""

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from preprocess import memory_df


MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 32
DATA_DIR = Path(__file__).resolve().parent / "data"
EMBEDDINGS_PATH = DATA_DIR / "baseline_embeddings.npy"
MEMORY_IDS_PATH = DATA_DIR / "baseline_memory_ids.npy"


def state_neutral_text(row):
    """Convert only CONTEXT_COLS and EXPERIENCE_COLS into a retrieval document.

    STATE_COLS (age, gender, race, prior visit counts, number of diagnoses)
    are deliberately excluded so semantic retrieval cannot see structured
    state through the text channel.
    """
    return (
        f"Admission type: {row['admission_type_id']}. "
        f"Admission source: {row['admission_source_id']}. "
        f"Time in hospital: {row['time_in_hospital']} days. "
        f"Medical specialty: {row['medical_specialty']}. "
        f"Number of lab procedures: {row['num_lab_procedures']}. "
        f"Number of procedures: {row['num_procedures']}. "
        f"Number of medications: {row['num_medications']}. "
        f"Insulin: {row['insulin']}. "
        f"Medication change: {row['change']}. "
        f"Diabetes medication: {row['diabetesMed']}."
    )


def _cache_matches_memory_bank(cached_embeddings, cached_memory_ids):
    """Ensure cached rows map one-to-one and in-order to the current bank."""
    current_memory_ids = memory_df["memory_id"].to_numpy(dtype=str)
    return (
        cached_embeddings.ndim == 2
        and cached_embeddings.shape[0] == len(memory_df)
        and cached_embeddings.shape[1] == 384
        and cached_memory_ids.shape == current_memory_ids.shape
        and np.array_equal(cached_memory_ids.astype(str), current_memory_ids)
    )


model = SentenceTransformer(MODEL_NAME)
memory_texts = memory_df.apply(state_neutral_text, axis=1).tolist()

if EMBEDDINGS_PATH.exists() and MEMORY_IDS_PATH.exists():
    cached_embeddings = np.load(EMBEDDINGS_PATH)
    cached_memory_ids = np.load(MEMORY_IDS_PATH, allow_pickle=False)
    if _cache_matches_memory_bank(cached_embeddings, cached_memory_ids):
        embeddings = cached_embeddings
    else:
        embeddings = model.encode(
            memory_texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        np.save(EMBEDDINGS_PATH, embeddings)
        np.save(MEMORY_IDS_PATH, memory_df["memory_id"].to_numpy(dtype=str))
else:
    embeddings = model.encode(
        memory_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    np.save(MEMORY_IDS_PATH, memory_df["memory_id"].to_numpy(dtype=str))


def semantic_retrieve(query_text, top_k=5, memory_bank=None, excluded_memory_ids=None):
    """Return memories ranked only by cosine similarity to ``query_text``.

    ``memory_bank`` optionally overrides which DataFrame's rows are returned
    for the same embedding-derived ranking. This is used by the state-shuffle
    experiment: row order and ``embeddings`` are unaffected by a STATE_COLS-
    only shuffle (embeddings are built from state-neutral text), so only the
    returned rows need to come from the shuffled bank.

    ``excluded_memory_ids`` optionally names memory_ids to exclude from the
    ranked output (e.g. RQ4's held-out patients, kept out of the retrievable
    memory pool for leakage prevention). Applied by skipping excluded rows
    while walking the full similarity ranking, so ``embeddings``' positional
    alignment is never disturbed. Default None reproduces prior behavior
    exactly.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    bank = memory_df if memory_bank is None else memory_bank
    excluded = excluded_memory_ids if excluded_memory_ids is not None else frozenset()

    query_embedding = model.encode(
        [query_text],
        normalize_embeddings=True,
    )[0]
    query_embedding = np.asarray(query_embedding, dtype=np.float32)
    similarity_scores = embeddings @ query_embedding
    sorted_indices = np.argsort(similarity_scores)[::-1]

    if excluded:
        memory_ids = bank["memory_id"].to_numpy()
        selected = []
        for index in sorted_indices:
            if memory_ids[index] in excluded:
                continue
            selected.append(index)
            if len(selected) == top_k:
                break
        top_indices = np.array(selected, dtype=int)
    else:
        top_indices = sorted_indices[:top_k]

    results = bank.iloc[top_indices].copy()
    results.insert(1, "similarity_score", similarity_scores[top_indices])
    return results


TEST_QUERY = (
    "A female patient in the 70-80 age group with several previous inpatient "
    "visits and multiple diagnoses is admitted for a hospital stay of several "
    "days. The patient is receiving insulin and multiple diabetes medications."
)


if __name__ == "__main__":
    print("Number of memories:", len(memory_df))
    print("Embedding shape:", embeddings.shape)
    print("\nTest query:")
    print(TEST_QUERY)
    print("\nTop 5 semantic retrieval results:")

    retrieved = semantic_retrieve(TEST_QUERY, top_k=5)
    for rank, (_, result) in enumerate(retrieved.iterrows(), start=1):
        print(f"\nRank: {rank}")
        print(f"Memory ID: {result['memory_id']}")
        print(f"Semantic similarity: {result['similarity_score']:.6f}")
        print(f"Age: {result['age']}")
        print(f"Gender: {result['gender']}")
        print(f"Race: {result['race']}")
        print(f"Number of inpatient visits: {result['number_inpatient']}")
        print(f"Number of emergency visits: {result['number_emergency']}")
        print(f"Number of diagnoses: {result['number_diagnoses']}")
        print(f"Medical specialty: {result['medical_specialty']}")
        print(f"Insulin: {result['insulin']}")
        print(f"DiabetesMed: {result['diabetesMed']}")
        print(f"Observed readmission outcome: {result['readmitted']}")
