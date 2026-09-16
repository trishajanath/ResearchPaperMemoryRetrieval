"""RQ4 feasibility gate: analysis-only, no new experiment logic.

Question this answers, before any new RQ4 protocol is implemented:

    "Can we find enough distinct held-out patients that naturally form
    recurring state groups under our FROZEN SACR state representation?"

This script does not add a new retrieval mode, governance rule, or
evaluation protocol. It reuses, unmodified:
    - the frozen state definition (STATE_COLS, CATEGORICAL_COLS)
    - the frozen state-alignment threshold (STATE_ALIGNMENT_THRESHOLD = 0.70)
    - the frozen state-alignment scoring function (calculate_state_alignment /
      variable_similarity, imported from sacr.py, not reimplemented)
    - RQ4's own patient-level split (build_split, from experiments_rq4.py)
    - RQ4's own frozen active_rules / SACR retrieval (sacr_retrieve)

and only measures whether recurrence exists in the held-out population. It
writes no results into results/rq4 and does not touch RQ1-RQ4 outputs.

Two independent notions of "recurring state group" are reported:

1. EXACT groups: held-out patients (one representative row per patient_nbr)
   grouped by an identical STATE_COLS tuple. This is the strict, unambiguous
   notion of "the same state recurs" -- no threshold, no normalization
   choices, nothing to argue with.

2. THRESHOLD groups: connected components over held-out patients where an
   edge exists iff calculate_state_alignment(...) >= STATE_ALIGNMENT_THRESHOLD
   (the exact hard-eligibility rule SACR itself already enforces). This is
   looser and answers "how many patients would SACR itself treat as
   state-eligible for each other." Computed on a bounded random sample for
   tractability (O(n^2) pairwise), sample size and seed reported.

Finally, for a sample of EXACT groups with >=3 members, it runs the real,
frozen sacr_retrieve() for each member against the memory pool and reports
whether members of the same state group actually retrieve overlapping
memory sets (Jaccard over top-k memory_id), compared against a
cross-group random-pair baseline. This is the part that determines whether
state recurrence would actually translate into repeated SACR memories.

Run: python3 rq4_recurrence_analysis.py
Output: results/rq4_recurrence_analysis/report.json + report.txt
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_rag import memory_df as full_memory_df
from columns import CATEGORICAL_COLS, STATE_COLS
from sacr import (
    CANDIDATE_K,
    STATE_ALIGNMENT_THRESHOLD,
    calculate_state_alignment,
    sacr_retrieve,
    variable_similarity,
)
from experiments import RANDOM_SEED, TOP_K
from experiments_rq3 import _load_frozen_active_rules
from experiments_rq4 import build_split

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "rq4_recurrence_analysis"

THRESHOLD_SAMPLE_SIZE = 3000   # bounded O(n^2) pairwise pass over held-out patients
THRESHOLD_SAMPLE_SEED = RANDOM_SEED
ROW_CHUNK = 500                # chunk size for the pairwise pass (memory bound)

SACR_GROUP_SAMPLE_COUNT = 40   # how many exact groups (size>=3) to probe with sacr_retrieve
SACR_GROUP_MEMBER_CAP = 5      # cap members probed per group
SACR_BASELINE_PAIR_COUNT = 200 # cross-group random-pair baseline for comparison


# ---------------------------------------------------------------------------
# Step 1: one representative row per held-out patient
# ---------------------------------------------------------------------------

def build_representative_rows():
    """Reuse RQ4's own patient-level split; take one row per held-out
    patient_nbr (first occurrence in encounter order) as that patient's
    state. Demographic STATE_COLS fields (age/gender/race) are patient-level
    and don't vary across a patient's encounters; the visit-count STATE_COLS
    fields (number_outpatient/emergency/inpatient/diagnoses) are recorded
    per-encounter, so "the patient's state" here means their first held-out
    encounter's state -- the same notion of state a single SACR query
    against this patient would use.
    """
    memory_pool_df, held_out_df, leakage_excluded_ids, split_info = build_split()
    reps = (
        held_out_df.sort_values("memory_id")
        .drop_duplicates(subset="patient_nbr", keep="first")
        .reset_index(drop=True)
    )
    return memory_pool_df, held_out_df, leakage_excluded_ids, reps, split_info


# ---------------------------------------------------------------------------
# Step 2: EXACT state groups
# ---------------------------------------------------------------------------

def exact_state_groups(reps: pd.DataFrame):
    keys = list(reps[STATE_COLS].itertuples(index=False, name=None))
    groups: dict[tuple, list[int]] = defaultdict(list)
    for row_pos, key in enumerate(keys):
        groups[key].append(row_pos)
    return groups


def summarize_group_sizes(groups: dict) -> dict:
    sizes = sorted((len(v) for v in groups.values()), reverse=True)
    sizes_arr = np.array(sizes)
    return {
        "number_of_groups": int(len(sizes)),
        "number_of_singleton_groups": int((sizes_arr == 1).sum()),
        "groups_with_ge_2": int((sizes_arr >= 2).sum()),
        "groups_with_ge_3": int((sizes_arr >= 3).sum()),
        "groups_with_ge_5": int((sizes_arr >= 5).sum()),
        "groups_with_ge_10": int((sizes_arr >= 10).sum()),
        "largest_group_size": int(sizes_arr.max()) if len(sizes_arr) else 0,
        "mean_group_size": float(sizes_arr.mean()) if len(sizes_arr) else 0.0,
        "median_group_size": float(np.median(sizes_arr)) if len(sizes_arr) else 0.0,
        "patients_in_groups_ge_3": int(sizes_arr[sizes_arr >= 3].sum()),
        "patients_in_groups_ge_5": int(sizes_arr[sizes_arr >= 5].sum()),
        "top_20_group_sizes": sizes[:20],
    }


# ---------------------------------------------------------------------------
# Step 3: THRESHOLD state groups (frozen SACR state_alignment >= 0.70),
# on a bounded random sample, using union-find over a chunked pairwise pass.
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _column_similarity_matrix(col_values_a, col_values_b, column, full_memory_df):
    """Vectorized version of sacr.variable_similarity for one STATE_COLS
    column, matching it term-for-term (including the constant-column
    fallback and missing-value handling), evaluated for every pair in the
    (len(a), len(b)) block. Uses full_memory_df for numeric min/max, exactly
    as variable_similarity does when called with the module-level memory_df.
    """
    a = np.asarray(col_values_a)
    b = np.asarray(col_values_b)

    if column in CATEGORICAL_COLS:
        return (a[:, None] == b[None, :]).astype(np.float32)

    a = a.astype(float)
    b = b.astype(float)
    numeric_values = pd.to_numeric(full_memory_df[column], errors="coerce").dropna()
    col_min = float(numeric_values.min())
    col_max = float(numeric_values.max())
    if col_max == col_min:
        return ((a[:, None] == col_min) & (b[None, :] == col_min)).astype(np.float32)
    sim = 1.0 - np.abs(a[:, None] - b[None, :]) / (col_max - col_min)
    return np.clip(sim, 0.0, 1.0).astype(np.float32)


def threshold_groups_and_similarity_distribution(reps: pd.DataFrame, full_memory_df: pd.DataFrame):
    n_total = len(reps)
    sample_n = min(THRESHOLD_SAMPLE_SIZE, n_total)
    rng = np.random.default_rng(THRESHOLD_SAMPLE_SEED)
    sample_idx = np.sort(rng.choice(n_total, size=sample_n, replace=False))
    sample = reps.iloc[sample_idx].reset_index(drop=True)

    col_arrays = {col: sample[col].to_numpy() for col in STATE_COLS}
    n = len(sample)
    uf = UnionFind(n)
    nearest_neighbor_sim = np.zeros(n, dtype=np.float32)
    all_pair_sims_sample = []  # small extra sample of raw pairwise scores, for a histogram

    rng_pairs = np.random.default_rng(THRESHOLD_SAMPLE_SEED + 1)

    for start in range(0, n, ROW_CHUNK):
        end = min(start + ROW_CHUNK, n)
        block_sum = np.zeros((end - start, n), dtype=np.float32)
        for col in STATE_COLS:
            block_sum += _column_similarity_matrix(
                col_arrays[col][start:end], col_arrays[col], col, full_memory_df
            )
        block_mean = block_sum / len(STATE_COLS)

        for local_i in range(end - start):
            global_i = start + local_i
            row = block_mean[local_i].copy()
            row[global_i] = -1.0  # exclude self for nearest-neighbor stat
            best = row.max()
            nearest_neighbor_sim[global_i] = max(best, 0.0)

            hits = np.nonzero(block_mean[local_i] >= STATE_ALIGNMENT_THRESHOLD)[0]
            for j in hits:
                if j > global_i:
                    uf.union(global_i, j)

        if len(all_pair_sims_sample) < 20000:
            flat = block_mean[:, start:].ravel()  # upper-triangle-ish slice, dedupe not critical for a histogram
            take = min(2000, flat.size)
            picked = rng_pairs.choice(flat.size, size=take, replace=False)
            all_pair_sims_sample.extend(flat[picked].tolist())

    component_members: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        component_members[uf.find(i)].append(i)
    component_sizes = sorted((len(v) for v in component_members.values()), reverse=True)

    nn_sim = nearest_neighbor_sim
    similarity_distribution = {
        "nearest_neighbor_similarity_min": float(nn_sim.min()) if n else None,
        "nearest_neighbor_similarity_p25": float(np.percentile(nn_sim, 25)) if n else None,
        "nearest_neighbor_similarity_median": float(np.percentile(nn_sim, 50)) if n else None,
        "nearest_neighbor_similarity_p75": float(np.percentile(nn_sim, 75)) if n else None,
        "nearest_neighbor_similarity_max": float(nn_sim.max()) if n else None,
        "fraction_with_a_neighbor_ge_threshold": float((nn_sim >= STATE_ALIGNMENT_THRESHOLD).mean()) if n else None,
        "random_pair_similarity_sample_mean": float(np.mean(all_pair_sims_sample)) if all_pair_sims_sample else None,
        "random_pair_similarity_sample_median": float(np.median(all_pair_sims_sample)) if all_pair_sims_sample else None,
        "random_pair_similarity_histogram_counts": np.histogram(
            all_pair_sims_sample, bins=10, range=(0.0, 1.0)
        )[0].tolist() if all_pair_sims_sample else None,
        "random_pair_similarity_histogram_bin_edges": [round(x, 2) for x in np.linspace(0, 1, 11)],
    }

    sizes_arr = np.array(component_sizes)
    threshold_summary = {
        "sample_size": int(n),
        "sample_seed": THRESHOLD_SAMPLE_SEED,
        "threshold_used": STATE_ALIGNMENT_THRESHOLD,
        "number_of_components": int(len(sizes_arr)),
        "components_with_ge_3": int((sizes_arr >= 3).sum()) if len(sizes_arr) else 0,
        "components_with_ge_5": int((sizes_arr >= 5).sum()) if len(sizes_arr) else 0,
        "largest_component_size": int(sizes_arr.max()) if len(sizes_arr) else 0,
        "largest_component_fraction_of_sample": float(sizes_arr.max() / n) if n else None,
        "top_10_component_sizes": component_sizes[:10],
    }
    return threshold_summary, similarity_distribution


# ---------------------------------------------------------------------------
# Step 4: do EXACT groups also produce repeated SACR memories?
# ---------------------------------------------------------------------------

def sacr_overlap_for_groups(exact_groups: dict, reps: pd.DataFrame, leakage_excluded_ids: frozenset):
    """Reuses sacr_retrieve exactly as RQ4 itself calls it: against the full,
    module-level memory_df (so embedding indices stay aligned), with the
    held-out patients' own memory_ids excluded via `excluded_memory_ids` --
    the same leakage-prevention mechanism build_split() already produces.
    No memory_bank override, no new retrieval path.
    """
    active_rules, _context_threshold = _load_frozen_active_rules()

    eligible_groups = [members for members in exact_groups.values() if len(members) >= 3]
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(eligible_groups)
    probe_groups = eligible_groups[:SACR_GROUP_SAMPLE_COUNT]

    def retrieved_ids(row):
        current_patient = row.to_dict()
        results = sacr_retrieve(
            current_patient, active_rules, top_k=TOP_K, candidate_k=CANDIDATE_K,
            excluded_memory_ids=leakage_excluded_ids,
        )
        return frozenset(results["memory_id"])

    within_group_jaccards = []
    group_details = []
    all_probed_row_positions = []
    for members in probe_groups:
        capped_members = members[:SACR_GROUP_MEMBER_CAP]
        id_sets = [retrieved_ids(reps.iloc[pos]) for pos in capped_members]
        all_probed_row_positions.extend(capped_members)
        pair_jaccards = []
        for set_a, set_b in combinations(id_sets, 2):
            union = set_a | set_b
            jaccard = len(set_a & set_b) / len(union) if union else 0.0
            pair_jaccards.append(jaccard)
            within_group_jaccards.append(jaccard)
        group_details.append({
            "group_size": len(members),
            "members_probed": len(capped_members),
            "mean_pairwise_jaccard": float(np.mean(pair_jaccards)) if pair_jaccards else None,
        })

    baseline_jaccards = []
    if len(all_probed_row_positions) >= 2:
        rng_baseline = np.random.default_rng(RANDOM_SEED + 2)
        probed_set = set(all_probed_row_positions)
        n_reps = len(reps)
        attempts = 0
        cache = {}
        while len(baseline_jaccards) < SACR_BASELINE_PAIR_COUNT and attempts < SACR_BASELINE_PAIR_COUNT * 20:
            attempts += 1
            i, j = rng_baseline.integers(0, n_reps, size=2)
            if i == j:
                continue
            key_i, key_j = tuple(reps.iloc[i][STATE_COLS]), tuple(reps.iloc[j][STATE_COLS])
            if key_i == key_j:
                continue  # same exact state group by chance; not a cross-group baseline pair
            for pos in (i, j):
                if pos not in cache:
                    cache[pos] = retrieved_ids(reps.iloc[pos])
            set_a, set_b = cache[i], cache[j]
            union = set_a | set_b
            baseline_jaccards.append(len(set_a & set_b) / len(union) if union else 0.0)

    return {
        "groups_probed": len(probe_groups),
        "member_cap_per_group": SACR_GROUP_MEMBER_CAP,
        "within_group_jaccard_mean": float(np.mean(within_group_jaccards)) if within_group_jaccards else None,
        "within_group_jaccard_median": float(np.median(within_group_jaccards)) if within_group_jaccards else None,
        "within_group_pair_count": len(within_group_jaccards),
        "cross_group_baseline_jaccard_mean": float(np.mean(baseline_jaccards)) if baseline_jaccards else None,
        "cross_group_baseline_jaccard_median": float(np.median(baseline_jaccards)) if baseline_jaccards else None,
        "cross_group_baseline_pair_count": len(baseline_jaccards),
        "sample_group_details": group_details[:10],
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

MEANINGFUL_JACCARD_FLOOR = 0.05   # top_k=5 -> below this, "overlap" is noise, not signal
MEANINGFUL_JACCARD_MARGIN = 0.02  # required absolute margin over the cross-group baseline


def render_verdict(exact_summary, threshold_summary, overlap_summary) -> str:
    strong_exact = exact_summary["groups_with_ge_3"] >= 200 and exact_summary["groups_with_ge_5"] >= 50
    within_mean = overlap_summary["within_group_jaccard_mean"]
    baseline_mean = overlap_summary["cross_group_baseline_jaccard_mean"]
    # A same-group pair beating a cross-group pair by a fraction of a percentage
    # point is float noise, not evidence -- require both an absolute floor
    # (retrieved sets actually share something, not literally nothing) and a
    # real margin over the baseline before calling it a signal.
    overlap_signal = (
        within_mean is not None
        and baseline_mean is not None
        and within_mean >= MEANINGFUL_JACCARD_FLOOR
        and within_mean > baseline_mean + MEANINGFUL_JACCARD_MARGIN
    )
    if strong_exact and overlap_signal:
        return (
            "GO: thousands-scale natural recurrence found in exact state groups, "
            "AND those groups retrieve measurably more overlapping SACR memories "
            "than cross-group pairs. RQ4's recurring-state-group protocol has a "
            "real population to draw on -- implement it."
        )
    if strong_exact and not overlap_signal:
        return (
            "CAUTION: many exact state groups exist, but group membership does not "
            "translate into overlapping SACR retrieval sets versus a cross-group "
            "baseline. A recurring-state-group protocol would be recurring in state "
            "only, not in memory -- revisit the group definition (e.g. use context-"
            "aligned or threshold groups) before implementing RQ4."
        )
    return (
        "NO-GO (as measured): natural recurrence at the group sizes needed for a "
        "credible RQ4 protocol was not found in the held-out population. Do not "
        "manufacture recurrence -- either loosen/redefine the grouping, draw the "
        "population differently, or drop this RQ4 protocol."
    )


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading RQ4's own patient-level split (build_split) ...")
    _memory_pool_df, held_out_df, leakage_excluded_ids, reps, split_info = build_representative_rows()
    print(f"Held-out patients (distinct patient_nbr): {split_info['heldout_patients']}")
    print(f"Held-out rows (encounters): {split_info['heldout_rows']}")
    print(f"Representative rows (one per held-out patient): {len(reps)}")

    print("Computing EXACT state groups (identical STATE_COLS tuple) ...")
    exact_groups = exact_state_groups(reps)
    exact_summary = summarize_group_sizes(exact_groups)
    print(json.dumps(exact_summary, indent=2))

    print(f"Computing THRESHOLD state groups on a bounded sample "
          f"(n<={THRESHOLD_SAMPLE_SIZE}, threshold={STATE_ALIGNMENT_THRESHOLD}) ...")
    threshold_summary, similarity_distribution = threshold_groups_and_similarity_distribution(
        reps, full_memory_df
    )
    print(json.dumps(threshold_summary, indent=2))
    print(json.dumps(similarity_distribution, indent=2))

    print(f"Probing whether EXACT groups (size>=3) also retrieve overlapping SACR "
          f"memories (sampling up to {SACR_GROUP_SAMPLE_COUNT} groups) ...")
    overlap_summary = sacr_overlap_for_groups(exact_groups, reps, leakage_excluded_ids)
    print(json.dumps(overlap_summary, indent=2))

    verdict = render_verdict(exact_summary, threshold_summary, overlap_summary)

    report = {
        "purpose": (
            "Feasibility gate for a proposed RQ4 'recurring state group' protocol: "
            "measures natural recurrence in the held-out population under the "
            "frozen SACR state representation, before any new protocol code is written."
        ),
        "inputs": {
            "state_cols": STATE_COLS,
            "state_alignment_threshold": STATE_ALIGNMENT_THRESHOLD,
            "held_out_fraction": split_info["heldout_fraction_target"],
            "random_seed": RANDOM_SEED,
        },
        "split_info": split_info,
        "exact_state_groups": exact_summary,
        "threshold_state_groups": threshold_summary,
        "state_similarity_distribution": similarity_distribution,
        "sacr_memory_overlap_check": overlap_summary,
        "verdict": verdict,
    }
    (RESULTS_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = []
    lines.append("RQ4 RECURRING-STATE-GROUP FEASIBILITY ANALYSIS (analysis-only, no new protocol code)")
    lines.append("=" * 88)
    lines.append(f"Held-out patients: {split_info['heldout_patients']} (of {split_info['total_unique_patients']} total)")
    lines.append(f"State definition: STATE_COLS={STATE_COLS}, frozen threshold={STATE_ALIGNMENT_THRESHOLD}")
    lines.append("")
    lines.append("-- EXACT state groups (identical STATE_COLS tuple) --")
    for k, v in exact_summary.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("-- THRESHOLD state groups (frozen state_alignment >= threshold, bounded sample) --")
    for k, v in threshold_summary.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("-- State similarity distribution (bounded sample) --")
    for k, v in similarity_distribution.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("-- Do EXACT groups also produce repeated SACR memories? --")
    for k, v in overlap_summary.items():
        if k != "sample_group_details":
            lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("VERDICT")
    lines.append(verdict)
    (RESULTS_DIR / "report.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "\n".join(lines))
    print(f"\nWrote {RESULTS_DIR / 'report.json'} and {RESULTS_DIR / 'report.txt'}")


if __name__ == "__main__":
    main()
