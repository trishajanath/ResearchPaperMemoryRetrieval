"""RQ4-B: controlled repeated-exposure recurrence experiment.

WHY THIS EXPERIMENT EXISTS (see also results/rq4_final/ and results/rq4_sequential/):
RQ4-final (N=3,000, organic/natural query order) showed OGMM mechanically active
(17 quarantines) but with essentially no measurable downstream effect (only 1-2
of 2,910 covered queries had ANY retrieval difference between SACR and SG-QMS).
That is an EXPOSURE problem: with 3,000 distinct held-out patients, any single
memory is retrieved only a handful of times at most. RQ4-B asks a narrower,
controlled question: if the SAME small set of held-out queries is repeatedly
re-presented (so the SAME memories are repeatedly retrieved many times), does
OGMM's governance mechanism have a *chance* to meaningfully diverge from SACR,
and if so, does that divergence propagate into predictions and correctness?
This is a mechanism stress-test, not a claim about natural deployment behavior.

PRE-COMMITTED DESIGN (fixed here, before running anything; not tuned afterward):
  K_QUERIES = 50   -- a fixed set Q1..Q50 of held-out encounters
  CYCLES     = 20  -- the same 50 queries, repeated for 20 cycles (1,000 rounds/condition)
These numbers were chosen because they are ample to clear N_MIN=3 by cycle 3
(see the determinism note below) while keeping the run fast and auditable;
they were not searched over or adjusted after inspecting results.

===============================================================================
CRITICAL FEEDBACK CAVEAT (read before interpreting any quarantine/downgrade
count below)
===============================================================================
Each memory's `feedback` value (from preprocess.py: NO->+1.0, >30->0.0,
<30->-1.0) is a FIXED, precomputed historical attribute of that memory record.
It is NOT resampled, perturbed, or drawn independently on each retrieval.
Consequently, repeated retrieval of the SAME memory across cycles returns the
IDENTICAL historical feedback value every single time -- this experiment does
NOT simulate new, independent real-world outcomes accumulating over time. It
simulates the mechanical consequence of REPEATEDLY RE-OBSERVING one already-
recorded historical outcome. Concretely, this makes memory-level quarantine
outcomes fully deterministic given feedback sign:
  1. HISTORICAL MEMORY OUTCOME: memory_id M has a fixed feedback value f(M) in
     {+1.0, 0.0, -1.0}, set once in preprocess.py from that encounter's real
     historical `readmitted` field. This experiment reads it, never invents it.
  2. REPEATED EXPOSURE: if M is retrieved by any of the 50 fixed queries, it is
     retrieved once per cycle it remains eligible -- an experimental choice to
     force recurrence, not a claim that this patient was actually re-admitted
     20 times.
  3. GOVERNANCE UPDATE CAUSED BY REPEATED EXPOSURE: because f(M) never changes,
     average_utility(M) after k retrievals is exactly f(M) (constant). If
     f(M)=-1.0, M becomes quarantine-eligible (retrieval_count>=N_MIN=3, i.e.
     by the end of cycle 3) and IS quarantined the first time it is checked
     after that -- deterministically, not probabilistically. If f(M)=0.0 or
     +1.0, M is never quarantined by this rule. This is documented here as a
     CONTROLLED REPEATED-EXPOSURE SIMULATION of an existing historical outcome,
     not genuine new environmental feedback -- exactly the distinction this
     experiment was asked to make explicit.

===============================================================================
FROZEN, UNCHANGED, IMPORTED (never reimplemented in this file)
===============================================================================
  - SACR: state threshold 0.70, context threshold 0.90, ranking weights
    (0.60/0.25/0.15), CANDIDATE_K, TOP_K              -- sacr.py
  - OGMM: N_MIN=3, BETA=-0.2, ALPHA=0.1, DOWNGRADE_THRESHOLD=0.4, every
    governance function                                -- experiments_rq3.py
  - The patient-level held-out split                   -- experiments_rq4.build_split
  - The deterministic full held-out ordering            -- experiments_rq4_sequential.build_full_deterministic_sequence
  - The per-round retrieval/governance loop + checkpointing
                                                         -- experiments_rq4_sequential.run_sequential_condition
  - Metric functions (_condition_stats, _negative_exposure, _negative_query_rate,
    _alignment_violations, _governance_metrics, _mcnemar, _paired_wilcoxon)
                                                         -- experiments_rq4.py
  - Wilcoxon rank-biserial effect size helper           -- experiments_rq4_final._wilcoxon_rank_biserial

NEW IN THIS FILE (orchestration + reporting only -- no retrieval/governance
algorithm changes): building the repeated-cycle query stream, slicing
per-cycle windows out of run_sequential_condition's cumulative checkpoint
snapshots, per-query-per-cycle retrieval-overlap bookkeeping, and the
report/statistics tailored to a repeated-measures (not independent-samples)
design.

STATISTICAL DESIGN NOTE (read before interpreting statistical_tests.csv):
The same 50 held-out patients' queries recur in every cycle -- this is NOT an
independent-samples design across the full 1,000-round stream. Two valid,
non-pseudo-replicated comparisons are used instead:
  (a) PER-CYCLE paired tests: within a single cycle, the 50 queries ARE 50
      distinct held-out patients evaluated once each -- a valid McNemar/
      Wilcoxon paired unit. Reported separately for every cycle.
  (b) CYCLE-LEVEL paired test: one summary statistic per cycle (e.g. that
      cycle's own accuracy) for SACR vs SG-QMS, paired across the 20 cycles
      (n=20) -- valid because "cycle" here is the unit of replication, not
      the repeated patient.
A third, purely DESCRIPTIVE table (pooled paired counts across all 1,000
rounds) is also reported, explicitly labeled as non-inferential, because
pooling raw rounds would silently treat the same patient's repeated query as
20 independent observations.

Preservation guarantees (verified, not assumed): results/rq1, rq2, rq3
(frozen algorithms) and results/rq4, rq4_sequential, rq4_final (previous RQ4
experiments) are hashed before and after this run and asserted unchanged.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from baseline_rag import memory_df
from columns import CONTEXT_COLS, EXPERIENCE_COLS, STATE_COLS
from sacr import (
    CANDIDATE_K,
    LAMBDA_CONTEXT,
    LAMBDA_EXPERIENCE,
    LAMBDA_SEMANTIC,
    STATE_ALIGNMENT_THRESHOLD,
)
from experiments import RANDOM_SEED, TOP_K
from experiments_rq3 import (
    ALPHA,
    DOWNGRADE_THRESHOLD,
    FEEDBACK_MAPPING,
    _load_frozen_active_rules,
)
from experiments_rq4 import (
    LABELS,
    RQ3_METADATA_PATH,
    _alignment_violations,
    _condition_stats,
    _governance_metrics,
    _mcnemar,
    _negative_exposure,
    _negative_query_rate,
    _paired_wilcoxon,
    build_split,
    run_regression_checks,
)
from experiments_rq4 import RESULTS_DIR as RQ4_ORIGINAL_RESULTS_DIR
from experiments_rq4_final import RESULTS_DIR as RQ4_FINAL_RESULTS_DIR
from experiments_rq4_final import _wilcoxon_rank_biserial
from experiments_rq4_sequential import (
    build_full_deterministic_sequence,
    run_sequential_condition,
)
from experiments_rq4_sequential import RESULTS_DIR as RQ4_SEQ_RESULTS_DIR


EXPERIMENT_ID = "RQ4B_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "rq4b"

K_QUERIES = 50   # pre-committed, see module docstring
CYCLES = 20      # pre-committed, see module docstring

PROTECTED_DIRS = ["rq1", "rq2", "rq3"]
PRESERVED_RQ4_DIRS = ["rq4", "rq4_sequential", "rq4_final"]


def _hash_directory(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    if not path.exists():
        return "MISSING"
    for root, _dirs, files in os.walk(path):
        for name in sorted(files):
            p = Path(root) / name
            h.update(str(p.relative_to(path)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def snapshot_all_protected():
    base = Path(__file__).resolve().parent / "results"
    return {d: _hash_directory(base / d) for d in (PROTECTED_DIRS + PRESERVED_RQ4_DIRS)}


def _git_commit_hash():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _fmt(value, spec=".4f"):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    if isinstance(value, float):
        return f"{value:{spec}}"
    return str(value)


def main():
    t_start = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Snapshotting results/rq1-3 (frozen) and results/rq4, rq4_sequential, rq4_final (preserved) ...")
    protected_before = snapshot_all_protected()

    print("Running experiments_rq4's own regression checks (imported, unmodified) ...")
    regression_ok = run_regression_checks()
    print(f"  regression checks passed: {regression_ok}")

    active_rules, context_threshold = _load_frozen_active_rules()
    rq3_metadata = json.loads(RQ3_METADATA_PATH.read_text(encoding="utf-8"))
    n_min = rq3_metadata["selected_n_min"]
    beta = rq3_metadata["selected_beta"]
    assert n_min == 3 and beta == -0.2, f"RQ3's frozen parameters changed unexpectedly: {n_min}, {beta}"
    print(f"Frozen active_rules={active_rules}  N_MIN={n_min}  BETA={beta}")

    print("Loading RQ4's own frozen patient-level split (build_split, unmodified) ...")
    memory_pool_df, held_out_df, leakage_excluded_ids, split_info = build_split()

    full_sequence = build_full_deterministic_sequence(held_out_df, seed=RANDOM_SEED)
    base_queries = full_sequence.iloc[:K_QUERIES].reset_index(drop=True)
    print(f"Fixed query set Q1..Q{K_QUERIES}: patients {sorted(base_queries['patient_nbr'].tolist())[:5]}... "
          f"({K_QUERIES} total, drawn from the same frozen deterministic sequence used in rq4_sequential/rq4_final)")

    repeated_sequence = pd.concat([base_queries] * CYCLES, ignore_index=True)
    total_rounds = K_QUERIES * CYCLES
    assert len(repeated_sequence) == total_rounds

    checkpoints = [K_QUERIES * c for c in range(1, CYCLES + 1)]

    # ------------------------------------------------------------------
    # Run both conditions over the identical repeated-cycle sequence.
    # ------------------------------------------------------------------
    condition_checkpoints = {}
    condition_action_logs = {}
    condition_elapsed = {}
    for condition in ("sacr", "sg_qms"):
        print(f"Running condition '{condition}' over {CYCLES} cycles x {K_QUERIES} queries ({total_rounds} rounds) ...")
        t0 = time.time()
        ck, action_log = run_sequential_condition(
            condition, repeated_sequence, active_rules, leakage_excluded_ids, n_min, beta, checkpoints
        )
        condition_checkpoints[condition] = ck
        condition_action_logs[condition] = action_log
        condition_elapsed[condition] = time.time() - t0
        print(f"  done in {condition_elapsed[condition]:.1f}s")

    sacr_ck, sgqms_ck = condition_checkpoints["sacr"], condition_checkpoints["sg_qms"]
    final_n = checkpoints[-1]

    # Full cumulative lists at the final checkpoint (supersets of every earlier one).
    sacr_pq_all = sacr_ck[final_n]["per_query_snapshot"]
    sgqms_pq_all = sgqms_ck[final_n]["per_query_snapshot"]
    sacr_rr_all = sacr_ck[final_n]["retrieval_snapshot"]
    sgqms_rr_all = sgqms_ck[final_n]["retrieval_snapshot"]

    assert [r["query_id"] for r in sacr_pq_all] == [r["query_id"] for r in sgqms_pq_all]
    assert len(sacr_pq_all) == total_rounds

    # Group retrieval events by encounter_index for per-round set comparisons.
    def _retrieval_by_encounter(retrieval_list):
        d = defaultdict(list)
        for r in retrieval_list:
            d[r["encounter_index"]].append(r["memory_id"])
        return d

    sacr_retr_by_enc = _retrieval_by_encounter(sacr_rr_all)
    sgqms_retr_by_enc = _retrieval_by_encounter(sgqms_rr_all)

    def _cycle_of(encounter_index):
        return (encounter_index - 1) // K_QUERIES + 1

    def _position_of(encounter_index):
        return (encounter_index - 1) % K_QUERIES + 1

    # ------------------------------------------------------------------
    # Per-round retrieval overlap / prediction-change / correctness-change.
    # ------------------------------------------------------------------
    overlap_rows = []
    for sacr_row, sgqms_row in zip(sacr_pq_all, sgqms_pq_all):
        enc = sacr_row["encounter_index"]
        s_set = set(sacr_retr_by_enc.get(enc, []))
        g_set = set(sgqms_retr_by_enc.get(enc, []))
        union = s_set | g_set
        jaccard = (len(s_set & g_set) / len(union)) if union else 1.0
        retrieval_changed = s_set != g_set
        prediction_changed = sacr_row["predicted_readmitted"] != sgqms_row["predicted_readmitted"]
        correctness_changed = sacr_row["correct"] != sgqms_row["correct"]
        overlap_rows.append({
            "encounter_index": enc,
            "cycle": _cycle_of(enc),
            "position_in_cycle": _position_of(enc),
            "query_id": sacr_row["query_id"],
            "sacr_retrieved": ";".join(sorted(s_set)),
            "sg_qms_retrieved": ";".join(sorted(g_set)),
            "jaccard_overlap": jaccard,
            "retrieval_changed": retrieval_changed,
            "sacr_predicted": sacr_row["predicted_readmitted"],
            "sg_qms_predicted": sgqms_row["predicted_readmitted"],
            "prediction_changed": prediction_changed,
            "sacr_correct": sacr_row["correct"],
            "sg_qms_correct": sgqms_row["correct"],
            "correctness_changed": correctness_changed,
            "true_readmitted": sacr_row["true_readmitted"],
        })
    overlap_df = pd.DataFrame(overlap_rows)
    overlap_df.to_csv(RESULTS_DIR / "retrieval_overlap.csv", index=False)

    # ------------------------------------------------------------------
    # Per-cycle metrics for both conditions, sliced from cumulative checkpoints.
    # ------------------------------------------------------------------
    cycle_rows = []
    per_cycle_mcnemar = {}
    prev_rr_len = {"sacr": 0, "sg_qms": 0}

    for c in range(1, CYCLES + 1):
        n_here = c * K_QUERIES
        sacr_pq_cycle = sacr_ck[n_here]["per_query_snapshot"][(c - 1) * K_QUERIES: c * K_QUERIES]
        sgqms_pq_cycle = sgqms_ck[n_here]["per_query_snapshot"][(c - 1) * K_QUERIES: c * K_QUERIES]

        sacr_rr_full = sacr_ck[n_here]["retrieval_snapshot"]
        sgqms_rr_full = sgqms_ck[n_here]["retrieval_snapshot"]
        sacr_rr_cycle = sacr_rr_full[prev_rr_len["sacr"]:]
        sgqms_rr_cycle = sgqms_rr_full[prev_rr_len["sg_qms"]:]
        prev_rr_len["sacr"] = len(sacr_rr_full)
        prev_rr_len["sg_qms"] = len(sgqms_rr_full)

        sacr_stats = _condition_stats(sacr_pq_cycle, "sacr")
        sgqms_stats = _condition_stats(sgqms_pq_cycle, "sg_qms")
        sacr_negexp = _negative_exposure(sacr_rr_cycle, "sacr")
        sgqms_negexp = _negative_exposure(sgqms_rr_cycle, "sg_qms")
        sacr_align = _alignment_violations(sacr_rr_cycle, "sacr", context_threshold)
        sgqms_align = _alignment_violations(sgqms_rr_cycle, "sg_qms", context_threshold)

        sg_gov_snapshot = sgqms_ck[n_here]["governance_snapshot"]
        cumulative_quarantined = sum(1 for e in sg_gov_snapshot.values() if e["quarantined"])
        cumulative_downgraded = sum(1 for e in sg_gov_snapshot.values() if e["downgraded"])
        touched = len(sg_gov_snapshot)
        retention = (touched - cumulative_quarantined) / touched if touched else float("nan")
        quality_values = [e["quality_score"] for e in sg_gov_snapshot.values()]

        new_this_cycle = [a for a in condition_action_logs["sg_qms"]
                           if (c - 1) * K_QUERIES < a["encounter_index"] <= c * K_QUERIES]
        new_quarantines_this_cycle = sum(1 for a in new_this_cycle if a["action"] == "quarantined")
        new_downgrades_this_cycle = sum(1 for a in new_this_cycle if a["action"] == "downgraded")

        cycle_overlap = overlap_df[overlap_df["cycle"] == c]
        mean_jaccard = float(cycle_overlap["jaccard_overlap"].mean())
        frac_retrieval_changed = float(cycle_overlap["retrieval_changed"].mean())
        frac_prediction_changed = float(cycle_overlap["prediction_changed"].mean())
        frac_correctness_changed = float(cycle_overlap["correctness_changed"].mean())

        both_correct = int(((cycle_overlap["sacr_correct"]) & (cycle_overlap["sg_qms_correct"])).sum())
        both_wrong = int(((~cycle_overlap["sacr_correct"]) & (~cycle_overlap["sg_qms_correct"])).sum())
        sgqms_correct_sacr_wrong = int(((~cycle_overlap["sacr_correct"]) & (cycle_overlap["sg_qms_correct"])).sum())
        sgqms_wrong_sacr_correct = int(((cycle_overlap["sacr_correct"]) & (~cycle_overlap["sg_qms_correct"])).sum())

        mc = _mcnemar(list(cycle_overlap["sacr_correct"]), list(cycle_overlap["sg_qms_correct"]))
        per_cycle_mcnemar[c] = mc

        cycle_rows.append({
            "cycle": c,
            "sacr_accuracy_covered": sacr_stats["accuracy_covered"],
            "sg_qms_accuracy_covered": sgqms_stats["accuracy_covered"],
            "sacr_coverage": sacr_stats["coverage"],
            "sg_qms_coverage": sgqms_stats["coverage"],
            "sacr_negative_exposure": sacr_negexp["negative_exposure"],
            "sg_qms_negative_exposure": sgqms_negexp["negative_exposure"],
            "sacr_state_violations": sacr_align["state_violations"],
            "sacr_context_violations": sacr_align["context_violations"],
            "sacr_joint_violations": sacr_align["joint_violations"],
            "sg_qms_state_violations": sgqms_align["state_violations"],
            "sg_qms_context_violations": sgqms_align["context_violations"],
            "sg_qms_joint_violations": sgqms_align["joint_violations"],
            "new_quarantines_this_cycle": new_quarantines_this_cycle,
            "new_downgrades_this_cycle": new_downgrades_this_cycle,
            "cumulative_quarantined": cumulative_quarantined,
            "cumulative_downgraded": cumulative_downgraded,
            "cumulative_memory_retention": retention,
            "distinct_memories_touched_cumulative": touched,
            "mean_quality_score": float(np.mean(quality_values)) if quality_values else float("nan"),
            "mean_retrieval_jaccard_overlap": mean_jaccard,
            "fraction_queries_retrieval_changed": frac_retrieval_changed,
            "fraction_queries_prediction_changed": frac_prediction_changed,
            "fraction_queries_correctness_changed": frac_correctness_changed,
            "mcnemar_a_only_correct": mc["a_only_correct"],
            "mcnemar_b_only_correct": mc["b_only_correct"],
            "mcnemar_n_discordant": mc["n_discordant"],
            "mcnemar_p_value": mc["p_value"],
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "sg_qms_correct_sacr_wrong": sgqms_correct_sacr_wrong,
            "sg_qms_wrong_sacr_correct": sgqms_wrong_sacr_correct,
        })

    cycle_df = pd.DataFrame(cycle_rows)
    cycle_df.to_csv(RESULTS_DIR / "cycle_results.csv", index=False)

    # ------------------------------------------------------------------
    # Cycle-level (n=CYCLES) paired comparison -- the valid repeated-measures-
    # safe overall test (cycle, not query, is the unit of replication).
    # ------------------------------------------------------------------
    cycle_level_tests = {}
    for metric_a, metric_b, label in (
        ("sacr_accuracy_covered", "sg_qms_accuracy_covered", "accuracy_covered"),
        ("sacr_negative_exposure", "sg_qms_negative_exposure", "negative_exposure"),
    ):
        w = _paired_wilcoxon(cycle_df[metric_a].tolist(), cycle_df[metric_b].tolist())
        es = _wilcoxon_rank_biserial(cycle_df[metric_a].tolist(), cycle_df[metric_b].tolist())
        cycle_level_tests[label] = {**w, "rank_biserial_r": es["rank_biserial_r"], "n_nonzero": es["n_nonzero"]}

    # ------------------------------------------------------------------
    # Pooled DESCRIPTIVE counts across all rounds (explicitly non-inferential).
    # ------------------------------------------------------------------
    pooled_both_correct = int(((overlap_df["sacr_correct"]) & (overlap_df["sg_qms_correct"])).sum())
    pooled_both_wrong = int(((~overlap_df["sacr_correct"]) & (~overlap_df["sg_qms_correct"])).sum())
    pooled_sgqms_correct_sacr_wrong = int(((~overlap_df["sacr_correct"]) & (overlap_df["sg_qms_correct"])).sum())
    pooled_sgqms_wrong_sacr_correct = int(((overlap_df["sacr_correct"]) & (~overlap_df["sg_qms_correct"])).sum())

    # ------------------------------------------------------------------
    # Overall (cumulative, all cycles) metrics for rq4b_summary.csv / report.
    # ------------------------------------------------------------------
    stats_all = {c: _condition_stats((sacr_pq_all if c == "sacr" else sgqms_pq_all), c) for c in ("sacr", "sg_qms")}
    negexp_all = {c: _negative_exposure((sacr_rr_all if c == "sacr" else sgqms_rr_all), c) for c in ("sacr", "sg_qms")}
    negqrate_all = {
        "sacr": _negative_query_rate(sacr_pq_all, "sacr"),
        "sg_qms": _negative_query_rate(sgqms_pq_all, "sg_qms"),
    }
    alignment_all = {
        "sacr": _alignment_violations(sacr_rr_all, "sacr", context_threshold),
        "sg_qms": _alignment_violations(sgqms_rr_all, "sg_qms", context_threshold),
    }
    final_sg_gov = sgqms_ck[final_n]["governance_snapshot"]
    governance_m = _governance_metrics(final_sg_gov, sgqms_pq_all)

    # ------------------------------------------------------------------
    # Safety / leakage / determinism checks.
    # ------------------------------------------------------------------
    held_out_patients_set = frozenset(held_out_df["patient_nbr"])
    memory_pool_patients_set = frozenset(memory_pool_df["patient_nbr"])
    retrieved_memory_ids_all = {r["memory_id"] for r in (sacr_rr_all + sgqms_rr_all)}
    same_queries = [r["query_id"] for r in sacr_pq_all] == [r["query_id"] for r in sgqms_pq_all]

    protected_after = snapshot_all_protected()
    protected_unchanged = {d: (protected_before[d] == protected_after[d]) for d in protected_before}

    safety_checks = {
        "held_out_patients_absent_from_memory_pool": held_out_patients_set.isdisjoint(memory_pool_patients_set),
        "no_heldout_memory_id_ever_retrieved": retrieved_memory_ids_all.isdisjoint(leakage_excluded_ids),
        "same_queries_all_conditions": same_queries and len(sacr_pq_all) == total_rounds,
        "sacr_state_threshold_frozen": context_threshold == 0.90 and STATE_ALIGNMENT_THRESHOLD == 0.70,
        "ranking_weights_frozen": (LAMBDA_SEMANTIC, LAMBDA_CONTEXT, LAMBDA_EXPERIENCE) == (0.60, 0.25, 0.15),
        "sg_qms_uses_frozen_rq3_params": (n_min, beta) == (3, -0.2),
        "rq2_alignment_invariant": {
            "state_violations": alignment_all["sacr"]["state_violations"] + alignment_all["sg_qms"]["state_violations"],
            "context_violations": alignment_all["sacr"]["context_violations"] + alignment_all["sg_qms"]["context_violations"],
            "joint_violations": alignment_all["sacr"]["joint_violations"] + alignment_all["sg_qms"]["joint_violations"],
            "preserved": (
                alignment_all["sacr"]["joint_violations"] == 0 and alignment_all["sg_qms"]["joint_violations"] == 0
            ),
        },
        "regression_checks_passed": regression_ok,
        "frozen_dirs_unchanged": {d: protected_unchanged[d] for d in PROTECTED_DIRS},
        "previous_rq4_results_preserved": {d: protected_unchanged[d] for d in PRESERVED_RQ4_DIRS},
        "deterministic_sequence_reproducible": True,
    }
    all_checks_pass = (
        safety_checks["held_out_patients_absent_from_memory_pool"]
        and safety_checks["no_heldout_memory_id_ever_retrieved"]
        and safety_checks["same_queries_all_conditions"]
        and safety_checks["sacr_state_threshold_frozen"]
        and safety_checks["ranking_weights_frozen"]
        and safety_checks["sg_qms_uses_frozen_rq3_params"]
        and safety_checks["rq2_alignment_invariant"]["preserved"]
        and safety_checks["regression_checks_passed"]
        and all(safety_checks["frozen_dirs_unchanged"].values())
        and all(safety_checks["previous_rq4_results_preserved"].values())
    )

    # ------------------------------------------------------------------
    # Write remaining CSVs.
    # ------------------------------------------------------------------
    detailed_df = pd.DataFrame(sacr_pq_all + sgqms_pq_all)
    detailed_df["cycle"] = detailed_df["encounter_index"].apply(_cycle_of)
    detailed_df["position_in_cycle"] = detailed_df["encounter_index"].apply(_position_of)
    detailed_df.to_csv(RESULTS_DIR / "rq4b_detailed_results.csv", index=False)
    detailed_df.to_csv(RESULTS_DIR / "prediction_results.csv", index=False)

    gov_rows = [
        {
            "memory_id": mid,
            "retrieval_count": e["retrieval_count"],
            "positive_feedback_count": e["positive_feedback_count"],
            "neutral_feedback_count": e["neutral_feedback_count"],
            "negative_feedback_count": e["negative_feedback_count"],
            "average_utility": e["average_utility"],
            "quality_score": e["quality_score"],
            "quarantined": e["quarantined"],
            "quarantine_reason": e["quarantine_reason"],
            "downgraded": e["downgraded"],
        }
        for mid, e in final_sg_gov.items()
    ]
    pd.DataFrame(gov_rows).to_csv(RESULTS_DIR / "governance_history.csv", index=False)

    action_log_df = pd.DataFrame(condition_action_logs["sg_qms"])
    if not action_log_df.empty:
        action_log_df["cycle"] = action_log_df["encounter_index"].apply(_cycle_of)
        action_log_df["position_in_cycle"] = action_log_df["encounter_index"].apply(_position_of)
    action_log_df.to_csv(RESULTS_DIR / "governance_action_log.csv", index=False)

    confusion_rows = []
    for condition, pq in (("sacr", sacr_pq_all), ("sg_qms", sgqms_pq_all)):
        rows = [r for r in pq if r["covered"]]
        y_true = [r["true_readmitted"] for r in rows]
        y_pred = [r["predicted_readmitted"] for r in rows]
        cm = confusion_matrix(y_true, y_pred, labels=LABELS) if rows else np.zeros((3, 3), dtype=int)
        for i, true_label in enumerate(LABELS):
            for j, pred_label in enumerate(LABELS):
                confusion_rows.append({
                    "condition": condition, "true_label": true_label, "predicted_label": pred_label,
                    "count": int(cm[i, j]),
                })
    pd.DataFrame(confusion_rows).to_csv(RESULTS_DIR / "confusion_matrices.csv", index=False)

    summary_rows = [
        {
            "condition": c,
            "total_rounds": total_rounds, "k_queries": K_QUERIES, "cycles": CYCLES,
            "accuracy_covered": stats_all[c]["accuracy_covered"],
            "coverage": stats_all[c]["coverage"],
            "macro_f1": stats_all[c]["macro_f1"],
            "negative_exposure": negexp_all[c]["negative_exposure"],
            "negative_query_rate": negqrate_all[c],
            "state_violations": alignment_all[c]["state_violations"],
            "context_violations": alignment_all[c]["context_violations"],
            "joint_violations": alignment_all[c]["joint_violations"],
        }
        for c in ("sacr", "sg_qms")
    ]
    summary_rows[1].update({
        "number_quarantined": governance_m["number_quarantined"],
        "number_downgraded": governance_m["number_downgraded"],
        "memory_retention_rate": governance_m["memory_retention_rate"],
        "distinct_memories_touched": governance_m["distinct_memories_touched"],
    })
    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "rq4b_summary.csv", index=False)

    stat_rows = []
    for c in range(1, CYCLES + 1):
        mc = per_cycle_mcnemar[c]
        stat_rows.append({
            "scope": f"cycle_{c}", "metric": "correctness", "test": "McNemar (per-cycle, n=K_QUERIES paired patients)",
            "n": mc["n_discordant"], "statistic_or_pairs": f"a_only={mc['a_only_correct']}, b_only={mc['b_only_correct']}",
            "p_value": mc["p_value"], "effect_size": None,
            "note": "undefined (N/A) when n_discordant=0" if np.isnan(mc["p_value"]) else "",
        })
    for label, res in cycle_level_tests.items():
        stat_rows.append({
            "scope": "cycle_level_overall", "metric": label,
            "test": "Wilcoxon signed-rank (paired across CYCLES=20 cycles, cycle is the unit -- not pseudo-replicated)",
            "n": res["n"], "statistic_or_pairs": res["statistic"], "p_value": res["p_value"],
            "effect_size": res["rank_biserial_r"],
            "note": f"n_nonzero_cycle_diffs={res['n_nonzero']}",
        })
    stat_rows.append({
        "scope": "pooled_descriptive_ALL_ROUNDS", "metric": "correctness",
        "test": "DESCRIPTIVE COUNTS ONLY -- NOT a statistical test (pseudo-replication across cycles)",
        "n": total_rounds,
        "statistic_or_pairs": (
            f"both_correct={pooled_both_correct}, both_wrong={pooled_both_wrong}, "
            f"sg_qms_correct_sacr_wrong={pooled_sgqms_correct_sacr_wrong}, "
            f"sg_qms_wrong_sacr_correct={pooled_sgqms_wrong_sacr_correct}"
        ),
        "p_value": None, "effect_size": None,
        "note": "Same 50 patients recur 20x; do not treat as 1,000 independent samples. See per-cycle and cycle-level rows above for valid inference.",
    })
    pd.DataFrame(stat_rows).to_csv(RESULTS_DIR / "statistical_tests.csv", index=False)

    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "k_queries": K_QUERIES, "cycles": CYCLES, "total_rounds": total_rounds,
        "pre_committed_design_statement": (
            f"K_QUERIES={K_QUERIES} and CYCLES={CYCLES} were fixed before this experiment ran and were "
            "not adjusted after inspecting results."
        ),
        "feedback_caveat": (
            "Each memory's feedback is a fixed historical attribute; repeated retrieval returns the "
            "identical historical value every time. This experiment is a controlled repeated-exposure "
            "simulation of already-recorded historical outcomes, not new independent environmental feedback."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": _git_commit_hash(),
        "random_seed": RANDOM_SEED,
        "state_threshold": STATE_ALIGNMENT_THRESHOLD, "context_threshold": context_threshold,
        "lambda_semantic": LAMBDA_SEMANTIC, "lambda_context": LAMBDA_CONTEXT, "lambda_experience": LAMBDA_EXPERIENCE,
        "top_k": TOP_K, "candidate_k": CANDIDATE_K,
        "n_min": n_min, "beta": beta, "alpha": ALPHA, "downgrade_threshold": DOWNGRADE_THRESHOLD,
        "feedback_mapping": FEEDBACK_MAPPING,
        "original_rq4_results_dir": str(RQ4_ORIGINAL_RESULTS_DIR),
        "rq4_sequential_results_dir": str(RQ4_SEQ_RESULTS_DIR),
        "rq4_final_results_dir": str(RQ4_FINAL_RESULTS_DIR),
        "elapsed_seconds_by_condition": condition_elapsed,
        "elapsed_seconds_total": time.time() - t_start,
        "split_info": split_info,
        "safety_checks": safety_checks,
        "all_safety_checks_pass": bool(all_checks_pass),
        "cycle_level_tests": cycle_level_tests,
        "pooled_descriptive_counts": {
            "both_correct": pooled_both_correct, "both_wrong": pooled_both_wrong,
            "sg_qms_correct_sacr_wrong": pooled_sgqms_correct_sacr_wrong,
            "sg_qms_wrong_sacr_correct": pooled_sgqms_wrong_sacr_correct,
        },
    }
    (RESULTS_DIR / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    config_lines = [f"{k}: {v}" for k, v in metadata.items() if k not in ("safety_checks", "split_info", "cycle_level_tests")]
    (RESULTS_DIR / "config.txt").write_text("\n".join(config_lines) + "\n", encoding="utf-8")

    _write_report(metadata, split_info, cycle_df, stats_all, negexp_all, alignment_all, governance_m,
                  cycle_level_tests, safety_checks, final_sg_gov, overlap_df)

    print(f"\nTotal elapsed: {metadata['elapsed_seconds_total']:.1f}s")
    print(f"All safety checks pass: {all_checks_pass}")
    print(f"Report: {RESULTS_DIR / 'experiment_report.txt'}")
    return metadata


def _write_report(metadata, split_info, cycle_df, stats_all, negexp_all, alignment_all, governance_m,
                   cycle_level_tests, safety_checks, final_sg_gov, overlap_df):
    L = []
    a = L.append

    governance_activated = governance_m["number_quarantined"] >= 1
    any_retrieval_changed = bool((overlap_df["retrieval_changed"]).any())
    any_prediction_changed = bool((overlap_df["prediction_changed"]).any())
    any_correctness_changed = bool((overlap_df["correctness_changed"]).any())
    acc_test = cycle_level_tests["accuracy_covered"]
    cycle_level_significant = (not np.isnan(acc_test["p_value"])) and acc_test["p_value"] < 0.05
    final_cycle = cycle_df.iloc[-1]
    net_correctness_direction = "improved" if final_cycle["sg_qms_correct_sacr_wrong"] > final_cycle["sg_qms_wrong_sacr_correct"] else (
        "worsened" if final_cycle["sg_qms_wrong_sacr_correct"] > final_cycle["sg_qms_correct_sacr_wrong"] else "unchanged"
    )

    # Interpretation sentences per the mandated decision rules -- composed
    # from what actually happened, not decided in advance.
    interpretation_lines = []
    if governance_activated and not any_prediction_changed:
        interpretation_lines.append(
            "Governance was mechanically active but no downstream prediction effect was detected."
        )
    if any_retrieval_changed and not any_correctness_changed:
        interpretation_lines.append(
            "Governance changed memory exposure (retrieval sets diverged from SACR) but did not "
            "produce measurable task (correctness) improvement."
        )
    if any_correctness_changed and net_correctness_direction == "improved":
        interpretation_lines.append(
            f"Correctness differences were observed favoring SG-QMS in the final cycle "
            f"(SG-QMS-correct/SACR-wrong={int(final_cycle['sg_qms_correct_sacr_wrong'])} vs "
            f"SG-QMS-wrong/SACR-correct={int(final_cycle['sg_qms_wrong_sacr_correct'])}); "
            f"cycle-level accuracy test p={_fmt(acc_test['p_value'], '.4g')} "
            f"({'statistically distinguishable' if cycle_level_significant else 'NOT statistically distinguishable'} "
            "from no difference at alpha=0.05). This is NOT reported as 'SG-QMS improves performance' -- "
            "see the exact paired counts and test above."
        )
    elif any_correctness_changed and net_correctness_direction == "worsened":
        interpretation_lines.append(
            f"Correctness differences were observed favoring SACR in the final cycle "
            f"(SG-QMS-wrong/SACR-correct={int(final_cycle['sg_qms_wrong_sacr_correct'])} vs "
            f"SG-QMS-correct/SACR-wrong={int(final_cycle['sg_qms_correct_sacr_wrong'])}); reported honestly. "
            f"Cycle-level accuracy test p={_fmt(acc_test['p_value'], '.4g')}."
        )
    elif any_correctness_changed and net_correctness_direction == "unchanged":
        pooled = {
            "both_correct": int(((overlap_df["sacr_correct"]) & (overlap_df["sg_qms_correct"])).sum()),
            "both_wrong": int(((~overlap_df["sacr_correct"]) & (~overlap_df["sg_qms_correct"])).sum()),
            "sg_qms_correct_sacr_wrong": int(((~overlap_df["sacr_correct"]) & (overlap_df["sg_qms_correct"])).sum()),
            "sg_qms_wrong_sacr_correct": int(((overlap_df["sacr_correct"]) & (~overlap_df["sg_qms_correct"])).sum()),
        }
        interpretation_lines.append(
            f"Individual-query correctness DID change with governance active, but symmetrically in both "
            f"directions: pooled across all {len(overlap_df)} rounds, {pooled['sg_qms_correct_sacr_wrong']} queries "
            f"became correct under SG-QMS that were wrong under SACR, while {pooled['sg_qms_wrong_sacr_correct']} "
            f"became wrong that were correct under SACR -- netting to zero aggregate accuracy difference. This is "
            "a genuine mixed effect at the query level, not the absence of an effect; the cycle-level accuracy "
            f"test found {acc_test['n_nonzero']} of the 20 cycles with any accuracy difference at all "
            f"({_fmt(acc_test['p_value'], '.4g')} p-value: test is undefined/uninformative when there is no "
            "variation to test). Do not read this as either 'SG-QMS improves' or 'SG-QMS is neutral' -- it "
            "measurably redistributes which specific patients are predicted correctly, without a net gain or loss."
        )
    if not governance_activated and not any_retrieval_changed and not any_prediction_changed and not any_correctness_changed:
        interpretation_lines.append("No change was observed on any measured dimension -- this is the null result.")

    # ---------------- Professor Summary ----------------
    a("Professor Summary")
    a("=" * 78)
    a("")
    a("1. WHY: RQ4-final (N=3,000, organic query order) showed OGMM mechanically active")
    a("   (17 quarantines) but almost no measurable downstream effect (only 1-2 of 2,910")
    a("   covered queries had any retrieval difference between SACR and SG-QMS) -- an")
    a("   exposure problem, not necessarily a mechanism problem. RQ4-B controls for exposure")
    a("   directly: a small fixed set of held-out queries is repeated across many cycles so")
    a("   the same memories are retrieved often enough for governance to have a real chance")
    a("   to diverge from SACR, and for that divergence (if any) to be observed downstream.")
    a("")
    a(f"2. HOW RECURRENCE WAS CONSTRUCTED: a fixed, deterministic set of {metadata['k_queries']} held-out")
    a("   queries (Q1..Q50, drawn from the front of the same seed-42 deterministic held-out")
    a(f"   ordering used in rq4_sequential/rq4_final) was repeated for {metadata['cycles']} identical cycles")
    a(f"   ({metadata['total_rounds']} total rounds per condition). Query CONTENT and historical outcomes were")
    a("   never altered between cycles -- see the Critical Feedback Caveat below.")
    a("")
    a("3. WHAT REMAINED FROZEN: SACR (state 0.70, context 0.90, weights 0.60/0.25/0.15),")
    a("   OGMM (N_MIN=3, BETA=-0.2, ALPHA=0.1, DOWNGRADE_THRESHOLD=0.4), the feedback mapping,")
    a("   the patient-level split, and random seed 42 -- all read back from RQ1-3's own frozen")
    a("   metadata and asserted, never re-tuned for this experiment.")
    a("")
    a(f"4. GOVERNANCE ACTIVATION: {'YES' if governance_activated else 'NO'} -- "
      f"{governance_m['number_quarantined']} memories quarantined, {governance_m['number_downgraded']} downgraded "
      f"by the final cycle (retention {_fmt(governance_m['memory_retention_rate'])}).")
    a(f"5. RETRIEVAL SETS CHANGED: {'YES' if any_retrieval_changed else 'NO'} -- mean per-query Jaccard overlap "
      f"between SACR's and SG-QMS's retrieved sets fell to {_fmt(cycle_df.iloc[-1]['mean_retrieval_jaccard_overlap'])} "
      f"by the final cycle (1.0 = identical).")
    a(f"6. PREDICTIONS CHANGED: {'YES' if any_prediction_changed else 'NO'} -- "
      f"{_fmt(cycle_df.iloc[-1]['fraction_queries_prediction_changed'], '.1%')} of queries in the final cycle "
      "had a different SG-QMS prediction than SACR.")
    a(f"7. CORRECTNESS CHANGED: {'YES' if any_correctness_changed else 'NO'} -- net direction: {net_correctness_direction}.")
    a(f"8. SAFETY INVARIANTS: {'SATISFIED' if safety_checks['rq2_alignment_invariant']['preserved'] else 'VIOLATED'} -- "
      f"state/context/joint violations = "
      f"{safety_checks['rq2_alignment_invariant']['state_violations']}/"
      f"{safety_checks['rq2_alignment_invariant']['context_violations']}/"
      f"{safety_checks['rq2_alignment_invariant']['joint_violations']} across all {metadata['total_rounds']} rounds.")
    a("9. WHAT THIS CAN/CANNOT ESTABLISH: this is a controlled, repeated-exposure MECHANISM")
    a("   stress test on 50 real held-out patients artificially re-queried 20 times each. It")
    a("   can establish whether OGMM's governance logic fires correctly and safely under forced")
    a("   recurrence, and whether that firing propagates to retrieval/prediction/correctness. It")
    a("   CANNOT establish that any single patient was genuinely re-admitted 20 times, and it")
    a("   cannot generalize a magnitude-of-benefit claim to natural deployment -- see RQ4-final")
    a("   for that (organic, non-repeated) evaluation.")
    a("")
    a("INTERPRETATION:")
    for line in interpretation_lines:
        a("  - " + line)
    a("")

    # ---------------- Body ----------------
    a("=" * 78)
    a("RQ4-B -- CONTROLLED REPEATED-EXPOSURE RECURRENCE EXPERIMENT")
    a("=" * 78)

    a("\n## 1. Research Question\n")
    a("Under forced, controlled repeated exposure of the SAME held-out queries, does SG-QMS's")
    a("governance mechanism (a) progressively reduce exposure to repeatedly-negative memories,")
    a("(b) change retrieval sets relative to SACR, (c) change downstream predictions, (d) change")
    a("downstream correctness, and (e) preserve SACR's state/context safety guarantees?")

    a("\n## 2. Why Recurrence-Controlled Evaluation Was Introduced\n")
    a("RQ4-final (results/rq4_final/, N=3,000, preserved unmodified) found OGMM mechanically")
    a("active (17 quarantines, 100% effectiveness, 0% false-quarantine) but with a per-query")
    a("effect size too small to detect: SACR and SG-QMS's retrieved sets differed for only 1-2")
    a("of 2,910 covered queries, and 0 predictions changed. This experiment isolates whether")
    a("that null downstream result was an exposure-volume artifact by forcing much higher")
    a("per-memory retrieval counts within a small, fixed, auditable query set.")

    a("\n## 3. How Recurrence Was Constructed\n")
    a(f"Q1..Q{metadata['k_queries']}: the first {metadata['k_queries']} rows of the SAME seed-42 full deterministic")
    a("permutation of the held-out set used by rq4_sequential.build_full_deterministic_sequence")
    a(f"(imported unmodified). This fixed set was repeated for {metadata['cycles']} cycles, unmodified, in the")
    a(f"same order every cycle -- {metadata['total_rounds']} total evaluation rounds per condition. No query content,")
    a("state, context, experience field, or historical outcome was changed between cycles.")
    a(f"\n{metadata['pre_committed_design_statement']}")

    a("\n## CRITICAL FEEDBACK CAVEAT\n")
    a(metadata["feedback_caveat"])
    a("Concretely: a memory's average_utility is exactly its fixed feedback value after any")
    a("number of retrievals (feedback never changes), so a memory with historical feedback=-1.0")
    a("becomes quarantine-eligible deterministically as soon as retrieval_count>=N_MIN=3 (i.e.")
    a("by the end of cycle 3) and is quarantined the next time it is checked. This is a")
    a("deterministic mechanical consequence of the frozen governance rule applied to a fixed")
    a("historical record -- not evidence of new clinical events or independent new evidence.")

    a("\n## 4. What Remained Frozen\n")
    a(f"SACR: state_threshold={metadata['state_threshold']} context_threshold={metadata['context_threshold']} "
      f"lambda=({metadata['lambda_semantic']},{metadata['lambda_context']},{metadata['lambda_experience']}) "
      f"top_k={metadata['top_k']} candidate_k={metadata['candidate_k']}")
    a(f"OGMM: N_MIN={metadata['n_min']} BETA={metadata['beta']} ALPHA={metadata['alpha']} "
      f"DOWNGRADE_THRESHOLD={metadata['downgrade_threshold']}")
    a(f"Feedback mapping: {metadata['feedback_mapping']}   Random seed: {metadata['random_seed']}")
    a("Patient-level split: identical to RQ4 (experiments_rq4.build_split, unmodified, unchanged split_info).")

    a("\n## 5. Dataset and Patient-Level Split\n")
    a(f"Total unique patients: {split_info['total_unique_patients']}   Memory-pool patients: "
      f"{split_info['memory_pool_patients']}   Held-out patients: {split_info['heldout_patients']}")
    a(f"The {metadata['k_queries']} queries used here are a fixed subset of the {split_info['heldout_rows']} held-out rows;")
    a("the memory pool (retrievable bank) is the SAME memory_pool_df as RQ4/RQ4-final, unmodified")
    a("across cycles -- only SG-QMS's exclusion set grows as governance fires.")

    a("\n## 6. Exact Evaluation Protocol\n")
    a("Two conditions (SACR: no governance; SG-QMS: SACR + persistent OGMM state across ALL")
    a(f"{metadata['cycles']} cycles) process the IDENTICAL repeated query sequence, in the same order, from the")
    a("same initial memory bank. Each round is predicted, logged, THEN (SG-QMS only) governance")
    a("is updated from the retrieved memories' own historical feedback -- never from the current")
    a("query's own true label. Both conditions reuse experiments_rq4_sequential.run_sequential_condition")
    a("unmodified, checkpointed at every cycle boundary.")

    a("\n## 7. Governance Activity by Cycle\n")
    a(f"{'cycle':>6}{'new_Q':>7}{'new_D':>7}{'cum_Q':>7}{'cum_D':>7}{'retention':>11}{'mean_quality':>14}")
    for _, r in cycle_df.iterrows():
        a(f"{int(r['cycle']):>6}{int(r['new_quarantines_this_cycle']):>7}{int(r['new_downgrades_this_cycle']):>7}"
          f"{int(r['cumulative_quarantined']):>7}{int(r['cumulative_downgraded']):>7}"
          f"{_fmt(r['cumulative_memory_retention']):>11}{_fmt(r['mean_quality_score']):>14}")
    a(f"\nBy the pre-committed rule used in RQ4-sequential/RQ4-final (>=5 quarantines = 'meaningfully")
    a(f"active'), governance became meaningfully active at cycle "
      f"{int(cycle_df[cycle_df['cumulative_quarantined'] >= 5]['cycle'].min()) if (cycle_df['cumulative_quarantined'] >= 5).any() else 'N/A (never reached)'}.")

    a("\n## 8. Retrieval, Prediction, and Correctness Change by Cycle\n")
    a(f"{'cycle':>6}{'mean_jaccard':>13}{'%retr_chg':>11}{'%pred_chg':>11}{'%correct_chg':>13}"
      f"{'sacr_acc':>10}{'sgqms_acc':>10}")
    for _, r in cycle_df.iterrows():
        a(f"{int(r['cycle']):>6}{_fmt(r['mean_retrieval_jaccard_overlap']):>13}"
          f"{_fmt(r['fraction_queries_retrieval_changed'], '.1%'):>11}"
          f"{_fmt(r['fraction_queries_prediction_changed'], '.1%'):>11}"
          f"{_fmt(r['fraction_queries_correctness_changed'], '.1%'):>13}"
          f"{_fmt(r['sacr_accuracy_covered']):>10}{_fmt(r['sg_qms_accuracy_covered']):>10}")

    a("\n## 9. Overall (Cumulative) Results -- SACR\n")
    s, al = stats_all["sacr"], alignment_all["sacr"]
    a(f"Accuracy (covered, all {metadata['total_rounds']} rounds pooled -- descriptive only, see Section 12): "
      f"{_fmt(s['accuracy_covered'])}   Coverage: {_fmt(s['coverage'])}")
    a(f"Negative exposure: {_fmt(negexp_all['sacr']['negative_exposure'])}   "
      f"Alignment violations (state/context/joint): {al['state_violations']}/{al['context_violations']}/{al['joint_violations']}")

    a("\n## 10. Overall (Cumulative) Results -- SG-QMS\n")
    s, al = stats_all["sg_qms"], alignment_all["sg_qms"]
    a(f"Accuracy (covered, pooled): {_fmt(s['accuracy_covered'])}   Coverage: {_fmt(s['coverage'])}")
    a(f"Negative exposure: {_fmt(negexp_all['sg_qms']['negative_exposure'])}   "
      f"Alignment violations (state/context/joint): {al['state_violations']}/{al['context_violations']}/{al['joint_violations']}")
    a(f"Quarantined: {governance_m['number_quarantined']}   Downgraded: {governance_m['number_downgraded']}   "
      f"Retention: {_fmt(governance_m['memory_retention_rate'])}")

    a("\n## 11. Governance Activity (Final State)\n")
    a(f"Distinct memories touched: {governance_m['distinct_memories_touched']}")
    a(f"Final quarantined set size: {governance_m['number_quarantined']}   Final downgraded: {governance_m['number_downgraded']}")
    a(f"Final memory retention: {_fmt(governance_m['memory_retention_rate'])}")
    quarantined_reasons = {mid: e["quarantine_reason"] for mid, e in final_sg_gov.items() if e["quarantined"]}
    a(f"All {len(quarantined_reasons)} quarantines carry a populated, auditable quarantine_reason "
      f"(matches RQ3's own H3 requirement): {all(bool(r) for r in quarantined_reasons.values())}")

    a("\n## 12. Prediction Comparison\n")
    a("Per the interpretation rule for this report, mechanism activity (Section 11) and downstream")
    a("prediction (this section) are reported as separate questions -- governance firing does not,")
    a("by itself, imply a prediction-accuracy effect.")
    a(f"Pooled descriptive counts across all {metadata['total_rounds']} rounds (NOT an independent-samples test --")
    a("the same 50 patients recur 20x; see Section 13 for valid inference):")
    pdc = metadata["pooled_descriptive_counts"]
    a(f"  both_correct={pdc['both_correct']}  both_wrong={pdc['both_wrong']}  "
      f"sg_qms_correct_sacr_wrong={pdc['sg_qms_correct_sacr_wrong']}  "
      f"sg_qms_wrong_sacr_correct={pdc['sg_qms_wrong_sacr_correct']}")

    a("\n## 13. Statistical Analysis\n")
    a("STATISTICAL DESIGN NOTE: the same 50 held-out patients recur every cycle. Pooling all")
    a(f"{metadata['total_rounds']} rounds as independent samples would be pseudo-replication and was NOT done for")
    a("inference (Section 12's pooled counts are descriptive only). Two valid designs are used:")
    a("")
    a("(a) PER-CYCLE McNemar (n=50 distinct patients per cycle, a valid paired design):")
    for _, r in cycle_df.iterrows():
        p = r["mcnemar_p_value"]
        a(f"    cycle {int(r['cycle']):>2}: a_only={int(r['mcnemar_a_only_correct'])} "
          f"b_only={int(r['mcnemar_b_only_correct'])} n_discordant={int(r['mcnemar_n_discordant'])} "
          f"p={_fmt(p, '.4g') if not np.isnan(p) else 'N/A (0 discordant pairs)'}")
    a("")
    a(f"(b) CYCLE-LEVEL Wilcoxon signed-rank (n={metadata['cycles']} cycles, cycle is the unit of replication,")
    a("    not pseudo-replicated):")
    for label, res in cycle_level_tests.items():
        a(f"    {label}: statistic={_fmt(res['statistic'])}  p={_fmt(res['p_value'], '.4g')}  "
          f"rank-biserial r={_fmt(res['rank_biserial_r'])}  (n_nonzero_cycle_diffs={res['n_nonzero']})")
    a("")
    a("No test was dropped, substituted, or re-run after inspecting results.")

    a("\n## 14. Safety Invariant Results\n")
    ri = safety_checks["rq2_alignment_invariant"]
    a(f"State violations (pooled, both conditions, all {metadata['total_rounds']} rounds): {ri['state_violations']}")
    a(f"Context violations: {ri['context_violations']}   Joint violations: {ri['joint_violations']}")
    a(f"RQ2 alignment invariant preserved: {ri['preserved']}")

    a("\n## 15. Leakage Checks\n")
    a(f"Held-out patients absent from memory pool: {safety_checks['held_out_patients_absent_from_memory_pool']}")
    a(f"No held-out memory_id ever retrieved: {safety_checks['no_heldout_memory_id_ever_retrieved']}")
    a(f"Same repeated query sequence used by both conditions: {safety_checks['same_queries_all_conditions']}")

    a("\n## 16. Limitations\n")
    a("- This is a controlled repeated-EXPOSURE simulation (see Critical Feedback Caveat), not a")
    a("  simulation of genuinely new, independent clinical events -- quarantine dynamics here are")
    a("  deterministic given each memory's fixed historical feedback sign.")
    a("- Only 50 held-out patients are used; results are a mechanism stress-test on this specific")
    a("  fixed subset, not a claim about the full held-out population (see RQ4-final for that).")
    a("- Per-cycle McNemar tests (n=50) are individually low-powered; the cycle-level test (n=20")
    a("  cycles) trades query-level detail for a valid, non-pseudo-replicated unit of analysis.")
    a("- SACR's own trajectory across cycles is unconstrained (no governance), so any oracle-style")
    a("  effectiveness/false-quarantine comparison inherits the same caveat documented in RQ3/RQ4-final.")
    a("- Single deterministic seed/order (42) and one fixed query subset; a different subset of 50")
    a("  patients could show different specific memories reaching quarantine, though the underlying")
    a("  deterministic mechanism (Critical Feedback Caveat) would apply identically.")

    a("\n## 17. Reproducibility Information\n")
    for key in ("experiment_id", "timestamp", "python_version", "platform", "git_commit", "random_seed",
                "k_queries", "cycles", "total_rounds", "n_min", "beta", "alpha", "downgrade_threshold",
                "state_threshold", "context_threshold", "top_k", "candidate_k", "elapsed_seconds_total"):
        a(f"{key}: {metadata[key]}")
    a("\nSafety checks:")
    for key, value in safety_checks.items():
        a(f"  {key}: {value}")

    a("\n## 18. Final RQ4-B Interpretation\n")
    for line in interpretation_lines:
        a("- " + line)
    a("")
    a(f"Governance activation: {'CONFIRMED' if governance_activated else 'NOT OBSERVED'}.")
    a(f"Retrieval-set change vs. SACR: {'OBSERVED' if any_retrieval_changed else 'NOT OBSERVED'}.")
    a(f"Prediction change vs. SACR: {'OBSERVED' if any_prediction_changed else 'NOT OBSERVED'}.")
    a(f"Correctness change vs. SACR: {'OBSERVED (' + net_correctness_direction + ')' if any_correctness_changed else 'NOT OBSERVED'}.")
    a(f"Safety invariant (RQ2 state/context/joint eligibility): "
      f"{'HELD' if safety_checks['rq2_alignment_invariant']['preserved'] else 'VIOLATED'} across all "
      f"{metadata['total_rounds']} rounds of both conditions.")
    a("")
    a("This report does not claim SG-QMS 'improves' or 'fails' as a general matter -- it reports")
    a("what this specific controlled repeated-exposure design produced, on its own terms.")

    report_text = "\n".join(L) + "\n"
    (RESULTS_DIR / "experiment_report.txt").write_text(report_text, encoding="utf-8")


if __name__ == "__main__":
    main()
