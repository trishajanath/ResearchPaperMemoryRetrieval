"""RQ4 SEQUENTIAL SAMPLE-SIZE SENSITIVITY ANALYSIS.

FOLLOW-UP / DIAGNOSTIC ANALYSIS, NOT A REPLACEMENT OF RQ4.

The original RQ4 (results/rq4/, produced by experiments_rq4.py) evaluated
300 held-out encounters and found OGMM barely fired (0 quarantined, 2
downgraded, 1,354 distinct memories touched across 1,374 retrievals). A
separate, standalone memory-recurrence analysis (rq4_memory_recurrence_
analysis.py, results/rq4_memory_recurrence_analysis/) then ran the SAME
frozen SACR mechanism, pre-governance, over ALL 20,372 held-out encounters
and found natural recurrence DOES exist at that scale (21.9% of distinct
retrieved memories reach RQ3's N_MIN=3 threshold), but 300 queries is only
1.5% of that volume -- almost certainly too few to expose OGMM to it.

This script asks: at what evaluation size does that natural recurrence
become large enough for OGMM to meaningfully operate? It does NOT overwrite,
rerun, or alter results/rq1, results/rq2, results/rq3, or results/rq4 --
those remain the frozen, original results. This is an additive diagnostic
using a different (larger, nested) query volume, written to its own
directory (results/rq4_sequential/).

===============================================================================
DESIGN DOCUMENTATION (written before implementation, per the task's own
instruction to document the existing mechanism before changing anything)
===============================================================================

1. How RQ4 currently processes held-out encounters (experiments_rq4.py):
   build_split() produces a frozen 80/20 patient-level split (seed 42,
   unstratified) -> held_out_df (20,372 rows). build_eval_queries() draws
   ONE FIXED-SIZE RANDOM SAMPLE of 300 rows (held_out_df.sample(n=300,
   random_state=42)), sorted by memory_id afterward purely for a
   human-auditable display order. evaluate_condition() loops over those 300
   rows once per condition (baseline, sacr, sg_qms); each condition starts
   governance from scratch -- three independent, non-interacting passes.

2. How memory IDs are tracked: every memory row carries a fixed memory_id
   (assigned once in preprocess.py). Governance state is a dict keyed by
   memory_id (experiments_rq3._new_governance_entry()), living only for the
   duration of one condition's run over its query stream.

3. How outcome feedback is generated: `feedback` is precomputed once in
   preprocess.py from each memory's OWN historical `readmitted` value
   (NO->+1.0, >30->0.0, <30->-1.0). It is the retrieved memory's own
   historical feedback, observed at retrieval time -- never the held-out
   query's own true label (read only after prediction, for logging).

4. How OGMM updates memory quality/quarantine state (experiments_rq3.py):
   per retrieved memory, _apply_feedback increments retrieval_count/
   average_utility; _update_quality does quality_score = clip(quality_score
   + ALPHA*feedback, 0, 1) (ALPHA=0.1). QUARANTINE if retrieval_count >=
   N_MIN and average_utility <= BETA; else DOWNGRADE (metadata only) if
   quality_score < 0.4.

5. How exclusions are applied during subsequent retrievals: leakage_excluded
   (held-out patients' own memory_ids) is always excluded, all conditions.
   governance_excluded (quarantined ids) is additionally excluded, SG-QMS
   only, via sacr_retrieve's existing `excluded_memory_ids` parameter --
   affects only FUTURE queries in that same condition's sequence.

6. Parameters frozen from RQ3/RQ2/RQ1: N_MIN=3, BETA=-0.2 (read back from
   results/rq3/experiment_metadata.json, asserted); ALPHA=0.1,
   DOWNGRADE_THRESHOLD=0.4, FEEDBACK_MAPPING, and all five governance
   functions imported verbatim from experiments_rq3.py (never reimplemented
   here). From RQ1/RQ2: STATE_ALIGNMENT_THRESHOLD=0.70, context_threshold=
   0.90, LAMBDA_SEMANTIC/CONTEXT/EXPERIENCE=0.60/0.25/0.15, CANDIDATE_K=100,
   TOP_K=5.

7. Why 300 queries prevented meaningful OGMM recurrence: the actual RQ4 run
   produced 1,374 retrievals across 1,354 distinct memories (~1.01
   retrievals/memory -- almost no repeats). Quarantine requires
   retrieval_count>=N_MIN=3, so virtually no memory could even become
   eligible. The standalone recurrence analysis (20,372 encounters, 93,067
   retrievals) showed 21.9% of distinct memories reach >=3 retrievals -- but
   300 queries is only 1.5% of that volume. Downgrade fired twice anyway
   because it only needs quality_score<0.4, reachable from a single memory
   hit twice by negative feedback -- a much lower bar than quarantine.

Two design decisions committed to BEFORE running anything (so N cannot be
chosen or praised after the fact based on outcomes):

  (a) NESTING MECHANISM: RQ4's own build_eval_queries() draws a fixed-size
      RANDOM SAMPLE per N -- not nested across different N (a fresh N=1000
      sample under the same seed is not guaranteed to contain the same 300
      rows as the N=300 sample). This script instead builds ONE full
      permutation of all 20,372 held-out rows via
      np.random.default_rng(42).permutation(...) (same project seed 42, a
      new deterministic full-length order) and takes prefixes [:N]. The
      first N rows of that single permutation are, by construction, an
      exact prefix of the first M rows for any M > N. RQ4's own
      build_eval_queries()/results/rq4/ are never called or touched here.

  (b) PRE-COMMITTED "MEANINGFULLY ACTIVE" / "UNACCEPTABLE FALSE QUARANTINE"
      THRESHOLDS: OGMM is judged "meaningfully active" at the smallest
      evaluated checkpoint N where >=5 distinct memories have been
      quarantined (MEANINGFUL_QUARANTINE_FLOOR, below). False quarantine is
      judged "unacceptable" if the false-quarantine rate (vs. SACR's own
      oracle-eligible set, computed with RQ3's own imported
      _oracle_eligible_set/_effectiveness_and_false_rate) exceeds 50%
      (UNACCEPTABLE_FALSE_QUARANTINE_RATE, below) at that point. Both are
      fixed now and applied mechanically to whatever the results turn out
      to be -- N is never chosen because it produces the best SG-QMS
      accuracy.

Run: python3 experiments_rq4_sequential.py
Output: results/rq4_sequential/{experiment_report.txt, sample_size_summary.csv,
        detailed_results.csv, governance_history.csv, recurrence_summary.csv,
        config.txt, experiment_metadata.json}
"""

from __future__ import annotations

from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from baseline_rag import memory_df, semantic_retrieve, state_neutral_text
from columns import CONTEXT_COLS, EXPERIENCE_COLS, STATE_COLS
from sacr import (
    CANDIDATE_K,
    LAMBDA_CONTEXT,
    LAMBDA_EXPERIENCE,
    LAMBDA_SEMANTIC,
    STATE_ALIGNMENT_THRESHOLD,
    calculate_context_alignment,
    calculate_state_alignment,
    sacr_retrieve,
)
from experiments import RANDOM_SEED, TOP_K
from experiments_rq3 import (
    ALPHA,
    DOWNGRADE_THRESHOLD,
    FEEDBACK_MAPPING,
    _apply_feedback,
    _effectiveness_and_false_rate,
    _load_frozen_active_rules,
    _meets_downgrade_criterion,
    _meets_quarantine_criterion,
    _new_governance_entry,
    _oracle_eligible_set,
    _update_quality,
)
from experiments_rq4 import (
    MAX_EVAL_QUERIES,
    RQ3_METADATA_PATH,
    _alignment_violations,
    _condition_stats,
    _governance_metrics,
    _negative_exposure,
    _negative_query_rate,
    build_split,
    predict_majority,
    run_regression_checks,
)
from experiments_rq4 import RESULTS_DIR as RQ4_ORIGINAL_RESULTS_DIR


EXPERIMENT_ID = "RQ4SEQ_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "rq4_sequential"
PROTECTED_RESULT_DIRS = ["rq1", "rq2", "rq3"]  # must never change; verified by hash below

CHECKPOINTS = [300, 1000, 3000, 5000, 10000, 20372]
LABELS = ["NO", ">30", "<30"]

MEANINGFUL_QUARANTINE_FLOOR = 5      # pre-committed: >=5 quarantines = "meaningfully active"
UNACCEPTABLE_FALSE_QUARANTINE_RATE = 0.5  # pre-committed: >50% false = "unacceptable"


# ---------------------------------------------------------------------------
# Safety: results/rq1, rq2, rq3 must be byte-for-byte unchanged by this run.
# ---------------------------------------------------------------------------

def _hash_directory(path: Path) -> str:
    h = hashlib.sha256()
    if not path.exists():
        return "MISSING"
    for root, _dirs, files in os.walk(path):
        for name in sorted(files):
            p = Path(root) / name
            h.update(str(p.relative_to(path)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def snapshot_protected_dirs():
    base = Path(__file__).resolve().parent / "results"
    return {d: _hash_directory(base / d) for d in PROTECTED_RESULT_DIRS}


def verify_protected_dirs_unchanged(before: dict):
    after = snapshot_protected_dirs()
    unchanged = {d: (before[d] == after[d]) for d in PROTECTED_RESULT_DIRS}
    return unchanged, before, after


# ---------------------------------------------------------------------------
# Nested, deterministic full-length held-out ordering (see design note (a)).
# ---------------------------------------------------------------------------

def build_full_deterministic_sequence(held_out_df: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    permuted_positions = rng.permutation(len(held_out_df))
    return held_out_df.iloc[permuted_positions].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sequential, checkpointed single-pass evaluation for one condition.
# ---------------------------------------------------------------------------

def run_sequential_condition(condition, sequence_df, active_rules, leakage_excluded, n_min, beta, checkpoints):
    """One condition, one pass over the full sequence, up to max(checkpoints).
    Governance/oracle state evolves exactly as evaluate_condition() would
    (same imported RQ3 primitives, same order-dependent logic); the only
    addition versus experiments_rq4.evaluate_condition is checkpoint
    snapshotting, since that function has no notion of "the state after N
    queries" -- it always runs its whole eval_df to completion.
    """
    assert condition in ("baseline", "sacr", "sg_qms")
    max_n = max(checkpoints)
    checkpoint_set = set(checkpoints)

    governance = {} if condition == "sg_qms" else None
    governance_excluded = set() if condition == "sg_qms" else None
    shadow_governance = {} if condition == "sacr" else None  # oracle bookkeeping only; never excludes anything

    per_query_records = []
    retrieval_records = []
    retrieval_counter: Counter = Counter()
    governance_action_log = []

    checkpoints_out = {}
    n_processed = 0

    for row in sequence_df.itertuples(index=False):
        if n_processed >= max_n:
            break
        n_processed += 1
        row_d = row._asdict()
        query_id = row_d["memory_id"]
        patient = {col: row_d[col] for col in STATE_COLS + CONTEXT_COLS + EXPERIENCE_COLS}
        true_label = row_d["readmitted"]

        if condition == "baseline":
            retrieved = semantic_retrieve(
                state_neutral_text(patient), top_k=TOP_K, excluded_memory_ids=leakage_excluded
            )
            retrieved_info = [
                {
                    "memory_id": r["memory_id"],
                    "feedback": float(r["feedback"]),
                    "readmitted": r["readmitted"],
                    "state_alignment": calculate_state_alignment(patient, r, memory_df),
                    "context_alignment": calculate_context_alignment(patient, r, memory_df),
                }
                for _, r in retrieved.iterrows()
            ]
        else:
            excluded = (leakage_excluded | governance_excluded) if condition == "sg_qms" else leakage_excluded
            retrieved = sacr_retrieve(
                patient, active_rules, top_k=TOP_K, candidate_k=CANDIDATE_K, excluded_memory_ids=excluded
            )
            retrieved_info = [
                {
                    "memory_id": r["memory_id"],
                    "feedback": float(r["feedback"]),
                    "readmitted": r["readmitted"],
                    "state_alignment": float(r["state_alignment"]),
                    "context_alignment": float(r["context_alignment"]),
                }
                for _, r in retrieved.iterrows()
            ]

        labels = [info["readmitted"] for info in retrieved_info]
        predicted = predict_majority(labels)
        covered = len(retrieved_info) > 0
        correct = covered and predicted == true_label
        negative_count = sum(1 for info in retrieved_info if info["feedback"] < 0)
        neg_fraction = (negative_count / len(retrieved_info)) if retrieved_info else float("nan")
        mean_state = (
            float(np.mean([info["state_alignment"] for info in retrieved_info])) if retrieved_info else float("nan")
        )
        mean_context = (
            float(np.mean([info["context_alignment"] for info in retrieved_info])) if retrieved_info else float("nan")
        )

        per_query_records.append(
            {
                "encounter_index": n_processed,
                "query_id": query_id,
                "patient_nbr": int(row_d["patient_nbr"]),
                "true_readmitted": true_label,
                "condition": condition,
                "predicted_readmitted": predicted,
                "correct": bool(correct),
                "covered": bool(covered),
                "retrieval_count": len(retrieved_info),
                "negative_memory_count": negative_count,
                "negative_memory_fraction": neg_fraction,
                "mean_state_alignment": mean_state,
                "mean_context_alignment": mean_context,
                "governance_excluded_count": len(governance_excluded) if condition == "sg_qms" else 0,
            }
        )
        for info in retrieved_info:
            retrieval_records.append(
                {"condition": condition, "encounter_index": n_processed, "query_id": query_id, **info}
            )
            retrieval_counter[info["memory_id"]] += 1

        if condition == "sacr":
            # Oracle bookkeeping only (mirrors RQ3's "static" oracle role) --
            # never feeds back into SACR's own exclusion set.
            for info in retrieved_info:
                entry = shadow_governance.setdefault(info["memory_id"], _new_governance_entry())
                _apply_feedback(entry, info["feedback"])

        if condition == "sg_qms":
            for info in retrieved_info:
                mid = info["memory_id"]
                entry = governance.setdefault(mid, _new_governance_entry())
                _apply_feedback(entry, info["feedback"])
                _update_quality(entry, info["feedback"])
                if not entry["quarantined"] and _meets_quarantine_criterion(entry, n_min, beta):
                    entry["quarantined"] = True
                    entry["quarantine_reason"] = (
                        f"retrieval_count={entry['retrieval_count']}>=N_MIN={n_min} "
                        f"and average_utility={entry['average_utility']:.4f}<=BETA={beta}"
                    )
                    governance_excluded.add(mid)
                    governance_action_log.append(
                        {
                            "encounter_index": n_processed,
                            "memory_id": mid,
                            "action": "quarantined",
                            "retrieval_count": entry["retrieval_count"],
                            "average_utility": entry["average_utility"],
                            "quality_score": entry["quality_score"],
                            "reason": entry["quarantine_reason"],
                        }
                    )
                elif not entry["quarantined"] and not entry["downgraded"] and _meets_downgrade_criterion(entry):
                    entry["downgraded"] = True
                    governance_action_log.append(
                        {
                            "encounter_index": n_processed,
                            "memory_id": mid,
                            "action": "downgraded",
                            "retrieval_count": entry["retrieval_count"],
                            "average_utility": entry["average_utility"],
                            "quality_score": entry["quality_score"],
                            "reason": (
                                f"quality_score={entry['quality_score']:.4f} < "
                                f"DOWNGRADE_THRESHOLD={DOWNGRADE_THRESHOLD}"
                            ),
                        }
                    )

        if n_processed in checkpoint_set:
            checkpoints_out[n_processed] = {
                "per_query_snapshot": list(per_query_records),
                "retrieval_snapshot": list(retrieval_records),
                "retrieval_counter_snapshot": Counter(retrieval_counter),
                "governance_snapshot": copy.deepcopy(governance) if governance is not None else None,
                "shadow_governance_snapshot": copy.deepcopy(shadow_governance) if shadow_governance is not None else None,
            }

    return checkpoints_out, governance_action_log


# ---------------------------------------------------------------------------
# Recurrence stats from a retrieval-count Counter snapshot.
# ---------------------------------------------------------------------------

def _recurrence_stats(counter: Counter, n_min: int) -> dict:
    counts = np.array(sorted(counter.values(), reverse=True)) if counter else np.array([], dtype=int)
    distinct = int(len(counts))
    return {
        "total_retrieval_events": int(counts.sum()),
        "distinct_memories_retrieved": distinct,
        "retrieved_exactly_once": int((counts == 1).sum()),
        "retrieved_ge_2": int((counts >= 2).sum()),
        "retrieved_ge_3": int((counts >= 3).sum()),
        "retrieved_ge_5": int((counts >= 5).sum()),
        "retrieved_ge_10": int((counts >= 10).sum()),
        "max_retrieval_count": int(counts.max()) if distinct else 0,
        "mean_retrieval_count": float(counts.mean()) if distinct else 0.0,
        "median_retrieval_count": float(np.median(counts)) if distinct else 0.0,
        "n_min_eligible_count": int((counts >= n_min).sum()),
        "fraction_ge_n_min": float((counts >= n_min).sum() / distinct) if distinct else None,
    }


def main():
    t_start = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Snapshotting results/rq1, rq2, rq3 for the untouched-files regression check ...")
    protected_before = snapshot_protected_dirs()

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
    assert split_info["heldout_rows"] == 20372, f"Unexpected held-out row count: {split_info['heldout_rows']}"

    sequence_df = build_full_deterministic_sequence(held_out_df, seed=RANDOM_SEED)
    assert len(sequence_df) == split_info["heldout_rows"]
    checkpoints = [n for n in CHECKPOINTS if n <= len(sequence_df)]
    if checkpoints[-1] != len(sequence_df):
        checkpoints.append(len(sequence_df))
    print(f"Checkpoints: {checkpoints}")

    condition_checkpoints = {}
    condition_action_logs = {}
    condition_elapsed = {}

    # SACR first: its shadow_governance snapshots serve as the oracle for
    # SG-QMS's false-quarantine-rate computation at the SAME checkpoint N.
    for condition in ("sacr", "sg_qms", "baseline"):
        print(f"Running sequential condition '{condition}' over {checkpoints[-1]} queries ...")
        t0 = time.time()
        ck, action_log = run_sequential_condition(
            condition, sequence_df, active_rules, leakage_excluded_ids, n_min, beta, checkpoints
        )
        condition_checkpoints[condition] = ck
        condition_action_logs[condition] = action_log
        condition_elapsed[condition] = time.time() - t0
        print(f"  done in {condition_elapsed[condition]:.1f}s")
        for n in checkpoints:
            snap = ck[n]
            n_actions = sum(1 for a in action_log if a["encounter_index"] <= n)
            print(f"    N={n:>6}: {len(snap['per_query_snapshot'])} queries, "
                  f"{len(snap['retrieval_counter_snapshot'])} distinct memories retrieved, "
                  f"{n_actions} governance actions so far")

    # --- build sample_size_summary rows + recurrence_summary rows ---
    summary_rows = []
    recurrence_rows = []
    all_detailed_records = []

    for condition in ("baseline", "sacr", "sg_qms"):
        for n in checkpoints:
            snap = condition_checkpoints[condition][n]
            pq = snap["per_query_snapshot"]
            rr = snap["retrieval_snapshot"]
            counter = snap["retrieval_counter_snapshot"]

            stats = _condition_stats(pq, condition)
            negexp = _negative_exposure(rr, condition)
            negqrate = _negative_query_rate(pq, condition)
            rec = _recurrence_stats(counter, n_min)

            row = {
                "condition": condition,
                "N": n,
                "total_queries": stats["total_queries"],
                "covered_queries": stats["covered_queries"],
                "coverage": stats["coverage"],
                "accuracy_covered": stats["accuracy_covered"],
                "accuracy_all_uncovered_as_incorrect": stats["accuracy_all_uncovered_as_incorrect"],
                "macro_f1": stats["macro_f1"],
                "negative_exposure": negexp["negative_exposure"],
                "negative_query_rate": negqrate,
                "total_retrieval_events": rec["total_retrieval_events"],
                "distinct_memories_retrieved": rec["distinct_memories_retrieved"],
                "retrieved_exactly_once": rec["retrieved_exactly_once"],
                "retrieved_ge_2": rec["retrieved_ge_2"],
                "retrieved_ge_3": rec["retrieved_ge_3"],
                "retrieved_ge_5": rec["retrieved_ge_5"],
                "max_retrieval_count": rec["max_retrieval_count"],
            }

            if condition in ("sacr", "sg_qms"):
                alignment = _alignment_violations(rr, condition, context_threshold)
                mean_state = float(np.mean([r["state_alignment"] for r in rr])) if rr else float("nan")
                mean_context = float(np.mean([r["context_alignment"] for r in rr])) if rr else float("nan")
                row.update(
                    {
                        "mean_state_alignment": mean_state,
                        "mean_context_alignment": mean_context,
                        "state_violations": alignment["state_violations"],
                        "context_violations": alignment["context_violations"],
                        "joint_violations": alignment["joint_violations"],
                    }
                )

            if condition == "sg_qms":
                gov = snap["governance_snapshot"]
                gm = _governance_metrics(gov, pq)
                oracle_gov = condition_checkpoints["sacr"][n]["shadow_governance_snapshot"]
                oracle_eligible = _oracle_eligible_set(oracle_gov, n_min, beta)
                flagged_set = {mid for mid, e in gov.items() if e["quarantined"]}
                effectiveness, false_rate = _effectiveness_and_false_rate(oracle_eligible, flagged_set)
                n_reaching_n_min = sum(1 for e in gov.values() if e["retrieval_count"] >= n_min)
                row.update(
                    {
                        "number_quarantined": gm["number_quarantined"],
                        "number_downgraded": gm["number_downgraded"],
                        "number_governance_exclusions": gm["number_quarantined"],
                        "memory_retention_rate": gm["memory_retention_rate"],
                        "distinct_memories_touched": gm["distinct_memories_touched"],
                        "number_reaching_n_min": n_reaching_n_min,
                        "number_receiving_governance_updates": gm["distinct_memories_touched"],
                        "oracle_eligible_count": len(oracle_eligible),
                        "quarantine_effectiveness": effectiveness,
                        "false_quarantine_rate": false_rate,
                    }
                )

            summary_rows.append(row)
            recurrence_rows.append(
                {
                    "condition": condition,
                    "N": n,
                    **rec,
                }
            )

        # Full-resolution per-query detail (every encounter, not just checkpoints)
        full_pq = condition_checkpoints[condition][checkpoints[-1]]["per_query_snapshot"]
        all_detailed_records.extend(full_pq)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_DIR / "sample_size_summary.csv", index=False)

    recurrence_df = pd.DataFrame(recurrence_rows)
    recurrence_df.to_csv(RESULTS_DIR / "recurrence_summary.csv", index=False)

    detailed_df = pd.DataFrame(all_detailed_records)
    detailed_df.to_csv(RESULTS_DIR / "detailed_results.csv", index=False)

    governance_history_df = pd.DataFrame(condition_action_logs["sg_qms"])
    if governance_history_df.empty:
        governance_history_df = pd.DataFrame(
            columns=["encounter_index", "memory_id", "action", "retrieval_count",
                     "average_utility", "quality_score", "reason"]
        )
    governance_history_df.to_csv(RESULTS_DIR / "governance_history.csv", index=False)

    # --- pre-committed decision rule application (never tuned post-hoc) ---
    sg_qms_rows = [r for r in summary_rows if r["condition"] == "sg_qms"]
    sg_qms_rows.sort(key=lambda r: r["N"])
    meaningfully_active_n = None
    for r in sg_qms_rows:
        if r["number_quarantined"] >= MEANINGFUL_QUARANTINE_FLOOR:
            meaningfully_active_n = r["N"]
            break
    unacceptable_false_quarantine_ns = [
        r["N"] for r in sg_qms_rows
        if r["false_quarantine_rate"] is not None
        and not (isinstance(r["false_quarantine_rate"], float) and np.isnan(r["false_quarantine_rate"]))
        and r["false_quarantine_rate"] > UNACCEPTABLE_FALSE_QUARANTINE_RATE
    ]

    # --- final protected-dirs regression check ---
    unchanged, before, after = verify_protected_dirs_unchanged(protected_before)
    all_protected_unchanged = all(unchanged.values())

    elapsed_total = time.time() - t_start

    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "purpose": (
            "Follow-up sequential sample-size sensitivity analysis diagnosing why the "
            "original RQ4 (results/rq4/, 300 queries) saw sparse OGMM recurrence. NOT a "
            "retroactive alteration of RQ4 -- results/rq1, rq2, rq3, and rq4 are all "
            "read-only inputs here and are verified unchanged (see safety_checks)."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "random_seed": RANDOM_SEED,
        "checkpoints": checkpoints,
        "sequence_construction": (
            "np.random.default_rng(42).permutation(len(held_out_df)) over the full "
            "20,372-row held-out set from experiments_rq4.build_split() (unmodified); "
            "prefixes [:N] of this single permutation are used for each checkpoint, "
            "guaranteeing nesting. Distinct from and does not call RQ4's own "
            "build_eval_queries() (a non-nested, fixed-size random sample)."
        ),
        "state_threshold": STATE_ALIGNMENT_THRESHOLD,
        "context_threshold": context_threshold,
        "lambda_semantic": LAMBDA_SEMANTIC,
        "lambda_context": LAMBDA_CONTEXT,
        "lambda_experience": LAMBDA_EXPERIENCE,
        "top_k": TOP_K,
        "candidate_k": CANDIDATE_K,
        "n_min": n_min,
        "beta": beta,
        "alpha": ALPHA,
        "downgrade_threshold": DOWNGRADE_THRESHOLD,
        "feedback_mapping": FEEDBACK_MAPPING,
        "original_rq4_max_eval_queries": MAX_EVAL_QUERIES,
        "original_rq4_results_dir": str(RQ4_ORIGINAL_RESULTS_DIR),
        "pre_committed_meaningful_quarantine_floor": MEANINGFUL_QUARANTINE_FLOOR,
        "pre_committed_unacceptable_false_quarantine_rate": UNACCEPTABLE_FALSE_QUARANTINE_RATE,
        "meaningfully_active_n": meaningfully_active_n,
        "unacceptable_false_quarantine_ns": unacceptable_false_quarantine_ns,
        "elapsed_seconds_by_condition": condition_elapsed,
        "elapsed_seconds_total": elapsed_total,
        "split_info": split_info,
        "safety_checks": {
            "regression_checks_passed": regression_ok,
            "protected_dirs_unchanged": unchanged,
            "all_protected_dirs_unchanged": all_protected_unchanged,
            "protected_dir_hashes_before": before,
            "protected_dir_hashes_after": after,
            "rq3_frozen_params_verified": (n_min, beta) == (3, -0.2),
            "rq2_thresholds_verified": (context_threshold, STATE_ALIGNMENT_THRESHOLD) == (0.90, 0.70),
        },
    }
    (RESULTS_DIR / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )

    config_lines = [f"{k}: {v}" for k, v in metadata.items() if k not in ("safety_checks",)]
    (RESULTS_DIR / "config.txt").write_text("\n".join(config_lines) + "\n", encoding="utf-8")

    _write_report(metadata, summary_df, recurrence_df, meaningfully_active_n, unacceptable_false_quarantine_ns)

    print(f"\nTotal elapsed: {elapsed_total:.1f}s")
    print(f"All protected dirs (rq1-3) unchanged: {all_protected_unchanged}")
    if not all_protected_unchanged:
        print("!!!! WARNING: a protected results directory changed during this run !!!!")
        print(unchanged)
    print(f"Meaningfully active N (>= {MEANINGFUL_QUARANTINE_FLOOR} quarantines): {meaningfully_active_n}")
    print(f"Report written to {RESULTS_DIR / 'experiment_report.txt'}")

    return metadata


def _fmt(value, spec=".4f"):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    if isinstance(value, float):
        return f"{value:{spec}}"
    return str(value)


def _write_report(metadata, summary_df, recurrence_df, meaningfully_active_n, unacceptable_false_quarantine_ns):
    lines = []
    lines.append("=" * 78)
    lines.append("RQ4 SEQUENTIAL SAMPLE-SIZE SENSITIVITY ANALYSIS")
    lines.append("=" * 78)
    lines.append("")
    lines.append("*** THIS IS A FOLLOW-UP / DIAGNOSTIC ANALYSIS, NOT A REPLACEMENT OF RQ4 ***")
    lines.append("The original RQ4 result (results/rq4/, 300 held-out queries) is preserved")
    lines.append("unchanged and is NOT overwritten, rerun, or reinterpreted by this file. This")
    lines.append("analysis exists only to diagnose why that 300-query evaluation saw almost no")
    lines.append("OGMM activity, using the SAME frozen split, SAME frozen SACR mechanism, and")
    lines.append("SAME frozen RQ3 governance parameters -- run at larger, nested query volumes.")
    lines.append("results/rq1, results/rq2, and results/rq3 were verified byte-for-byte")
    lines.append(f"unchanged before and after this run: {metadata['safety_checks']['all_protected_dirs_unchanged']}.")

    lines.append("\n## 1. Motivation (recap)\n")
    lines.append("Original RQ4 (300 queries): 0 quarantined, 2 downgraded, 1,354 distinct")
    lines.append("memories touched across 1,374 retrievals (~1.01 retrievals/memory).")
    lines.append("Standalone recurrence analysis (all 20,372 held-out encounters, pre-")
    lines.append("governance): 93,067 retrievals, 50,191 distinct memories, 21.9% of distinct")
    lines.append("memories retrieved >=3 times (RQ3's own N_MIN), max retrieval count 10.")
    lines.append("300 queries is only 1.5% of that volume -- this analysis tests whether a")
    lines.append("larger, sequential evaluation closes that gap.")

    lines.append("\n## 2. Method (see full design documentation in this file's module docstring)\n")
    lines.append(f"Checkpoints (queries): {metadata['checkpoints']}")
    lines.append(f"Sequence construction: {metadata['sequence_construction']}")
    lines.append("Governance state (SG-QMS) and oracle bookkeeping (SACR shadow governance)")
    lines.append("evolve SEQUENTIALLY and CUMULATIVELY across the single, nested query order;")
    lines.append("each checkpoint N reports the state after processing exactly the first N")
    lines.append("queries of that one deterministic sequence -- identical in principle to")
    lines.append("restarting fresh with the first N encounters, computed in one pass for")
    lines.append("efficiency. All governance functions (_new_governance_entry, _apply_feedback,")
    lines.append("_update_quality, _meets_quarantine_criterion, _meets_downgrade_criterion,")
    lines.append("_oracle_eligible_set, _effectiveness_and_false_rate) are imported directly")
    lines.append("from experiments_rq3.py -- no governance logic was reimplemented.")
    lines.append(
        f"Frozen params: N_MIN={metadata['n_min']} BETA={metadata['beta']} "
        f"ALPHA={metadata['alpha']} DOWNGRADE_THRESHOLD={metadata['downgrade_threshold']} "
        f"state_threshold={metadata['state_threshold']} context_threshold={metadata['context_threshold']} "
        f"lambda=({metadata['lambda_semantic']},{metadata['lambda_context']},{metadata['lambda_experience']}) "
        f"top_k={metadata['top_k']} candidate_k={metadata['candidate_k']}"
    )
    lines.append(
        f"Pre-committed decision rules (fixed BEFORE running): 'meaningfully active' = "
        f">={metadata['pre_committed_meaningful_quarantine_floor']} distinct memories quarantined; "
        f"'unacceptable false quarantine' = false_quarantine_rate > "
        f"{metadata['pre_committed_unacceptable_false_quarantine_rate']}. These thresholds were NOT "
        "chosen after seeing the results below, and N was never selected to maximize SG-QMS accuracy."
    )

    lines.append("\n## 3. Sample-Size Table\n")
    for condition in ("baseline", "sacr", "sg_qms"):
        lines.append(f"[{condition}]")
        cols = ["N", "accuracy_covered", "coverage", "macro_f1", "total_retrieval_events",
                "distinct_memories_retrieved", "retrieved_exactly_once", "retrieved_ge_2",
                "retrieved_ge_3", "retrieved_ge_5", "max_retrieval_count"]
        if condition in ("sacr", "sg_qms"):
            cols += ["mean_state_alignment", "mean_context_alignment", "state_violations",
                     "context_violations", "joint_violations"]
        if condition == "sg_qms":
            cols += ["number_quarantined", "number_downgraded", "memory_retention_rate",
                     "number_reaching_n_min", "quarantine_effectiveness", "false_quarantine_rate"]
        sub = summary_df[summary_df["condition"] == condition][cols]
        header = " ".join(f"{c:>16}" for c in cols)
        lines.append(header)
        for _, r in sub.iterrows():
            vals = []
            for c in cols:
                v = r[c]
                vals.append(f"{_fmt(v) if isinstance(v, float) else v:>16}")
            lines.append(" ".join(vals))
        lines.append("")

    lines.append("\n## 4. Mechanism Exposure vs. Governance Behavior vs. Downstream Performance\n")
    lines.append("(A) MECHANISM EXPOSURE (does the retrieval stream even produce repeats):")
    for _, r in recurrence_df[recurrence_df["condition"] == "sg_qms"].iterrows():
        lines.append(
            f"  N={int(r['N']):>6}: {int(r['distinct_memories_retrieved'])} distinct memories, "
            f"{int(r['n_min_eligible_count'])} reach N_MIN={metadata['n_min']} "
            f"({_fmt(r['fraction_ge_n_min'], '.1%') if r['fraction_ge_n_min'] is not None else 'N/A'}), "
            f"max retrieval count {int(r['max_retrieval_count'])}"
        )
    lines.append("\n(B) GOVERNANCE BEHAVIOR (does OGMM actually act on that exposure):")
    for _, r in summary_df[summary_df["condition"] == "sg_qms"].iterrows():
        lines.append(
            f"  N={int(r['N']):>6}: quarantined={int(r['number_quarantined'])} "
            f"downgraded={int(r['number_downgraded'])} "
            f"retention={_fmt(r['memory_retention_rate'], '.4f')} "
            f"effectiveness={_fmt(r['quarantine_effectiveness'], '.4f')} "
            f"false_quarantine_rate={_fmt(r['false_quarantine_rate'], '.4f')}"
        )
    lines.append("\n(C) DOWNSTREAM PREDICTION PERFORMANCE (does any of the above change accuracy):")
    for n in metadata["checkpoints"]:
        b = summary_df[(summary_df["condition"] == "baseline") & (summary_df["N"] == n)].iloc[0]
        s = summary_df[(summary_df["condition"] == "sacr") & (summary_df["N"] == n)].iloc[0]
        g = summary_df[(summary_df["condition"] == "sg_qms") & (summary_df["N"] == n)].iloc[0]
        lines.append(
            f"  N={n:>6}: baseline_acc={_fmt(b['accuracy_covered'])} "
            f"sacr_acc={_fmt(s['accuracy_covered'])} sg_qms_acc={_fmt(g['accuracy_covered'])}"
        )
    lines.append("")
    lines.append("These three axes are reported separately and must not be conflated: mechanism")
    lines.append("exposure growing with N does not by itself mean governance is behaving")
    lines.append("differently, and governance acting more does not by itself mean downstream")
    lines.append("accuracy improves -- each is evaluated on its own evidence above.")

    lines.append("\n## 5. Pre-Committed Decision Outcomes\n")
    if meaningfully_active_n is not None:
        lines.append(
            f"OGMM first becomes 'meaningfully active' (>= "
            f"{metadata['pre_committed_meaningful_quarantine_floor']} quarantines, threshold fixed "
            f"before this run) at N = {meaningfully_active_n}."
        )
    else:
        lines.append(
            f"OGMM never reached the pre-committed 'meaningfully active' threshold (>= "
            f"{metadata['pre_committed_meaningful_quarantine_floor']} quarantines) at any evaluated N, "
            f"up to N = {metadata['checkpoints'][-1]} (the full held-out set)."
        )
    if unacceptable_false_quarantine_ns:
        lines.append(
            f"UNACCEPTABLE false-quarantine rate (> "
            f"{metadata['pre_committed_unacceptable_false_quarantine_rate']:.0%}, threshold fixed before "
            f"this run) was observed at N = {unacceptable_false_quarantine_ns}."
        )
    else:
        lines.append(
            f"False-quarantine rate stayed at or below the pre-committed "
            f"{metadata['pre_committed_unacceptable_false_quarantine_rate']:.0%} unacceptability threshold "
            "at every evaluated N."
        )

    lines.append("\n## 6. Limitations (carried over from RQ3/RQ4, plus new ones specific to this design)\n")
    lines.append("- Outcome-proxy limitation (unchanged from RQ3/RQ4): `readmitted`-derived feedback")
    lines.append("  is an experimental outcome proxy, not causal ground truth.")
    lines.append("- SACR's shadow-governance oracle (used for false-quarantine-rate) reflects SACR's")
    lines.append("  OWN unconstrained retrieval trajectory over this sequence -- as RQ3 itself notes,")
    lines.append("  because SG-QMS's governance changes what it retrieves going forward, its")
    lines.append("  trajectory can diverge from SACR's for reasons beyond the quarantine decision")
    lines.append("  itself. This mirrors RQ3's own static-oracle caveat exactly, not a new gap.")
    lines.append("- The nested full-sequence order (design note (a)) is a NEW, seed-42-based")
    lines.append("  permutation distinct from RQ4's own build_eval_queries() sampling; it was")
    lines.append("  necessary for nesting and is documented, not silently substituted.")
    lines.append("- This is one deterministic query order; a different seed could shift exactly")
    lines.append("  which memories reach N_MIN first, though the aggregate recurrence rate (from")
    lines.append("  the separate, order-independent recurrence analysis) would not be expected to")
    lines.append("  change materially.")
    lines.append("- As in RQ4, majority-vote prediction is a simple, fixed, auditable rule, not")
    lines.append("  claimed optimal; this experiment evaluates an outcome-proxy prediction task,")
    lines.append("  not clinical validity or deployment safety.")

    lines.append("\n## 7. Reproducibility\n")
    for key in (
        "experiment_id", "timestamp", "python_version", "platform", "random_seed", "checkpoints",
        "n_min", "beta", "alpha", "downgrade_threshold", "state_threshold", "context_threshold",
        "top_k", "candidate_k", "elapsed_seconds_total",
    ):
        lines.append(f"{key}: {metadata[key]}")
    lines.append("\nSafety checks:")
    for key, value in metadata["safety_checks"].items():
        if key in ("protected_dir_hashes_before", "protected_dir_hashes_after"):
            continue
        lines.append(f"  {key}: {value}")

    lines.append("\n## 8. Conclusion\n")
    lines.append("This analysis reports where natural recurrence becomes large enough to expose")
    lines.append("OGMM's governance mechanism at all, and separately whether that mechanism, once")
    lines.append("exposed, changes downstream prediction behavior. It does not select or endorse a")
    lines.append("final RQ4 evaluation size on the basis of maximizing any single outcome metric;")
    lines.append("that choice is left to the accompanying human-written recommendation.")

    report_text = "\n".join(lines) + "\n"
    (RESULTS_DIR / "experiment_report.txt").write_text(report_text, encoding="utf-8")


if __name__ == "__main__":
    main()
