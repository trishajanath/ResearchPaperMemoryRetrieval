"""Analysis-only memory-recurrence study (pre-OGMM), for a future RQ4 decision.

Question: does the dataset naturally give repeated exposure to the SAME
memory across many different held-out queries, or does SACR retrieval
scatter across a mostly-distinct set of memories? OGMM only has something
to govern (quarantine/downgrade based on repeated feedback) if the SAME
memory_id keeps getting retrieved. This script measures that directly,
before any governance runs, so governance cannot shape the answer.

Strict constraints, honored by construction:
    - Uses experiments_rq4.build_split() UNCHANGED -- the same frozen,
      deterministic, patient-level split RQ4 itself uses. Not reimplemented,
      not reseeded, not rebalanced.
    - Uses sacr.sacr_retrieve() UNCHANGED, called exactly the way RQ4's own
      "sacr" condition calls it: full memory_df as the bank (so precomputed
      embedding indices stay valid), frozen active_rules from RQ2
      (_load_frozen_active_rules, imported not recomputed), TOP_K/CANDIDATE_K
      imported from experiments.py/sacr.py, and the held-out patients'
      own memory_ids excluded via `excluded_memory_ids` for leakage
      prevention -- identical mechanism, identical set, as RQ4.
    - No governance/OGMM logic runs at all (no quality_score updates, no
      quarantine, no downgrade) -- this is deliberately the "sacr" condition
      only, never "sg_qms", so governance cannot alter recurrence counts.
    - Writes only to results/rq4_memory_recurrence_analysis/ -- never
      touches results/rq1, rq2, rq3, or rq4, and imports (never edits)
      experiments_rq4.py, sacr.py, experiments_rq3.py, columns.py.

Query volume: RQ4 itself caps evaluation at MAX_EVAL_QUERIES=300 for
computational tractability of its full three-condition, governance-tracking
pipeline. That cap is RQ4's own decision and is left untouched. This script
has no governance loop and no paired-condition comparison, so it is far
cheaper per query; it runs every held-out row (one query per held-out
encounter, not capped/sampled) to get the largest natural-recurrence signal
the frozen split can produce.

Run: python3 rq4_memory_recurrence_analysis.py
Output: results/rq4_memory_recurrence_analysis/report.json,
        results/rq4_memory_recurrence_analysis/report.txt,
        results/rq4_memory_recurrence_analysis/retrieval_counts.csv
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from columns import CONTEXT_COLS, EXPERIENCE_COLS, STATE_COLS
from sacr import CANDIDATE_K, sacr_retrieve
from experiments import TOP_K
from experiments_rq3 import _load_frozen_active_rules
from experiments_rq4 import build_split

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "rq4_memory_recurrence_analysis"
RQ3_METADATA_PATH = Path(__file__).resolve().parent / "results" / "rq3" / "experiment_metadata.json"
RQ4_MAX_EVAL_QUERIES = 300  # RQ4's own pre-committed cap, read here only for scale comparison
PROGRESS_EVERY = 1000


def _load_frozen_governance_params():
    """Read RQ3's own selected N_MIN/BETA back from its metadata (single
    source of truth), the same way experiments_rq4.py reads RQ2's threshold.
    N_MIN is the operationally relevant number here: it's the retrieval
    count OGMM itself requires before a memory is even eligible for
    quarantine/downgrade, so it's the right bar for "does recurrence exist",
    not an arbitrary cutoff invented for this script.
    """
    rq3_metadata = json.loads(RQ3_METADATA_PATH.read_text(encoding="utf-8"))
    return rq3_metadata["selected_n_min"], rq3_metadata["selected_beta"]


def run_recurrence_scan():
    print("Loading RQ4's own frozen patient-level split (build_split, unmodified) ...")
    _memory_pool_df, held_out_df, leakage_excluded_ids, split_info = build_split()
    active_rules, context_threshold = _load_frozen_active_rules()
    print(f"Frozen active_rules: {active_rules}")
    print(f"Held-out rows (one query per held-out encounter, uncapped): {len(held_out_df)}")

    eval_df = held_out_df.sort_values("memory_id").reset_index(drop=True)

    memory_counter = Counter()
    coverage_counts = []  # how many memories each query actually retrieved (<=TOP_K)
    n_queries = len(eval_df)
    start = time.time()

    for i, row in enumerate(eval_df.itertuples(index=False)):
        row_d = row._asdict()
        patient = {col: row_d[col] for col in STATE_COLS + CONTEXT_COLS + EXPERIENCE_COLS}
        retrieved = sacr_retrieve(
            patient, active_rules, top_k=TOP_K, candidate_k=CANDIDATE_K,
            excluded_memory_ids=leakage_excluded_ids,
        )
        ids = retrieved["memory_id"].tolist()
        coverage_counts.append(len(ids))
        memory_counter.update(ids)

        if (i + 1) % PROGRESS_EVERY == 0 or (i + 1) == n_queries:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            remaining = (n_queries - (i + 1)) / rate if rate > 0 else float("nan")
            print(
                f"  processed {i + 1}/{n_queries} queries "
                f"({elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining, "
                f"{len(memory_counter)} distinct memories retrieved so far)"
            )

    elapsed_total = time.time() - start
    print(f"Done: {n_queries} queries in {elapsed_total:.1f}s")

    return {
        "memory_counter": memory_counter,
        "coverage_counts": coverage_counts,
        "n_queries": n_queries,
        "split_info": split_info,
        "active_rules": active_rules,
        "context_threshold": context_threshold,
        "elapsed_seconds": elapsed_total,
    }


def summarize(scan_result: dict) -> dict:
    counter = scan_result["memory_counter"]
    coverage_counts = np.array(scan_result["coverage_counts"])
    counts = np.array(sorted(counter.values(), reverse=True))
    n_queries = scan_result["n_queries"]

    total_retrieval_events = int(counts.sum())
    distinct_memories = int(len(counts))

    n_min, beta = _load_frozen_governance_params()
    n_min_eligible_mask = counts >= n_min
    governance_cross_reference = {
        "source": "RQ3 frozen experiment_metadata.json (selected_n_min, selected_beta)",
        "n_min": int(n_min),
        "beta": float(beta),
        "distinct_memories_meeting_n_min": int(n_min_eligible_mask.sum()),
        "fraction_of_distinct_memories_meeting_n_min": (
            float(n_min_eligible_mask.sum() / distinct_memories) if distinct_memories else None
        ),
        "retrieval_events_from_n_min_eligible_memories": int(counts[n_min_eligible_mask].sum()),
        "fraction_of_events_from_n_min_eligible_memories": (
            float(counts[n_min_eligible_mask].sum() / total_retrieval_events) if total_retrieval_events else None
        ),
        "rq4_own_eval_cap": RQ4_MAX_EVAL_QUERIES,
        "this_run_queries": int(n_queries),
        "this_run_vs_rq4_cap_ratio": float(n_queries / RQ4_MAX_EVAL_QUERIES) if RQ4_MAX_EVAL_QUERIES else None,
        "note": (
            f"RQ4's own evaluation runs only {RQ4_MAX_EVAL_QUERIES} queries "
            f"({RQ4_MAX_EVAL_QUERIES / n_queries:.1%} of this run's volume). The recurrence "
            "measured here is an upper bound on what RQ4's own eval could exploit, not a "
            "claim about what RQ4 actually observed at its own, much smaller, query cap."
        ),
    }

    top_20 = counter.most_common(20)

    hist_edges = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 100, np.inf]
    hist_labels = []
    hist_values = []
    for lo, hi in zip(hist_edges[:-1], hist_edges[1:]):
        if np.isinf(hi):
            label = f">={int(lo)}"
            mask = counts >= lo
        else:
            label = f"{int(lo)}-{int(hi) - 1}"
            mask = (counts >= lo) & (counts < hi)
        hist_labels.append(label)
        hist_values.append(int(mask.sum()))

    summary = {
        "n_queries": int(n_queries),
        "total_retrieval_events": total_retrieval_events,
        "distinct_memories_retrieved": distinct_memories,
        "memories_retrieved_exactly_once": int((counts == 1).sum()),
        "memories_retrieved_more_than_once": int((counts > 1).sum()),
        "memories_retrieved_ge_3": int((counts >= 3).sum()),
        "memories_retrieved_ge_5": int((counts >= 5).sum()),
        "memories_retrieved_ge_10": int((counts >= 10).sum()),
        "memories_retrieved_ge_20": int((counts >= 20).sum()),
        "max_retrieval_count": int(counts.max()) if len(counts) else 0,
        "mean_retrieval_count": float(counts.mean()) if len(counts) else 0.0,
        "median_retrieval_count": float(np.median(counts)) if len(counts) else 0.0,
        "top_20_memories": [{"memory_id": mid, "retrieval_count": cnt} for mid, cnt in top_20],
        "retrieval_count_distribution": {
            "bin_labels": hist_labels,
            "bin_counts": hist_values,
        },
        "coverage": {
            "queries_with_full_top_k": int((coverage_counts == TOP_K).sum()),
            "queries_with_zero_retrieved": int((coverage_counts == 0).sum()),
            "mean_retrieved_per_query": float(coverage_counts.mean()) if len(coverage_counts) else 0.0,
        },
        "governance_cross_reference": governance_cross_reference,
    }
    return summary


POWER_LAW_FLOOR = 20  # the user's illustrative "real recurrence" example starts around this magnitude


def render_verdict(summary: dict) -> str:
    distinct = summary["distinct_memories_retrieved"]
    max_count = summary["max_retrieval_count"]
    once_fraction = (
        summary["memories_retrieved_exactly_once"] / distinct if distinct else 1.0
    )
    gcr = summary["governance_cross_reference"]
    n_min = gcr["n_min"]
    n_min_eligible = gcr["distinct_memories_meeting_n_min"]
    n_min_fraction = gcr["fraction_of_distinct_memories_meeting_n_min"] or 0.0
    event_fraction = gcr["fraction_of_events_from_n_min_eligible_memories"] or 0.0

    if distinct == 0:
        return "NO DATA: no memories were retrieved at all -- cannot assess recurrence."

    if once_fraction >= 0.90 and max_count < n_min:
        return (
            f"NO NATURAL RECURRENCE: {once_fraction:.1%} of the {distinct} distinct retrieved "
            f"memories were retrieved exactly once, and the single most-retrieved memory only hit "
            f"{max_count} -- below RQ3's own frozen N_MIN={n_min}. OGMM would never see enough "
            "repeats to act. Do not pretend the dataset supports a natural recurrence experiment; "
            "keep the existing RQ4 null finding or build an explicitly labeled controlled stress-test."
        )

    if max_count >= POWER_LAW_FLOOR:
        magnitude_note = (
            f"the single most-retrieved memory was retrieved {max_count} times, in the same "
            "order of magnitude as a genuine power-law recurrence pattern."
        )
    else:
        magnitude_note = (
            f"but the single most-retrieved memory was only retrieved {max_count} times out of "
            f"{summary['n_queries']} queries -- nowhere near a power-law-style hotspot pattern "
            "(no memory dominates; recurrence is mild and broadly spread, not concentrated)."
        )

    return (
        f"MODEST NATURAL RECURRENCE, GATED BY SCALE: {n_min_eligible} of {distinct} distinct "
        f"retrieved memories ({n_min_fraction:.1%}) meet RQ3's frozen governance precondition "
        f"(retrieved >= N_MIN={n_min} times), accounting for {event_fraction:.1%} of all retrieval "
        f"events at this run's volume ({summary['n_queries']} queries) -- {magnitude_note} "
        f"Critically, RQ4's own evaluation caps at {gcr['rq4_own_eval_cap']} queries, only "
        f"{gcr['rq4_own_eval_cap'] / summary['n_queries']:.1%} of the volume used here, so RQ4's actual "
        "eval almost certainly sees far less repeat exposure than this scan shows -- this is an "
        "upper bound on organic recurrence, not evidence that RQ4's own results already reflect it. "
        "If a memory-recurrence RQ4 protocol is built, it needs a query volume closer to this run's "
        "scale (or explicit oversampling of high-recurrence patients) rather than RQ4's current "
        "300-query cap, and should report recurrence as a modest, broadly-distributed effect -- not "
        "a hotspot/power-law effect, since none was observed."
    )


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    scan_result = run_recurrence_scan()
    summary = summarize(scan_result)
    verdict = render_verdict(summary)

    report = {
        "purpose": (
            "Analysis-only memory-recurrence study: measures how often the same "
            "memory_id is retrieved by SACR across a large, uncapped set of held-out "
            "queries, BEFORE any OGMM/governance runs, to determine whether the "
            "dataset naturally supports a memory-recurrence RQ4 protocol."
        ),
        "method": {
            "split_source": "experiments_rq4.build_split() (frozen, unmodified)",
            "retrieval_function": "sacr.sacr_retrieve() (frozen, unmodified)",
            "condition": "equivalent to RQ4's 'sacr' condition -- no governance/OGMM applied",
            "active_rules": scan_result["active_rules"],
            "context_threshold_source": "RQ2 frozen metadata",
            "top_k": TOP_K,
            "candidate_k": CANDIDATE_K,
            "leakage_prevention": "held-out patients' own memory_ids excluded via excluded_memory_ids, same as RQ4",
            "query_cap": "none (RQ4's own MAX_EVAL_QUERIES=300 cap is untouched and unused here)",
        },
        "split_info": scan_result["split_info"],
        "elapsed_seconds": scan_result["elapsed_seconds"],
        "summary": summary,
        "verdict": verdict,
    }
    (RESULTS_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    counts_df = pd.DataFrame(
        scan_result["memory_counter"].most_common(),
        columns=["memory_id", "retrieval_count"],
    )
    counts_df.to_csv(RESULTS_DIR / "retrieval_counts.csv", index=False)

    lines = []
    lines.append("RQ4 MEMORY-RECURRENCE ANALYSIS (analysis-only, pre-OGMM, no RQ1-4 changes)")
    lines.append("=" * 88)
    lines.append(f"Queries run: {summary['n_queries']} (one per held-out encounter, uncapped)")
    lines.append(f"Elapsed: {scan_result['elapsed_seconds']:.1f}s")
    lines.append(f"Total retrieval events (n_queries * up to top_k): {summary['total_retrieval_events']}")
    lines.append(f"Distinct memories retrieved: {summary['distinct_memories_retrieved']}")
    lines.append("")
    lines.append("-- Recurrence counts --")
    lines.append(f"  retrieved exactly once:  {summary['memories_retrieved_exactly_once']}")
    lines.append(f"  retrieved  > 1 time:     {summary['memories_retrieved_more_than_once']}")
    lines.append(f"  retrieved >= 3 times:    {summary['memories_retrieved_ge_3']}")
    lines.append(f"  retrieved >= 5 times:    {summary['memories_retrieved_ge_5']}")
    lines.append(f"  retrieved >= 10 times:   {summary['memories_retrieved_ge_10']}")
    lines.append(f"  retrieved >= 20 times:   {summary['memories_retrieved_ge_20']}")
    lines.append(f"  max retrieval count:     {summary['max_retrieval_count']}")
    lines.append(f"  mean retrieval count:    {summary['mean_retrieval_count']:.3f}")
    lines.append(f"  median retrieval count:  {summary['median_retrieval_count']:.3f}")
    lines.append("")
    lines.append("-- Retrieval count distribution (bucketed) --")
    for label, val in zip(
        summary["retrieval_count_distribution"]["bin_labels"],
        summary["retrieval_count_distribution"]["bin_counts"],
    ):
        lines.append(f"  {label:>8}: {val}")
    lines.append("")
    lines.append("-- Top 20 most-retrieved memories --")
    for entry in summary["top_20_memories"]:
        lines.append(f"  {entry['memory_id']}: {entry['retrieval_count']}")
    lines.append("")
    lines.append("-- Coverage --")
    lines.append(f"  queries retrieving full top_k ({TOP_K}): {summary['coverage']['queries_with_full_top_k']}")
    lines.append(f"  queries retrieving zero memories: {summary['coverage']['queries_with_zero_retrieved']}")
    lines.append(f"  mean retrieved per query: {summary['coverage']['mean_retrieved_per_query']:.3f}")
    lines.append("")
    gcr = summary["governance_cross_reference"]
    lines.append(f"-- Cross-reference against RQ3's frozen governance threshold (N_MIN={gcr['n_min']}, BETA={gcr['beta']}) --")
    lines.append(f"  distinct memories meeting N_MIN: {gcr['distinct_memories_meeting_n_min']} "
                 f"({gcr['fraction_of_distinct_memories_meeting_n_min']:.1%} of distinct retrieved memories)")
    lines.append(f"  retrieval events from N_MIN-eligible memories: {gcr['retrieval_events_from_n_min_eligible_memories']} "
                 f"({gcr['fraction_of_events_from_n_min_eligible_memories']:.1%} of all events)")
    lines.append(f"  RQ4's own eval cap: {gcr['rq4_own_eval_cap']} queries vs. this run's {gcr['this_run_queries']} "
                 f"({gcr['rq4_own_eval_cap'] / gcr['this_run_queries']:.1%} of this run's volume)")
    lines.append(f"  {gcr['note']}")
    lines.append("")
    lines.append("VERDICT")
    lines.append(verdict)
    (RESULTS_DIR / "report.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "\n".join(lines))
    print(f"\nWrote {RESULTS_DIR / 'report.json'}, {RESULTS_DIR / 'report.txt'}, "
          f"{RESULTS_DIR / 'retrieval_counts.csv'}")


if __name__ == "__main__":
    main()
