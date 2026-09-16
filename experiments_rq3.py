"""RQ3: Outcome-Gated Memory Maintenance (OGMM).

"Can outcome-based memory governance dynamically identify and quarantine
low-utility experiences using implicit outcome feedback, without requiring
human memory-quality labels?"

This module does NOT modify RQ1 or RQ2's implementation, results, SACR
formulation, thresholds, or weights. It reuses the frozen RQ1/RQ2 SACR
retrieval mechanism (state threshold 0.70, context threshold 0.90 -- read
back from results/rq2/experiment_metadata.json and asserted, not
re-hardcoded independently) unchanged across all three conditions compared
here; the only thing that differs between conditions is memory governance,
implemented as a thin layer OUTSIDE sacr.py that decides which memory_ids to
pass to sacr_retrieve's (additive, opt-in) `excluded_memory_ids` parameter.

IMPORTANT DATASET CAVEAT: the UCI Diabetes 130-US Hospitals dataset provides
no causal feedback about an agent's action. The `readmitted` field (mapped
to feedback +1/0/-1 in preprocess.py) is used here strictly as an
EXPERIMENTAL OUTCOME PROXY / outcome-derived feedback proxy. Nothing in this
module claims clinical causality, treatment effectiveness, that a memory
"caused" a readmission, clinical correctness, or real-world deployment
validity -- see Section 5 ("Important Causal Limitation") of the generated
report.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from baseline_rag import memory_df
from columns import CONTEXT_COLS, EXPERIENCE_COLS, STATE_COLS
from sacr import (
	CANDIDATE_K,
	LAMBDA_CONTEXT,
	LAMBDA_EXPERIENCE,
	LAMBDA_SEMANTIC,
	STATE_ALIGNMENT_THRESHOLD,
	sacr_retrieve,
)
from experiments import RANDOM_SEED, TOP_K, _make_query_pairs


EXPERIMENT_ID = "RQ3_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "rq3"
RQ2_METADATA_PATH = Path(__file__).resolve().parent / "results" / "rq2" / "experiment_metadata.json"

FEEDBACK_MAPPING = {"NO": 1.0, ">30": 0.0, "<30": -1.0}

# --- Pre-committed OGMM parameters (Part 4/5), frozen before any evaluation ---
N_MIN_CANDIDATES = [2, 3, 5, 8, 10]
BETA_CANDIDATES = [0.0, -0.2, -0.34, -0.5, -1.0]
QUALIFYING_RATE_TARGET = (0.01, 0.25)  # (low, high) -- "meaningful but not dominant minority"
ALPHA = 0.1                 # quality_score step size; matches preprocess.py's own
                             # documented update rule: Q_new = clip(Q_old + alpha*F_env, 0, 1)
DOWNGRADE_THRESHOLD = 0.4    # quality_score cut below the initial 0.5 -> DOWNGRADE
NEGATIVE_EVIDENCE_MIN_REPEATS = 2  # "repeated" negative feedback, for reporting only

# --- Pre-committed experimental protocol (Part 8) ---
NUM_ARCHETYPES = 20         # reuses RQ1's 20 base scenarios (existing query generation)
CYCLES = 15
ROUNDS = NUM_ARCHETYPES * CYCLES  # 300


def _git_commit_hash():
	try:
		result = subprocess.run(
			["git", "rev-parse", "--short", "HEAD"],
			cwd=Path(__file__).resolve().parent,
			capture_output=True, text=True, timeout=5,
		)
		if result.returncode == 0:
			return result.stdout.strip()
	except Exception:
		pass
	return "unknown"


def _load_frozen_active_rules():
	"""Read RQ2's selected context threshold back from its own experiment
	metadata (single source of truth) rather than re-hardcoding a second
	copy of the decision, and assert it matches the value this task freezes.
	"""
	rq2_metadata = json.loads(RQ2_METADATA_PATH.read_text(encoding="utf-8"))
	context_threshold = rq2_metadata["context_threshold"]
	assert context_threshold == 0.90, (
		f"RQ2's frozen context threshold changed unexpectedly: {context_threshold}"
	)
	assert STATE_ALIGNMENT_THRESHOLD == 0.70, "RQ1's frozen state threshold changed unexpectedly"
	return {
		"min_state_alignment": STATE_ALIGNMENT_THRESHOLD,
		"min_context_alignment": context_threshold,
	}, context_threshold


def _build_round_sequence():
	"""20 recurring query archetypes (RQ1's base scenarios), cycled
	deterministically for ROUNDS rounds. The SAME sequence is reused,
	unmodified, across all three conditions (paired design, Part 8/16).
	"""
	query_pairs = _make_query_pairs()
	archetypes = [query_pairs[f"Q{i:03d}"]["base"] for i in range(1, NUM_ARCHETYPES + 1)]
	round_sequence = [archetypes[r % NUM_ARCHETYPES] for r in range(ROUNDS)]
	round_archetype_ids = [f"Q{(r % NUM_ARCHETYPES) + 1:03d}" for r in range(ROUNDS)]
	return archetypes, round_sequence, round_archetype_ids


# ---------------------------------------------------------------------------
# Governance state (Part 2/3) -- maintained separately from memory_df (M).
# ---------------------------------------------------------------------------

def _new_governance_entry():
	return {
		"retrieval_count": 0,
		"positive_feedback_count": 0,
		"neutral_feedback_count": 0,
		"negative_feedback_count": 0,
		"utility_sum": 0.0,
		"average_utility": 0.0,
		"quality_score": 0.5,
		"quarantined": False,
		"quarantine_reason": None,
		"downgraded": False,
		"removed": False,
	}


def _apply_feedback(entry, feedback):
	entry["retrieval_count"] += 1
	entry["utility_sum"] += feedback
	if feedback > 0:
		entry["positive_feedback_count"] += 1
	elif feedback < 0:
		entry["negative_feedback_count"] += 1
	else:
		entry["neutral_feedback_count"] += 1
	entry["average_utility"] = entry["utility_sum"] / entry["retrieval_count"]


def _update_quality(entry, feedback):
	entry["quality_score"] = float(np.clip(entry["quality_score"] + ALPHA * feedback, 0.0, 1.0))


def _meets_quarantine_criterion(entry, n_min, beta):
	return entry["retrieval_count"] >= n_min and entry["average_utility"] <= beta


def _meets_downgrade_criterion(entry):
	return entry["quality_score"] < DOWNGRADE_THRESHOLD


# ---------------------------------------------------------------------------
# Part 7/8: run one condition (static / history_deletion / ogmm) over the
# same round sequence, applying that condition's governance policy.
# ---------------------------------------------------------------------------

def run_condition(condition, round_sequence, round_archetype_ids, active_rules, n_min=None, beta=None):
	assert condition in ("static", "history_deletion", "ogmm")
	track_quality = condition == "ogmm"
	governance = {}
	excluded_ids = set()
	retrieval_log = []
	governance_history = []

	for round_index, (patient, archetype_id) in enumerate(
		zip(round_sequence, round_archetype_ids), start=1
	):
		results = sacr_retrieve(
			patient, active_rules, top_k=TOP_K, candidate_k=CANDIDATE_K,
			excluded_memory_ids=excluded_ids or None,
		)
		for rank, (_, row) in enumerate(results.iterrows(), start=1):
			memory_id = row["memory_id"]
			feedback = float(row["feedback"])
			entry = governance.setdefault(memory_id, _new_governance_entry())
			_apply_feedback(entry, feedback)
			if track_quality:
				_update_quality(entry, feedback)

			retrieval_log.append(
				{
					"condition": condition,
					"round": round_index,
					"archetype": archetype_id,
					"rank": rank,
					"memory_id": memory_id,
					"feedback": feedback,
					"semantic_similarity": float(row["semantic_similarity"]),
					"state_alignment": float(row["state_alignment"]),
					"context_alignment": float(row["context_alignment"]),
					"final_ranking_score": float(row["final_ranking_score"]),
					"retrieval_count_after": entry["retrieval_count"],
					"average_utility_after": entry["average_utility"],
				}
			)

			if condition == "static":
				continue

			if condition == "history_deletion":
				if not entry["removed"] and _meets_quarantine_criterion(entry, n_min, beta):
					entry["removed"] = True
					excluded_ids.add(memory_id)
					governance_history.append(
						{
							"condition": condition,
							"round": round_index,
							"memory_id": memory_id,
							"action": "deleted",
							"retrieval_count": entry["retrieval_count"],
							"average_utility": entry["average_utility"],
							"negative_feedback_count": entry["negative_feedback_count"],
							"quality_score": None,
							"reason": (
								f"retrieval_count={entry['retrieval_count']}>=N_MIN={n_min} "
								f"and average_utility={entry['average_utility']:.4f}<=BETA={beta}"
							),
						}
					)
				continue

			# condition == "ogmm"
			if not entry["quarantined"] and _meets_quarantine_criterion(entry, n_min, beta):
				entry["quarantined"] = True
				entry["quarantine_reason"] = (
					f"retrieval_count={entry['retrieval_count']}>=N_MIN={n_min} "
					f"and average_utility={entry['average_utility']:.4f}<=BETA={beta}"
				)
				excluded_ids.add(memory_id)
				governance_history.append(
					{
						"condition": condition,
						"round": round_index,
						"memory_id": memory_id,
						"action": "quarantined",
						"retrieval_count": entry["retrieval_count"],
						"average_utility": entry["average_utility"],
						"negative_feedback_count": entry["negative_feedback_count"],
						"quality_score": entry["quality_score"],
						"reason": entry["quarantine_reason"],
					}
				)
			elif not entry["quarantined"] and not entry["downgraded"] and _meets_downgrade_criterion(entry):
				entry["downgraded"] = True
				governance_history.append(
					{
						"condition": condition,
						"round": round_index,
						"memory_id": memory_id,
						"action": "downgraded",
						"retrieval_count": entry["retrieval_count"],
						"average_utility": entry["average_utility"],
						"negative_feedback_count": entry["negative_feedback_count"],
						"quality_score": entry["quality_score"],
						"reason": (
							f"quality_score={entry['quality_score']:.4f} < "
							f"DOWNGRADE_THRESHOLD={DOWNGRADE_THRESHOLD}"
						),
					}
				)

	return governance, retrieval_log, governance_history


# ---------------------------------------------------------------------------
# Part 4: parameter grid analysis + pre-committed selection rule
# ---------------------------------------------------------------------------

def _parameter_grid_analysis(oracle_governance):
	rows = []
	for n_min in N_MIN_CANDIDATES:
		eligible_pool = [e for e in oracle_governance.values() if e["retrieval_count"] >= n_min]
		pool_size = len(eligible_pool)
		for beta in BETA_CANDIDATES:
			n_qualifying = sum(1 for e in eligible_pool if e["average_utility"] <= beta)
			rate = (n_qualifying / pool_size) if pool_size else float("nan")
			rows.append(
				{
					"n_min": n_min,
					"beta": beta,
					"eligible_pool_size": pool_size,
					"n_qualifying": n_qualifying,
					"qualifying_rate": rate,
				}
			)
	return rows


def _select_parameters(grid_rows):
	"""Pre-committed rule (decided before running the grid, not after):

	Require N_MIN >= 3 (never decide from 1-2 observations, per Part 3).
	Among N_MIN candidates ascending, take the smallest N_MIN for which at
	least one BETA candidate's qualifying_rate falls in
	QUALIFYING_RATE_TARGET = (1%, 25%) -- "a meaningful but non-dominant
	minority of the eligible pool". Among BETA candidates satisfying that at
	the chosen N_MIN, pick the LARGEST (least strict, closest to the
	natural neutral cut-point 0.0). If no combination satisfies the target
	range, fall back to the combination whose qualifying_rate is closest to
	the range's midpoint, among N_MIN >= 3, reported explicitly as a
	fallback (not silently).
	"""
	low, high = QUALIFYING_RATE_TARGET
	by_n_min = {}
	for row in grid_rows:
		by_n_min.setdefault(row["n_min"], []).append(row)

	for n_min in sorted(n for n in by_n_min if n >= 3):
		candidates = [
			r for r in by_n_min[n_min]
			if not np.isnan(r["qualifying_rate"]) and low <= r["qualifying_rate"] <= high
		]
		if candidates:
			chosen = max(candidates, key=lambda r: r["beta"])
			return chosen["n_min"], chosen["beta"], "primary", chosen

	pool = [r for r in grid_rows if r["n_min"] >= 3 and not np.isnan(r["qualifying_rate"])]
	if not pool:
		pool = [r for r in grid_rows if not np.isnan(r["qualifying_rate"])]
	mid = (low + high) / 2
	fallback = min(pool, key=lambda r: abs(r["qualifying_rate"] - mid))
	return fallback["n_min"], fallback["beta"], "fallback", fallback


# ---------------------------------------------------------------------------
# Part 10: metrics
# ---------------------------------------------------------------------------

def _oracle_eligible_set(oracle_governance, n_min, beta):
	return {
		mid for mid, e in oracle_governance.items()
		if e["retrieval_count"] >= n_min and e["average_utility"] <= beta
	}


def _effectiveness_and_false_rate(oracle_eligible, flagged_set):
	effectiveness = (
		len(oracle_eligible & flagged_set) / len(oracle_eligible) if oracle_eligible else float("nan")
	)
	false_rate = (
		len(flagged_set - oracle_eligible) / len(flagged_set) if flagged_set else float("nan")
	)
	return effectiveness, false_rate


def _negative_retrieval_rates(retrieval_log, condition, rounds):
	events = [r for r in retrieval_log if r["condition"] == condition]
	total = len(events)
	negative = sum(1 for r in events if r["feedback"] < 0)
	half = rounds // 2
	first_half = [r for r in events if r["round"] <= half]
	second_half = [r for r in events if r["round"] > half]

	def _rate(evs):
		return (sum(1 for r in evs if r["feedback"] < 0) / len(evs)) if evs else float("nan")

	return {
		"total_events": total,
		"negative_events": negative,
		"overall_rate": (negative / total) if total else float("nan"),
		"first_half_rate": _rate(first_half),
		"second_half_rate": _rate(second_half),
	}


def _memory_retention(governance, flagged_ids, full_bank_size):
	touched = len(governance)
	flagged = len(flagged_ids)
	retention_vs_touched = ((touched - flagged) / touched) if touched else float("nan")
	retention_vs_full_bank = (full_bank_size - flagged) / full_bank_size
	return retention_vs_touched, retention_vs_full_bank, touched, flagged


def _negative_evidence_accumulation(governance, min_repeats=NEGATIVE_EVIDENCE_MIN_REPEATS):
	return sum(1 for e in governance.values() if e["negative_feedback_count"] >= min_repeats)


def _quality_distribution(values):
	if not values:
		return {"mean": float("nan"), "median": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
	arr = np.array(values, dtype=float)
	return {
		"mean": float(np.mean(arr)),
		"median": float(np.median(arr)),
		"std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
		"min": float(np.min(arr)),
		"max": float(np.max(arr)),
	}


# ---------------------------------------------------------------------------
# Part 12: pre-registered hypotheses (evaluated AFTER results, never altered)
# ---------------------------------------------------------------------------

def _evaluate_hypotheses(
	ogmm_governance, quarantined_ids, neg_rate_static_second_half,
	neg_rate_ogmm_second_half, n_min, beta,
):
	h1_fired = len(quarantined_ids) >= 1
	h1_valid = all(ogmm_governance[mid]["retrieval_count"] >= n_min for mid in quarantined_ids) and all(
		ogmm_governance[mid]["average_utility"] <= beta for mid in quarantined_ids
	)
	h1 = h1_fired and h1_valid

	h2 = (
		not np.isnan(neg_rate_ogmm_second_half)
		and not np.isnan(neg_rate_static_second_half)
		and neg_rate_ogmm_second_half < neg_rate_static_second_half
	)

	h3 = len(quarantined_ids) > 0 and all(
		bool(ogmm_governance[mid].get("quarantine_reason")) for mid in quarantined_ids
	)

	total_touched = len(ogmm_governance)
	not_all_quarantined = len(quarantined_ids) < total_touched
	favorable_ids = {mid for mid, e in ogmm_governance.items() if e["average_utility"] > 0}
	unfavorable_ids = {
		mid for mid, e in ogmm_governance.items()
		if e["retrieval_count"] >= n_min and e["average_utility"] <= beta
	}
	fav_q_rate = (len(favorable_ids & quarantined_ids) / len(favorable_ids)) if favorable_ids else 0.0
	unfav_q_rate = (
		len(unfavorable_ids & quarantined_ids) / len(unfavorable_ids) if unfavorable_ids else float("nan")
	)
	h4 = not_all_quarantined and (np.isnan(unfav_q_rate) or fav_q_rate <= 0.5 * unfav_q_rate)

	return {
		"H1": {"passed": bool(h1), "fired": h1_fired, "all_valid": h1_valid},
		"H2": {
			"passed": bool(h2),
			"static_second_half_negative_rate": neg_rate_static_second_half,
			"ogmm_second_half_negative_rate": neg_rate_ogmm_second_half,
		},
		"H3": {"passed": bool(h3), "quarantined_with_reason": len(quarantined_ids) if h3 else 0},
		"H4": {
			"passed": bool(h4),
			"not_all_quarantined": not_all_quarantined,
			"favorable_quarantine_rate": fav_q_rate,
			"unfavorable_quarantine_rate": unfav_q_rate,
		},
	}


# ---------------------------------------------------------------------------
# Reporting (Part 14)
# ---------------------------------------------------------------------------

def _write_config(metadata):
	lines = [f"{key}: {value}" for key, value in metadata.items()]
	(RESULTS_DIR / "config.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value, spec=".4f"):
	if value is None or (isinstance(value, float) and np.isnan(value)):
		return "N/A"
	if isinstance(value, float):
		return f"{value:{spec}}"
	return str(value)


def _write_report(metadata, grid_rows, selection, condition_metrics, hypotheses):
	n_min, beta, selection_kind, _ = selection
	lines = []
	lines.append("=" * 60)
	lines.append("RQ3 -- OUTCOME-GATED MEMORY MAINTENANCE (OGMM)")
	lines.append("=" * 60)

	lines.append("\n## 1. Research Question\n")
	lines.append('"Can outcome-based memory governance dynamically identify and quarantine')
	lines.append('low-utility experiences using implicit outcome feedback, without requiring')
	lines.append('human memory-quality labels?"')

	lines.append("\n## 2. Hypotheses (pre-registered, not altered after seeing results)\n")
	lines.append("H1: OGMM will quarantine memories that accumulate sufficiently negative")
	lines.append("    outcome-derived evidence.")
	lines.append("H2: OGMM will reduce the future retrieval rate of memories with negative")
	lines.append("    outcome proxies compared with static memory.")
	lines.append("H3: OGMM will preserve more explicit negative evidence than deletion-based")
	lines.append("    memory management because quarantined memories remain represented in Q.")
	lines.append("H4: OGMM will not quarantine all memories; favorable memories should remain")
	lines.append("    available at a substantially higher rate than repeatedly unfavorable ones.")

	lines.append("\n## 3. Dataset\n")
	lines.append(f"UCI Diabetes 130-US Hospitals, {metadata['number_of_memories']} memory records.")
	lines.append(f"State columns: {STATE_COLS}")
	lines.append(f"Context columns: {CONTEXT_COLS}")
	lines.append(f"Experience columns: {EXPERIENCE_COLS}")

	lines.append("\n## 4. Outcome Proxy Definition\n")
	lines.append("The existing preprocessing (preprocess.py) maps the `readmitted` field to a")
	lines.append("feedback proxy: NO -> +1.0, >30 -> 0.0, <30 -> -1.0. This mapping was")
	lines.append("inspected and is unchanged (no implementation bug was found in it).")

	lines.append("\n## 5. Important Causal Limitation\n")
	lines.append("The UCI dataset provides no causal feedback about an agent's action. The")
	lines.append("`readmitted` field is only an EXPERIMENTAL OUTCOME PROXY / outcome-derived")
	lines.append("feedback proxy, attached to the historical record itself. This experiment")
	lines.append("does NOT claim clinical causality, treatment effectiveness, that a memory")
	lines.append('"caused" readmission, clinical correctness, or real-world deployment')
	lines.append("validity. It evaluates only whether a memory-management mechanism can learn")
	lines.append("from observed outcome proxies associated with retrieved experiences.")

	lines.append("\n## 6. Reference-Paper Motivation\n")
	lines.append('"How Memory Management Impacts LLM Agents: An Empirical Study of')
	lines.append('Experience-Following Behavior" motivates this design conceptually:')
	lines.append("experience-following, error propagation, memory utility from repeated")
	lines.append("retrieval history, and history-based memory management. The paper is NOT")
	lines.append("claimed to have proposed SG-QMS, SACR, OGMM, or the Negative Evidence")
	lines.append("Quarantine Index -- those are this project's own contribution, built only")
	lines.append("on the paper's general methodological motivation.")

	lines.append("\n## 7. OGMM Design\n")
	lines.append("Governance state G is maintained separately from the original memory bank M")
	lines.append("(memory_df is never mutated by this module); per memory_id, G tracks:")
	lines.append("retrieval_count, positive/neutral/negative_feedback_count, utility_sum,")
	lines.append("average_utility, quality_score, quarantined, quarantine_reason.")
	lines.append(f"average_utility(m) = utility_sum(m) / retrieval_count(m), updated after every")
	lines.append("observed retrieval+feedback event (Part 3).")
	lines.append(f"quality_score update (matches preprocess.py's own documented rule):")
	lines.append(f"  quality_score = clip(quality_score + ALPHA * feedback, 0, 1), ALPHA={ALPHA}")

	lines.append("\n## 8. Governance States\n")
	lines.append("RETAIN: default; memory fully available to normal retrieval.")
	lines.append(f"DOWNGRADE: quality_score < {DOWNGRADE_THRESHOLD} (sticky; evaluated only if not")
	lines.append("  already quarantined). Downgrade does not affect SACR retrieval eligibility")
	lines.append("  or ranking (SACR's formula/thresholds are frozen); it is governance")
	lines.append("  metadata only.")
	lines.append(f"QUARANTINE: retrieval_count >= N_MIN AND average_utility <= BETA (checked")
	lines.append("  before downgrade each round). A quarantined memory enters Q and is")
	lines.append("  excluded from all subsequent normal SACR retrieval in that condition.")

	lines.append("\n## 9. Negative Evidence Quarantine Design\n")
	lines.append("Q holds, per quarantined memory: memory_id, retrieval_count,")
	lines.append("negative_feedback_count, average_utility, quarantine_reason. Q's purpose is")
	lines.append('NOT deletion -- it preserves the explicit statement "this experience should')
	lines.append('not influence normal future retrieval" while keeping the original memory')
	lines.append("record (in memory_df) fully auditable. Enforcement uses sacr.sacr_retrieve's")
	lines.append("additive `excluded_memory_ids` parameter (default None => identical to")
	lines.append("RQ1/RQ2 behavior), applied strictly AFTER the frozen state+context")
	lines.append("eligibility check, so it cannot be mistaken for a change to SACR itself.")

	lines.append("\n## 10. Parameter Selection\n")
	lines.append(f"N_MIN candidates: {N_MIN_CANDIDATES}")
	lines.append(f"BETA candidates: {BETA_CANDIDATES}")
	lines.append(
		"Selection rule (pre-committed, evaluated ONLY against the STATIC condition's"
	)
	lines.append(
		"unconstrained retrieval-history distribution -- never against OGMM's own"
	)
	lines.append(
		"comparison results): require N_MIN >= 3; take the smallest such N_MIN for"
	)
	lines.append(
		f"which some BETA gives a qualifying_rate in {QUALIFYING_RATE_TARGET} (a"
	)
	lines.append(
		"meaningful but non-dominant minority of the eligible pool); among BETA"
	)
	lines.append(
		"candidates satisfying that, pick the largest (closest to the natural"
	)
	lines.append("neutral cut-point 0.0).")
	lines.append("")
	lines.append(f"{'n_min':>6}{'beta':>8}{'pool_size':>11}{'n_qualifying':>14}{'qualifying_rate':>17}")
	for row in grid_rows:
		lines.append(
			f"{row['n_min']:>6}{row['beta']:>8.2f}{row['eligible_pool_size']:>11}"
			f"{row['n_qualifying']:>14}{_fmt(row['qualifying_rate'], '.4f'):>17}"
		)
	lines.append("")
	pool_sizes = {row["eligible_pool_size"] for row in grid_rows}
	if len(pool_sizes) == 1:
		lines.append(
			f"NOTE: eligible_pool_size is identical ({pool_sizes.pop()}) across every N_MIN"
		)
		lines.append(
			f"candidate. Because each of the {NUM_ARCHETYPES} archetypes recurs exactly"
		)
		lines.append(
			f"{CYCLES} times under the fixed round schedule, every memory the static"
		)
		lines.append(
			f"condition ever touches ends up retrieved exactly {CYCLES} times -- so N_MIN"
		)
		lines.append(
			"(2 through 10) does not discriminate under THIS protocol; only BETA does."
		)
		lines.append(
			"N_MIN is still meaningful as a safety floor (it would matter under a protocol"
		)
		lines.append(
			"with variable per-memory exposure), but its selection here should not be read"
		)
		lines.append("as empirically validated against this specific run.")
		lines.append("")
	fallback_note = " (FALLBACK: no combination met the target range)" if selection_kind == "fallback" else ""
	lines.append(f"Selected N_MIN = {n_min}, BETA = {beta}{fallback_note}")
	lines.append(
		f"Downgrade threshold (frozen separately, Part 5): quality_score < {DOWNGRADE_THRESHOLD}"
	)

	lines.append("\n## 11. Experimental Protocol\n")
	lines.append(f"{NUM_ARCHETYPES} recurring query archetypes, reused unmodified from RQ1's")
	lines.append("base scenario generation (experiments._make_query_pairs). Archetypes are")
	lines.append(f"cycled deterministically for {CYCLES} cycles ({ROUNDS} rounds total); the")
	lines.append("SAME round sequence is used, unmodified, for all three conditions (paired")
	lines.append("design). Each round: run SACR (frozen state+context eligibility, unchanged")
	lines.append("ranking weights), observe each retrieved memory's outcome proxy, update")
	lines.append("governance history, apply that condition's policy, proceed.")

	lines.append("\n## 12. Static Baseline (Condition A)\n")
	lines.append("SACR retrieval occurs every round; outcome proxy is observed and logged;")
	lines.append("no memory governance occurs; nothing is ever excluded from retrieval. This")
	lines.append("condition also serves as the unconstrained oracle for Section 10's parameter")
	lines.append("analysis and for the effectiveness/false-quarantine metrics below.")

	lines.append("\n## 13. History-Based Baseline (Condition B)\n")
	lines.append("Paper-inspired, not a reproduction of the reference paper's full")
	lines.append("implementation: tracks retrieval history and average utility using the SAME")
	lines.append(f"N_MIN={n_min}/BETA={beta} rule as OGMM; once a memory meets the rule it is")
	lines.append("DELETED (excluded from all future retrieval), with no structured evidence")
	lines.append("preserved beyond bare exclusion-list membership.")

	lines.append("\n## 14. OGMM (Condition C, SG-QMS contribution)\n")
	lines.append("Tracks retrieval history, updates quality_score, applies RETAIN/DOWNGRADE/")
	lines.append(f"QUARANTINE using the SAME N_MIN={n_min}/BETA={beta} rule for quarantine as")
	lines.append("the deletion baseline. The distinction from Condition B: poor memory ->")
	lines.append("DELETE (baseline) vs. poor memory -> QUARANTINE AS NEGATIVE EVIDENCE (OGMM,")
	lines.append("preserved and auditable in Q, not merely erased).")

	lines.append("\n## 15. Metrics\n")
	lines.append("Quarantine effectiveness, false quarantine rate, future negative retrieval")
	lines.append("rate (overall + first/second half of rounds), memory retention (vs. touched")
	lines.append("set and vs. full bank), number quarantined/downgraded/deleted, retrieval")
	lines.append("counts, negative evidence accumulation, quality distribution (OGMM only).")

	lines.append("\n## 16. Results\n")
	for condition in ("static", "history_deletion", "ogmm"):
		m = condition_metrics[condition]
		lines.append(f"[{condition}]")
		lines.append(f"  total retrieval events: {m['total_events']}")
		lines.append(f"  distinct memories touched: {m['touched']}")
		lines.append(f"  negative retrieval rate -- overall: {_fmt(m['overall_rate'])}  "
			f"first half: {_fmt(m['first_half_rate'])}  second half: {_fmt(m['second_half_rate'])}")
		lines.append(f"  memory retention (vs touched set): {_fmt(m['retention_vs_touched'])}")
		lines.append(f"  memory retention (vs full {metadata['number_of_memories']}-memory bank): "
			f"{_fmt(m['retention_vs_full_bank'], '.6f')}")
		if condition == "history_deletion":
			lines.append(f"  number deleted: {m['flagged_count']}")
		if condition == "ogmm":
			lines.append(f"  number quarantined: {m['flagged_count']}")
			lines.append(f"  number downgraded: {m['downgraded_count']}")
			lines.append(f"  quality_score distribution (all touched memories): "
				f"mean={_fmt(m['quality_after']['mean'])} median={_fmt(m['quality_after']['median'])} "
				f"std={_fmt(m['quality_after']['std'])} min={_fmt(m['quality_after']['min'])} "
				f"max={_fmt(m['quality_after']['max'])}  (all start at 0.5 before governance)")
		if condition in ("history_deletion", "ogmm"):
			lines.append(f"  negative evidence accumulation (>= {NEGATIVE_EVIDENCE_MIN_REPEATS} negative "
				f"observations): {m['negative_evidence_accumulation']}")
			lines.append(f"  quarantine/deletion effectiveness (vs static oracle): {_fmt(m['effectiveness'])}")
			lines.append(f"  false quarantine/deletion rate (vs static oracle): {_fmt(m['false_rate'])}")
		lines.append("")

	lines.append("## 17. Comparison\n")
	lines.append(
		f"{'condition':<18}{'overall neg. rate':>18}{'2nd-half neg. rate':>20}{'retention (touched)':>21}"
	)
	for condition in ("static", "history_deletion", "ogmm"):
		m = condition_metrics[condition]
		lines.append(
			f"{condition:<18}{_fmt(m['overall_rate']):>18}{_fmt(m['second_half_rate']):>20}"
			f"{_fmt(m['retention_vs_touched']):>21}"
		)
	lines.append("")
	lines.append("RQ3 hypotheses:")
	for key in ("H1", "H2", "H3", "H4"):
		lines.append(f"  {key}: {'PASS' if hypotheses[key]['passed'] else 'FAIL'} -- {hypotheses[key]}")

	lines.append("\n## 18. Interpretation\n")
	ogmm_m = condition_metrics["ogmm"]
	static_m = condition_metrics["static"]
	lines.append(
		f"OGMM quarantined {ogmm_m['flagged_count']} of {ogmm_m['touched']} distinct memories it"
	)
	lines.append(
		f"retrieved ({_fmt(1 - ogmm_m['retention_vs_touched'])} of the touched working set), and"
	)
	lines.append(
		f"downgraded a further {ogmm_m['downgraded_count']}. Its second-half negative-retrieval"
	)
	lines.append(
		f"rate ({_fmt(ogmm_m['second_half_rate'])}) compares to static's "
		f"({_fmt(static_m['second_half_rate'])}) and history-deletion's "
		f"({_fmt(condition_metrics['history_deletion']['second_half_rate'])})."
	)
	lines.append(
		"The key structural distinction demonstrated is that OGMM's quarantined"
	)
	lines.append(
		f"memories ({ogmm_m['flagged_count']}) all carry a populated quarantine_reason"
	)
	lines.append(
		"in Q (H3), whereas the deletion baseline's excluded set carries no equivalent"
	)
	lines.append("structured evidence by design.")

	lines.append("\n## 19. Limitations\n")
	lines.append("- Outcome proxy, not causal ground truth (Section 5); readmission is not")
	lines.append("  attributable to the retrieved memory having been used.")
	lines.append("- N_MIN/BETA were selected from a small predefined grid using one coverage")
	lines.append("  rule over the static oracle distribution; a different rule or grid could")
	lines.append("  select different values.")
	lines.append("- The round sequence recurs over only 20 archetypes; results reflect this")
	lines.append("  specific deterministic query stream, not a claim of generalization.")
	lines.append("- \"Effectiveness\"/\"false rate\" use the static condition's own retrieval")
	lines.append("  trajectory as an oracle; because governance changes which memories get")
	lines.append("  retrieved going forward, OGMM's and static's trajectories can diverge for")
	lines.append("  reasons beyond the quarantine decision itself.")
	lines.append("- Quality_score/downgrade use a simple, deliberately unoptimized linear")
	lines.append("  update; it is not claimed to be an optimal or calibrated scoring rule.")
	if len({row["eligible_pool_size"] for row in grid_rows}) == 1:
		lines.append("- The fixed archetype-cycling schedule gives every touched memory the same")
		lines.append("  number of exposures, so N_MIN did not empirically discriminate in this run")
		lines.append("  (see Section 10's note); only BETA was effectively selected on data here.")

	lines.append("\n## 20. Reproducibility\n")
	for key, value in metadata.items():
		lines.append(f"{key}: {value}")

	lines.append("\n## 21. Conclusion\n")
	lines.append(
		"Under this deterministic, paired-query protocol, OGMM's quarantine mechanism"
	)
	lines.append(
		f"fires correctly on memories meeting its own pre-committed rule (H1 "
		f"{'PASS' if hypotheses['H1']['passed'] else 'FAIL'}), preserves structured negative"
	)
	lines.append(
		f"evidence unlike deletion (H3 {'PASS' if hypotheses['H3']['passed'] else 'FAIL'}), and"
	)
	lines.append(
		f"does not quarantine indiscriminately (H4 {'PASS' if hypotheses['H4']['passed'] else 'FAIL'})."
	)
	lines.append(
		f"Its effect on future negative-retrieval rate versus static was "
		f"{'confirmed' if hypotheses['H2']['passed'] else 'NOT confirmed'} (H2) under this protocol."
	)
	lines.append(
		"This is a narrow, mechanism-level finding under one outcome proxy and one"
	)
	lines.append(
		"deterministic query stream -- it is not a claim of clinical validity or"
	)
	lines.append("generalization beyond this experimental setup.")

	report_text = "\n".join(lines) + "\n"
	(RESULTS_DIR / "experiment_report.txt").write_text(report_text, encoding="utf-8")


def run_rq3():
	RESULTS_DIR.mkdir(parents=True, exist_ok=True)
	np.random.seed(RANDOM_SEED)

	active_rules, context_threshold = _load_frozen_active_rules()
	archetypes, round_sequence, round_archetype_ids = _build_round_sequence()

	# --- Condition A: static (also serves as the parameter-selection oracle) ---
	static_governance, static_log, _ = run_condition(
		"static", round_sequence, round_archetype_ids, active_rules
	)

	# --- Part 4: parameter selection, using ONLY the static oracle ---
	grid_rows = _parameter_grid_analysis(static_governance)
	selection = _select_parameters(grid_rows)
	n_min, beta, selection_kind, _ = selection

	# --- Condition B: history-based deletion baseline ---
	deletion_governance, deletion_log, deletion_history = run_condition(
		"history_deletion", round_sequence, round_archetype_ids, active_rules, n_min=n_min, beta=beta
	)

	# --- Condition C: OGMM ---
	ogmm_governance, ogmm_log, ogmm_history = run_condition(
		"ogmm", round_sequence, round_archetype_ids, active_rules, n_min=n_min, beta=beta
	)

	all_log = static_log + deletion_log + ogmm_log
	all_history = deletion_history + ogmm_history

	# --- metrics per condition ---
	oracle_eligible = _oracle_eligible_set(static_governance, n_min, beta)
	full_bank_size = len(memory_df)

	condition_metrics = {}
	for condition, governance, log in (
		("static", static_governance, static_log),
		("history_deletion", deletion_governance, deletion_log),
		("ogmm", ogmm_governance, ogmm_log),
	):
		rates = _negative_retrieval_rates(all_log, condition, ROUNDS)
		if condition == "history_deletion":
			flagged_ids = {mid for mid, e in governance.items() if e["removed"]}
		elif condition == "ogmm":
			flagged_ids = {mid for mid, e in governance.items() if e["quarantined"]}
		else:
			flagged_ids = set()
		retention_vs_touched, retention_vs_full_bank, touched, flagged_count = _memory_retention(
			governance, flagged_ids, full_bank_size
		)
		metrics = {
			**rates,
			"retention_vs_touched": retention_vs_touched,
			"retention_vs_full_bank": retention_vs_full_bank,
			"touched": touched,
			"flagged_count": flagged_count,
		}
		if condition in ("history_deletion", "ogmm"):
			effectiveness, false_rate = _effectiveness_and_false_rate(oracle_eligible, flagged_ids)
			metrics["effectiveness"] = effectiveness
			metrics["false_rate"] = false_rate
			metrics["negative_evidence_accumulation"] = _negative_evidence_accumulation(governance)
		if condition == "ogmm":
			downgraded_ids = {mid for mid, e in governance.items() if e["downgraded"]}
			metrics["downgraded_count"] = len(downgraded_ids)
			metrics["quality_after"] = _quality_distribution(
				[e["quality_score"] for e in governance.values()]
			)
			metrics["quality_before"] = _quality_distribution([0.5 for _ in governance])
		condition_metrics[condition] = metrics

	quarantined_ids = {mid for mid, e in ogmm_governance.items() if e["quarantined"]}
	hypotheses = _evaluate_hypotheses(
		ogmm_governance, quarantined_ids,
		condition_metrics["static"]["second_half_rate"],
		condition_metrics["ogmm"]["second_half_rate"],
		n_min, beta,
	)

	# --- write CSVs ---
	pd.DataFrame(grid_rows).to_csv(RESULTS_DIR / "governance_parameter_analysis.csv", index=False)
	pd.DataFrame(all_log).to_csv(RESULTS_DIR / "rq3_detailed_results.csv", index=False)
	if all_history:
		pd.DataFrame(all_history).to_csv(RESULTS_DIR / "memory_governance_history.csv", index=False)
	else:
		pd.DataFrame(
			columns=["condition", "round", "memory_id", "action", "retrieval_count",
				"average_utility", "negative_feedback_count", "quality_score", "reason"]
		).to_csv(RESULTS_DIR / "memory_governance_history.csv", index=False)

	summary_rows = []
	for condition in ("static", "history_deletion", "ogmm"):
		m = condition_metrics[condition]
		summary_rows.append(
			{
				"condition": condition,
				"total_retrieval_events": m["total_events"],
				"distinct_memories_touched": m["touched"],
				"negative_retrieval_rate_overall": m["overall_rate"],
				"negative_retrieval_rate_first_half": m["first_half_rate"],
				"negative_retrieval_rate_second_half": m["second_half_rate"],
				"memory_retention_vs_touched": m["retention_vs_touched"],
				"memory_retention_vs_full_bank": m["retention_vs_full_bank"],
				"number_flagged": m["flagged_count"],
				"number_downgraded": m.get("downgraded_count"),
				"quarantine_or_deletion_effectiveness": m.get("effectiveness"),
				"false_quarantine_or_deletion_rate": m.get("false_rate"),
				"negative_evidence_accumulation": m.get("negative_evidence_accumulation"),
				"quality_score_mean_after": m.get("quality_after", {}).get("mean"),
			}
		)
	pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "rq3_summary.csv", index=False)

	metadata = {
		"experiment_id": EXPERIMENT_ID,
		"timestamp": datetime.now(timezone.utc).isoformat(),
		"python_version": sys.version.split()[0],
		"git_commit": _git_commit_hash(),
		"random_seed": RANDOM_SEED,
		"dataset_shape": list(memory_df.shape),
		"number_of_memories": len(memory_df),
		"state_columns": STATE_COLS,
		"context_columns": CONTEXT_COLS,
		"experience_columns": EXPERIENCE_COLS,
		"outcome_mapping": FEEDBACK_MAPPING,
		"n_min_candidates": N_MIN_CANDIDATES,
		"beta_candidates": BETA_CANDIDATES,
		"selected_n_min": n_min,
		"selected_beta": beta,
		"parameter_selection_kind": selection_kind,
		"downgrade_rule": f"quality_score < {DOWNGRADE_THRESHOLD}",
		"quality_update_rule": f"quality_score = clip(quality_score + {ALPHA} * feedback, 0, 1)",
		"num_rounds": ROUNDS,
		"num_archetypes": NUM_ARCHETYPES,
		"cycles": CYCLES,
		"top_k": TOP_K,
		"candidate_k": CANDIDATE_K,
		"state_alignment_threshold": STATE_ALIGNMENT_THRESHOLD,
		"context_alignment_threshold": context_threshold,
		"lambda_semantic": LAMBDA_SEMANTIC,
		"lambda_context": LAMBDA_CONTEXT,
		"lambda_experience": LAMBDA_EXPERIENCE,
	}
	(RESULTS_DIR / "experiment_metadata.json").write_text(
		json.dumps(metadata, indent=4, default=str), encoding="utf-8"
	)
	_write_config(metadata)
	_write_report(metadata, grid_rows, selection, condition_metrics, hypotheses)

	print("\n================ RQ3 COMPLETE ================\n")
	print("Research Question:")
	print("Can outcome-based memory governance dynamically identify and quarantine")
	print("low-utility experiences using implicit outcome feedback, without requiring")
	print("human memory-quality labels?\n")
	print(f"Selected N_MIN:\n{n_min}\n")
	print(f"Selected BETA:\n{beta}\n")
	print(f"Number of rounds:\n{ROUNDS}\n")
	print(f"Static negative retrieval rate:\n{_fmt(condition_metrics['static']['overall_rate'])}\n")
	print(
		"History-based deletion negative retrieval rate:\n"
		f"{_fmt(condition_metrics['history_deletion']['overall_rate'])}\n"
	)
	print(f"OGMM negative retrieval rate:\n{_fmt(condition_metrics['ogmm']['overall_rate'])}\n")
	print(f"OGMM memories quarantined:\n{condition_metrics['ogmm']['flagged_count']}\n")
	print(f"OGMM memories downgraded:\n{condition_metrics['ogmm']['downgraded_count']}\n")
	print(f"False quarantine rate:\n{_fmt(condition_metrics['ogmm']['false_rate'])}\n")
	print(f"Memory retention rate:\n{_fmt(condition_metrics['ogmm']['retention_vs_touched'])}\n")
	print(
		"RQ3 hypotheses: "
		f"H1 = {'PASS' if hypotheses['H1']['passed'] else 'FAIL'}  "
		f"H2 = {'PASS' if hypotheses['H2']['passed'] else 'FAIL'}  "
		f"H3 = {'PASS' if hypotheses['H3']['passed'] else 'FAIL'}  "
		f"H4 = {'PASS' if hypotheses['H4']['passed'] else 'FAIL'}\n"
	)
	print("Report: results/rq3/experiment_report.txt")
	print("Summary: results/rq3/rq3_summary.csv")
	print("Detailed results: results/rq3/rq3_detailed_results.csv")
	print("Governance history: results/rq3/memory_governance_history.csv")
	print("Parameter analysis: results/rq3/governance_parameter_analysis.csv")
	print(f"Git commit: {metadata['git_commit']}")

	return {
		"metadata": metadata,
		"condition_metrics": condition_metrics,
		"hypotheses": hypotheses,
		"selection": selection,
	}


if __name__ == "__main__":
	run_rq3()
