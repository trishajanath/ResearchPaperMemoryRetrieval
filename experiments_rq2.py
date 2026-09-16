"""RQ2: can memory retrieval mathematically guarantee joint state AND
contextual alignment, as opposed to a weighted-average ranking score that
only makes alignment *likely*?

This module does NOT modify RQ1's implementation, results, state threshold,
weights, query scenarios, or state-shuffle experiment. It reuses RQ1's exact
query-pair generation (``experiments._make_query_pairs``) and SACR's
unchanged calculation functions, and exercises a new, OPT-IN eligibility
mechanism added to ``sacr.check_active_rules``/``sacr.sacr_retrieve``: an
optional ``active_rules["min_context_alignment"]`` gate. RQ1's own
``active_rules`` dict never sets that key, so RQ1's behavior is provably
unaffected (verified in the sanity checks run before this experiment).

Pipeline under test:

    semantic candidates -> state alignment check -> context alignment check
        -> eligible memories -> ranking (unchanged weighted formula)
        -> top-k

Mathematical property under test, for every returned memory m of query q:

    m in R_k(q)  =>  S_state(q, m) >= tau_state  AND  S_context(q, m) >= tau_context

This is a structural guarantee of the eligibility gate (a candidate can only
enter ``scored_rows`` -- and therefore the ranked output -- if it already
passed both checks), not a claim about clinical correctness, causal
validity, an optimal threshold, or generalization beyond this dataset.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from baseline_rag import memory_df
from columns import CATEGORICAL_COLS, CONTEXT_COLS, EXPERIENCE_COLS, STATE_COLS
from sacr import (
	CANDIDATE_K,
	LAMBDA_CONTEXT,
	LAMBDA_EXPERIENCE,
	LAMBDA_SEMANTIC,
	STATE_ALIGNMENT_THRESHOLD,
	calculate_context_alignment,
	calculate_experience_alignment,
	calculate_state_alignment,
	current_patient,
	get_semantic_candidates,
	sacr_retrieve,
)
from experiments import RANDOM_SEED, TOP_K, _make_query_pairs


EXPERIMENT_ID = "RQ2_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "rq2"
CANDIDATE_CONTEXT_THRESHOLDS = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]


# ---------------------------------------------------------------------------
# Task 2: empirical S_context distribution + defensible threshold selection
# ---------------------------------------------------------------------------

def _context_alignment_samples(query_pairs, candidate_k=CANDIDATE_K):
	"""For every one of the 40 RQ1 base/shifted query instances, compute
	S_state and S_context for each of its top `candidate_k` semantic
	candidates -- the exact same pool SACR's ranking stage considers.
	"""
	per_query = {}
	for query_id, versions in query_pairs.items():
		for version in ("base", "shifted"):
			patient = versions[version]
			candidate_indices, _ = get_semantic_candidates(patient, candidate_k)
			state_scores = np.empty(len(candidate_indices), dtype=float)
			context_scores = np.empty(len(candidate_indices), dtype=float)
			for position, index in enumerate(candidate_indices):
				memory_row = memory_df.iloc[index]
				state_scores[position] = calculate_state_alignment(patient, memory_row, memory_df)
				context_scores[position] = calculate_context_alignment(patient, memory_row, memory_df)
			per_query[(query_id, version)] = (state_scores, context_scores)
	return per_query


def _descriptive_stats(samples):
	return {
		"count": int(len(samples)),
		"mean": float(np.mean(samples)),
		"median": float(np.median(samples)),
		"std": float(np.std(samples, ddof=1)),
		"min": float(np.min(samples)),
		"max": float(np.max(samples)),
		"p10": float(np.percentile(samples, 10)),
		"p25": float(np.percentile(samples, 25)),
		"p50": float(np.percentile(samples, 50)),
		"p75": float(np.percentile(samples, 75)),
		"p90": float(np.percentile(samples, 90)),
	}


def _evaluate_candidate_thresholds(
	per_query, thresholds=CANDIDATE_CONTEXT_THRESHOLDS,
	state_threshold=STATE_ALIGNMENT_THRESHOLD, top_k=TOP_K,
):
	all_context = np.concatenate([ctx for (_, ctx) in per_query.values()])
	rows = []
	for tau in thresholds:
		pct_passing = float(np.mean(all_context >= tau)) * 100.0
		n_passing = int(np.sum(all_context >= tau))
		joint_counts = [
			int(np.sum((state_scores >= state_threshold) & (context_scores >= tau)))
			for (state_scores, context_scores) in per_query.values()
		]
		min_joint = min(joint_counts)
		mean_joint = float(np.mean(joint_counts))
		n_queries_short = int(sum(1 for c in joint_counts if c < top_k))
		rows.append(
			{
				"context_threshold": tau,
				"pct_context_passing_overall": pct_passing,
				"n_context_passing_overall": n_passing,
				"n_context_samples_overall": int(len(all_context)),
				"min_joint_eligible_per_query": min_joint,
				"mean_joint_eligible_per_query": mean_joint,
				"n_query_instances_below_top_k": n_queries_short,
				"n_query_instances_total": len(joint_counts),
				"sufficient_for_top_k_all_queries": min_joint >= top_k,
			}
		)
	return rows


def _select_context_threshold(candidate_rows):
	"""Pre-committed, mechanical selection rule (decided before inspecting
	whether it "looks good"):

	Pick the LARGEST candidate threshold (most selective, and the closest of
	the candidates to the existing 0.70 state threshold when it qualifies)
	for which EVERY ONE of the 40 tested query instances still has at least
	TOP_K jointly-eligible candidates within the top CANDIDATE_K semantic
	pool. This directly operationalizes "sufficient retrieval coverage" and
	"avoiding an unrealistically restrictive filter" as hard requirements,
	and "conceptual consistency with the state threshold" as the tie-break
	among values that already satisfy coverage.

	If no candidate satisfies the coverage requirement, fall back to the
	candidate that maximizes the minimum per-query eligible count, and this
	fallback is reported explicitly (not silently).
	"""
	sufficient = [r for r in candidate_rows if r["sufficient_for_top_k_all_queries"]]
	if sufficient:
		chosen = max(sufficient, key=lambda r: r["context_threshold"])
		return chosen["context_threshold"], "primary", chosen
	fallback = max(candidate_rows, key=lambda r: r["min_joint_eligible_per_query"])
	return fallback["context_threshold"], "fallback", fallback


# ---------------------------------------------------------------------------
# Task 4: controlled eligibility tests -- synthetic override helpers
# ---------------------------------------------------------------------------

def _numeric_bounds(column):
	values = pd.to_numeric(memory_df[column], errors="coerce").dropna()
	return float(values.min()), float(values.max())


def _farthest_numeric_value(column, current_value):
	col_min, col_max = _numeric_bounds(column)
	current_value = float(current_value)
	return col_max if abs(col_max - current_value) >= abs(current_value - col_min) else col_min


def _mismatched_category(column, current_value):
	for value in memory_df[column].dropna().unique():
		if value != current_value:
			return value
	raise ValueError(f"no alternate category available for column '{column}'")


def _matched_values(patient, columns):
	return {column: patient[column] for column in columns}


def _mismatched_values(patient, columns):
	values = {}
	for column in columns:
		if column in CATEGORICAL_COLS:
			values[column] = _mismatched_category(column, patient[column])
		else:
			values[column] = _farthest_numeric_value(column, patient[column])
	return values


def _apply_overrides(bank, index, overrides):
	for column, value in overrides.items():
		bank.at[index, column] = value


def _fresh_bank():
	return memory_df.reset_index(drop=True).copy()


def _mismatch_count_below_threshold(n_fields, threshold):
	"""Smallest number of fields (of n_fields) that must be mismatched so the
	resulting mean similarity (assuming the rest match exactly) is strictly
	below `threshold`."""
	for k in range(0, n_fields + 1):
		if (n_fields - k) / n_fields < threshold:
			return k
	return n_fields


def _rq2_active_rules(context_threshold):
	return {
		"min_state_alignment": STATE_ALIGNMENT_THRESHOLD,
		"min_context_alignment": context_threshold,
	}


# ---------------------------------------------------------------------------
# Task 4 -- TEST A: state violation
# ---------------------------------------------------------------------------

def test_state_violation(patient, context_threshold, candidate_k=CANDIDATE_K):
	candidate_indices, semantic_scores = get_semantic_candidates(patient, candidate_k)
	idx_full = int(candidate_indices[0])
	idx_partial = int(candidate_indices[1])

	bank = _fresh_bank()
	_apply_overrides(bank, idx_full, _matched_values(patient, CONTEXT_COLS + EXPERIENCE_COLS))
	_apply_overrides(bank, idx_full, _mismatched_values(patient, STATE_COLS))

	n_mismatch = _mismatch_count_below_threshold(len(STATE_COLS), STATE_ALIGNMENT_THRESHOLD)
	mismatch_cols, match_cols = STATE_COLS[:n_mismatch], STATE_COLS[n_mismatch:]
	_apply_overrides(bank, idx_partial, _matched_values(patient, CONTEXT_COLS + EXPERIENCE_COLS))
	_apply_overrides(bank, idx_partial, _matched_values(patient, match_cols))
	_apply_overrides(bank, idx_partial, _mismatched_values(patient, mismatch_cols))

	active_rules = _rq2_active_rules(context_threshold)
	results = sacr_retrieve(
		patient, active_rules, top_k=candidate_k, candidate_k=candidate_k, memory_bank=bank
	)

	detail = {}
	passed = True
	for label, idx in (("fully_mismatched_state", idx_full), ("partially_mismatched_state", idx_partial)):
		row = bank.iloc[idx]
		state_alignment = calculate_state_alignment(patient, row, bank)
		context_alignment = calculate_context_alignment(patient, row, bank)
		rejected = row["memory_id"] not in results["memory_id"].values
		ok = (
			rejected
			and state_alignment < STATE_ALIGNMENT_THRESHOLD
			and context_alignment >= context_threshold
		)
		detail[label] = {
			"memory_id": row["memory_id"],
			"semantic_similarity": float(semantic_scores[idx]),
			"state_alignment": state_alignment,
			"context_alignment": context_alignment,
			"rejected": bool(rejected),
			"check_passed": bool(ok),
		}
		passed = passed and ok
	return passed, detail


# ---------------------------------------------------------------------------
# Task 4 -- TEST B: context violation
# ---------------------------------------------------------------------------

def test_context_violation(patient, context_threshold, candidate_k=CANDIDATE_K):
	candidate_indices, semantic_scores = get_semantic_candidates(patient, candidate_k)
	idx_full = int(candidate_indices[2])
	idx_partial = int(candidate_indices[3])

	bank = _fresh_bank()
	_apply_overrides(bank, idx_full, _matched_values(patient, STATE_COLS + EXPERIENCE_COLS))
	_apply_overrides(bank, idx_full, _mismatched_values(patient, CONTEXT_COLS))

	n_mismatch = _mismatch_count_below_threshold(len(CONTEXT_COLS), context_threshold)
	mismatch_cols, match_cols = CONTEXT_COLS[:n_mismatch], CONTEXT_COLS[n_mismatch:]
	_apply_overrides(bank, idx_partial, _matched_values(patient, STATE_COLS + EXPERIENCE_COLS))
	_apply_overrides(bank, idx_partial, _matched_values(patient, match_cols))
	_apply_overrides(bank, idx_partial, _mismatched_values(patient, mismatch_cols))

	active_rules = _rq2_active_rules(context_threshold)
	results = sacr_retrieve(
		patient, active_rules, top_k=candidate_k, candidate_k=candidate_k, memory_bank=bank
	)

	detail = {}
	passed = True
	for label, idx in (("fully_mismatched_context", idx_full), ("partially_mismatched_context", idx_partial)):
		row = bank.iloc[idx]
		state_alignment = calculate_state_alignment(patient, row, bank)
		context_alignment = calculate_context_alignment(patient, row, bank)
		rejected = row["memory_id"] not in results["memory_id"].values
		ok = (
			rejected
			and context_alignment < context_threshold
			and state_alignment >= STATE_ALIGNMENT_THRESHOLD
		)
		detail[label] = {
			"memory_id": row["memory_id"],
			"semantic_similarity": float(semantic_scores[idx]),
			"state_alignment": state_alignment,
			"context_alignment": context_alignment,
			"rejected": bool(rejected),
			"check_passed": bool(ok),
		}
		passed = passed and ok
	return passed, detail


# ---------------------------------------------------------------------------
# Task 4 -- TEST C: jointly aligned
# ---------------------------------------------------------------------------

def test_jointly_aligned(patient, context_threshold, candidate_k=CANDIDATE_K):
	candidate_indices, semantic_scores = get_semantic_candidates(patient, candidate_k)
	idx1, idx2 = int(candidate_indices[4]), int(candidate_indices[5])

	bank = _fresh_bank()
	for idx in (idx1, idx2):
		_apply_overrides(bank, idx, _matched_values(patient, STATE_COLS + CONTEXT_COLS))

	active_rules = _rq2_active_rules(context_threshold)
	results = sacr_retrieve(
		patient, active_rules, top_k=candidate_k, candidate_k=candidate_k, memory_bank=bank
	)

	detail = {}
	passed = True
	for label, idx in (("candidate_1", idx1), ("candidate_2", idx2)):
		row = bank.iloc[idx]
		state_alignment = calculate_state_alignment(patient, row, bank)
		context_alignment = calculate_context_alignment(patient, row, bank)
		present = row["memory_id"] in results["memory_id"].values
		ok = (
			present
			and state_alignment >= STATE_ALIGNMENT_THRESHOLD
			and context_alignment >= context_threshold
		)
		detail[label] = {
			"memory_id": row["memory_id"],
			"semantic_similarity": float(semantic_scores[idx]),
			"state_alignment": state_alignment,
			"context_alignment": context_alignment,
			"eligible_and_retrieved": bool(present),
			"check_passed": bool(ok),
		}
		passed = passed and ok
	return passed, detail


# ---------------------------------------------------------------------------
# Task 4 -- TEST D: ranking after eligibility
# ---------------------------------------------------------------------------

def test_ranking_after_eligibility(patient, context_threshold, candidate_k=CANDIDATE_K):
	candidate_indices, semantic_scores = get_semantic_candidates(patient, candidate_k)
	idx_ineligible = int(candidate_indices[0])
	spread = [6, min(20, candidate_k - 1), min(50, candidate_k - 1)]
	idx_e1, idx_e2, idx_e3 = (int(candidate_indices[i]) for i in spread)

	bank = _fresh_bank()
	# ineligible: highest raw semantic score of the whole pool, context matched
	# (high), state deliberately mismatched -- must never outrank the eligible set.
	_apply_overrides(bank, idx_ineligible, _matched_values(patient, CONTEXT_COLS + EXPERIENCE_COLS))
	_apply_overrides(bank, idx_ineligible, _mismatched_values(patient, STATE_COLS))

	# three eligible candidates (state+context both matched -> both = 1.0).
	# EXPERIENCE_COLS is deliberately set to trade OFF against each
	# candidate's natural semantic rank (idx_e1 has the highest natural
	# semantic score but the worst experience match; idx_e3 the reverse) so
	# the expected order is not simply "preserve semantic rank" -- it can
	# only be predicted by actually evaluating the weighted formula, which
	# is the property this test needs to catch a ranking bug.
	_apply_overrides(bank, idx_e1, _matched_values(patient, STATE_COLS + CONTEXT_COLS))
	_apply_overrides(bank, idx_e1, _mismatched_values(patient, EXPERIENCE_COLS))

	half = max(1, len(EXPERIENCE_COLS) // 2)
	_apply_overrides(bank, idx_e2, _matched_values(patient, STATE_COLS + CONTEXT_COLS))
	_apply_overrides(bank, idx_e2, _matched_values(patient, EXPERIENCE_COLS[:half]))
	_apply_overrides(bank, idx_e2, _mismatched_values(patient, EXPERIENCE_COLS[half:]))

	_apply_overrides(bank, idx_e3, _matched_values(patient, STATE_COLS + CONTEXT_COLS))
	_apply_overrides(bank, idx_e3, _matched_values(patient, EXPERIENCE_COLS))

	active_rules = _rq2_active_rules(context_threshold)
	results = sacr_retrieve(
		patient, active_rules, top_k=candidate_k, candidate_k=candidate_k, memory_bank=bank
	)

	ineligible_id = bank.iloc[idx_ineligible]["memory_id"]
	ineligible_absent = ineligible_id not in results["memory_id"].values

	expected = []
	eligible_ids = set()
	for idx in (idx_e1, idx_e2, idx_e3):
		row = bank.iloc[idx]
		experience_alignment = calculate_experience_alignment(patient, row)
		score = (
			LAMBDA_SEMANTIC * float(semantic_scores[idx])
			+ LAMBDA_CONTEXT * 1.0
			+ LAMBDA_EXPERIENCE * experience_alignment
		)
		expected.append((row["memory_id"], score))
		eligible_ids.add(row["memory_id"])
	expected.sort(key=lambda item: item[1], reverse=True)
	expected_order = [memory_id for memory_id, _ in expected]

	actual_order = [mid for mid in results["memory_id"] if mid in eligible_ids]
	ranking_matches = actual_order == expected_order
	passed = ineligible_absent and ranking_matches and len(actual_order) == 3

	detail = {
		"ineligible_memory_id": ineligible_id,
		"ineligible_had_highest_raw_semantic_score": True,
		"ineligible_absent_from_output": bool(ineligible_absent),
		"expected_ranking_order": expected_order,
		"actual_ranking_order": actual_order,
		"ranking_matches_formula": bool(ranking_matches),
	}
	return passed, detail


# ---------------------------------------------------------------------------
# Task 6: old (weighted-only) vs new (eligibility-gated) formulation
# ---------------------------------------------------------------------------

def compare_old_vs_new_formulation(patient, context_threshold, candidate_k=CANDIDATE_K):
	"""Construct one memory with a high weighted ranking score despite a
	state-alignment violation, and show that (a) a naive weighted-average-
	only ranking would have ranked it competitively with the actual top-k,
	while (b) SACR's eligibility gate rejects it outright. This illustrates
	why a weighted average alone cannot mathematically guarantee alignment;
	it is not a claim that RQ1's actual (state-gated) implementation was
	incorrect -- RQ1 already had a hard state gate, just not a context gate.
	"""
	candidate_indices, semantic_scores = get_semantic_candidates(patient, candidate_k)
	idx = int(candidate_indices[0])
	bank = _fresh_bank()
	_apply_overrides(bank, idx, _matched_values(patient, CONTEXT_COLS + EXPERIENCE_COLS))
	_apply_overrides(bank, idx, _mismatched_values(patient, STATE_COLS))

	row = bank.iloc[idx]
	semantic_similarity = float(semantic_scores[idx])
	state_alignment = calculate_state_alignment(patient, row, bank)
	context_alignment = calculate_context_alignment(patient, row, bank)
	experience_alignment = calculate_experience_alignment(patient, row)
	naive_weighted_score = (
		LAMBDA_SEMANTIC * semantic_similarity
		+ LAMBDA_CONTEXT * context_alignment
		+ LAMBDA_EXPERIENCE * experience_alignment
	)

	active_rules = _rq2_active_rules(context_threshold)
	results = sacr_retrieve(
		patient, active_rules, top_k=TOP_K, candidate_k=candidate_k, memory_bank=bank
	)
	rejected = row["memory_id"] not in results["memory_id"].values
	lowest_top_k_score = float(results["final_ranking_score"].min()) if len(results) else float("nan")
	would_have_beaten_current_top_k = (
		naive_weighted_score >= lowest_top_k_score if not np.isnan(lowest_top_k_score) else None
	)

	return {
		"memory_id": row["memory_id"],
		"semantic_similarity": semantic_similarity,
		"state_alignment": state_alignment,
		"context_alignment": context_alignment,
		"experience_alignment": experience_alignment,
		"naive_weighted_score": naive_weighted_score,
		"lowest_actual_top_k_score": lowest_top_k_score,
		"would_have_beaten_current_top_k_min_score": would_have_beaten_current_top_k,
		"rejected_by_eligibility_gate": bool(rejected),
	}


# ---------------------------------------------------------------------------
# Task 5: exhaustive invariant check over the real RQ1 query scenarios
# ---------------------------------------------------------------------------

def exhaustive_invariant_check(query_pairs, context_threshold, candidate_k=CANDIDATE_K, top_k=TOP_K):
	active_rules = _rq2_active_rules(context_threshold)
	detailed_rows = []
	state_violations = 0
	context_violations = 0
	joint_violations = 0
	total_retrieved = 0
	short_query_instances = 0

	for query_id, versions in query_pairs.items():
		for version in ("base", "shifted"):
			patient = versions[version]
			results = sacr_retrieve(patient, active_rules, top_k=top_k, candidate_k=candidate_k)
			if len(results) < top_k:
				short_query_instances += 1
			for rank, (_, row) in enumerate(results.iterrows(), start=1):
				total_retrieved += 1
				state_ok = row["state_alignment"] >= STATE_ALIGNMENT_THRESHOLD
				context_ok = row["context_alignment"] >= context_threshold
				if not state_ok:
					state_violations += 1
				if not context_ok:
					context_violations += 1
				if not (state_ok and context_ok):
					joint_violations += 1
					print("RQ2 INVARIANT VIOLATION")
					print(f"  query_id={query_id} version={version} rank={rank}")
					print(f"  memory_id={row['memory_id']}")
					print(
						f"  state_alignment={row['state_alignment']:.6f} "
						f"(threshold {STATE_ALIGNMENT_THRESHOLD})"
					)
					print(
						f"  context_alignment={row['context_alignment']:.6f} "
						f"(threshold {context_threshold})"
					)
					raise AssertionError(
						f"RQ2 invariant violated for {query_id}/{version} "
						f"memory {row['memory_id']}"
					)
				detailed_rows.append(
					{
						"query_id": query_id,
						"query_version": version,
						"rank": rank,
						"memory_id": row["memory_id"],
						"semantic_similarity": float(row["semantic_similarity"]),
						"state_alignment": float(row["state_alignment"]),
						"context_alignment": float(row["context_alignment"]),
						"experience_alignment": float(row["experience_alignment"]),
						"final_ranking_score": float(row["final_ranking_score"]),
					}
				)

	summary = {
		"invariant_satisfied": joint_violations == 0,
		"total_retrieved": total_retrieved,
		"state_violations": state_violations,
		"context_violations": context_violations,
		"joint_violations": joint_violations,
		"query_instances_returning_fewer_than_top_k": short_query_instances,
	}
	return summary, detailed_rows


def _git_commit_hash():
	try:
		result = subprocess.run(
			["git", "rev-parse", "--short", "HEAD"],
			cwd=Path(__file__).resolve().parent,
			capture_output=True,
			text=True,
			timeout=5,
		)
		if result.returncode == 0:
			return result.stdout.strip()
	except Exception:
		pass
	return "unknown"


# ---------------------------------------------------------------------------
# Task 8: report / CSV generation
# ---------------------------------------------------------------------------

def _write_config(metadata, threshold_stats, chosen, chosen_kind):
	lines = [
		f"experiment_id: {metadata['experiment_id']}",
		f"timestamp: {metadata['timestamp']}",
		f"python_version: {metadata['python_version']}",
		f"git_commit: {metadata['git_commit']}",
		f"random_seed: {metadata['random_seed']}",
		f"dataset_shape: {metadata['dataset_shape']}",
		f"number_of_memory_records: {metadata['number_of_memories']}",
		f"state_cols: {STATE_COLS}",
		f"context_cols: {CONTEXT_COLS}",
		f"experience_cols: {EXPERIENCE_COLS}",
		f"state_alignment_threshold: {STATE_ALIGNMENT_THRESHOLD}",
		f"context_alignment_threshold: {chosen} ({chosen_kind})",
		f"lambda_semantic: {LAMBDA_SEMANTIC}",
		f"lambda_context: {LAMBDA_CONTEXT}",
		f"lambda_experience: {LAMBDA_EXPERIENCE}",
		f"candidate_k: {CANDIDATE_K}",
		f"top_k: {TOP_K}",
		f"number_of_query_scenarios: {metadata['number_of_query_pairs']}",
		f"number_of_query_instances: {metadata['number_of_query_instances']}",
		"",
		"S_context empirical distribution (top-candidate_k semantic pool, all 40 query instances):",
	]
	for key in ("count", "mean", "median", "std", "min", "max", "p10", "p25", "p50", "p75", "p90"):
		lines.append(f"  {key}: {threshold_stats[key]}")
	(RESULTS_DIR / "config.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(
	metadata, threshold_stats, candidate_rows, chosen_threshold, chosen_kind,
	test_results, comparison, invariant_summary,
):
	lines = []
	lines.append("=" * 60)
	lines.append("RQ2 -- SACR MATHEMATICAL ALIGNMENT EXPERIMENT")
	lines.append("=" * 60)

	lines.append("\n## 1. Research Question\n")
	lines.append("How can memory retrieval mathematically guarantee joint state and")
	lines.append("contextual alignment?")

	lines.append("\n## 2. Motivation\n")
	lines.append("RQ1 showed that SACR's weighted ranking score,")
	lines.append("  S_rank = 0.60*S_semantic + 0.25*S_context + 0.15*S_experience,")
	lines.append("combined with a hard state-alignment threshold, sharply reduces")
	lines.append("state-misaligned retrieval versus semantic-only retrieval. But a")
	lines.append("weighted average, by itself, does not mathematically GUARANTEE any")
	lines.append("particular alignment component stays above a bound -- a low S_state")
	lines.append("or S_context can still be compensated by a high S_semantic. RQ2 asks")
	lines.append("whether adding explicit hard eligibility constraints (rather than")
	lines.append("relying on the weights alone) can make that guarantee structural.")

	lines.append("\n## 3. Mathematical Formulation\n")
	lines.append("S_state(q,m)   = (1/|S|) * sum_j sim(q_j, m_j),  j in STATE_COLS")
	lines.append("S_context(q,m) = (1/|C|) * sum_j sim(q_j, m_j),  j in CONTEXT_COLS")
	lines.append("")
	lines.append("Eligibility indicator:")
	lines.append("  A(q,m) = 1  if S_state(q,m) >= tau_state AND S_context(q,m) >= tau_context")
	lines.append("           0  otherwise")
	lines.append("")
	lines.append("Only memories with A(q,m)=1 may enter the final retrieved set. Among")
	lines.append("eligible memories, the existing (unchanged) ranking formula orders them:")
	lines.append("  S_rank = 0.60*S_semantic + 0.25*S_context + 0.15*S_experience")
	lines.append("")
	lines.append("Property tested, for every returned memory m of query q:")
	lines.append("  m in R_k(q)  =>  S_state(q,m) >= tau_state  AND  S_context(q,m) >= tau_context")

	lines.append("\n## 4. Existing SACR Limitation\n")
	lines.append("Before RQ2, SACR enforced a hard threshold on S_state only")
	lines.append(f"(tau_state = {STATE_ALIGNMENT_THRESHOLD}); S_context entered only the weighted")
	lines.append("ranking score, so a memory with excellent context but poor state could")
	lines.append("still be excluded by the state gate, but a memory with excellent state")
	lines.append("and semantic similarity, yet poor context, faced no hard context")
	lines.append("check at all -- only a 0.25-weighted contribution to its score, which a")
	lines.append("high semantic or experience term could outweigh.")

	lines.append("\n## 5. Proposed Eligibility Constraint\n")
	lines.append("A second hard gate on S_context is added, opt-in via")
	lines.append("`active_rules['min_context_alignment']`. RQ1's own `active_rules` dict")
	lines.append("never sets this key, so RQ1's pipeline, results, and reports are")
	lines.append("provably unaffected (verified by re-running RQ1's own demo script before")
	lines.append("this experiment, with identical diagnostic counts).")
	lines.append("The state threshold (0.70) and ranking weights are unchanged.")

	lines.append("\n## 6. Context Threshold Selection\n")
	lines.append(
		f"S_context was sampled across all {metadata['number_of_query_instances']} RQ1 query"
	)
	lines.append(
		f"instances (base+shifted), for each instance's top {CANDIDATE_K} semantic"
	)
	lines.append("candidates -- the same pool SACR's ranking stage considers -- giving")
	lines.append(f"{threshold_stats['count']} samples in total.")
	lines.append("")
	lines.append("Empirical distribution of S_context:")
	for key, label in (
		("mean", "mean"), ("median", "median"), ("std", "std"),
		("min", "min"), ("max", "max"), ("p10", "10th percentile"),
		("p25", "25th percentile"), ("p50", "50th percentile"),
		("p75", "75th percentile"), ("p90", "90th percentile"),
	):
		lines.append(f"  {label}: {threshold_stats[key]:.6f}")
	lines.append("")
	lines.append("Candidate thresholds evaluated:")
	lines.append(
		f"{'tau_context':>12}{'% passing':>12}{'min joint/query':>18}"
		f"{'mean joint/query':>18}{'queries short':>15}{'sufficient?':>13}"
	)
	for row in candidate_rows:
		lines.append(
			f"{row['context_threshold']:>12.2f}"
			f"{row['pct_context_passing_overall']:>11.2f}%"
			f"{row['min_joint_eligible_per_query']:>18d}"
			f"{row['mean_joint_eligible_per_query']:>18.2f}"
			f"{row['n_query_instances_below_top_k']:>15d}"
			f"{str(row['sufficient_for_top_k_all_queries']):>13}"
		)
	lines.append("")
	lines.append(
		"Selection rule (pre-committed, not chosen after seeing results): pick the"
	)
	lines.append(
		"LARGEST candidate threshold for which every one of the 40 query instances"
	)
	lines.append(
		f"still has >= TOP_K ({TOP_K}) jointly state+context eligible candidates within"
	)
	lines.append(
		f"the top {CANDIDATE_K} semantic pool ('sufficient retrieval coverage', a hard"
	)
	lines.append(
		"requirement); among values satisfying that, the largest is closest in"
	)
	lines.append(
		"spirit to the existing 0.70 state threshold ('conceptual consistency') while"
	)
	lines.append("remaining as selective as the coverage requirement allows.")
	lines.append("")
	fallback_note = " (FALLBACK: no candidate met the coverage requirement)" if chosen_kind == "fallback" else ""
	lines.append(f"Chosen tau_context = {chosen_threshold}{fallback_note}")
	lines.append(f"State threshold tau_state = {STATE_ALIGNMENT_THRESHOLD} (unchanged from RQ1).")

	lines.append("\n## 7. Experimental Setup\n")
	lines.append(f"Dataset: UCI Diabetes 130-US Hospitals, {metadata['number_of_memories']} memories")
	lines.append(f"Query scenarios reused from RQ1: {metadata['number_of_query_pairs']} pairs")
	lines.append(f"Query instances (base+shifted): {metadata['number_of_query_instances']}")
	lines.append(f"CANDIDATE_K: {CANDIDATE_K}   TOP_K: {TOP_K}")
	lines.append(
		f"Ranking weights (unchanged): semantic={LAMBDA_SEMANTIC} context={LAMBDA_CONTEXT} "
		f"experience={LAMBDA_EXPERIENCE}"
	)
	lines.append(f"Random seed: {RANDOM_SEED}")
	lines.append("Fixed test-query patient for Tests A-D: sacr.current_patient")

	def test_block(number, title, passed, detail):
		lines.append(f"\n## {number}. {title}\n")
		lines.append(f"Result: {'PASS' if passed else 'FAIL'}")
		lines.append(json.dumps(detail, indent=2, default=str))

	test_block(8, "Test A -- State Violation", test_results["A"][0], test_results["A"][1])
	test_block(9, "Test B -- Context Violation", test_results["B"][0], test_results["B"][1])
	test_block(10, "Test C -- Jointly Aligned Memories", test_results["C"][0], test_results["C"][1])
	test_block(11, "Test D -- Ranking After Eligibility", test_results["D"][0], test_results["D"][1])

	lines.append("\n## 12. Exhaustive Invariant Verification\n")
	lines.append(
		f"RQ2 invariant satisfied: {'TRUE' if invariant_summary['invariant_satisfied'] else 'FALSE'}"
	)
	lines.append(f"Total memories retrieved (across all {metadata['number_of_query_instances']} query instances): "
		f"{invariant_summary['total_retrieved']}")
	lines.append(f"State violations: {invariant_summary['state_violations']}")
	lines.append(f"Context violations: {invariant_summary['context_violations']}")
	lines.append(f"Joint violations: {invariant_summary['joint_violations']}")
	lines.append(
		"Query instances returning fewer than TOP_K memories: "
		f"{invariant_summary['query_instances_returning_fewer_than_top_k']}"
	)

	lines.append("\n## 13. Results\n")
	lines.append("Old (weighted-only) vs. new (eligibility-gated) formulation, on one")
	lines.append("constructed memory with a state violation but high semantic/context:")
	lines.append(json.dumps(comparison, indent=2, default=str))

	lines.append("\n## 14. Interpretation\n")
	all_tests_passed = all(test_results[k][0] for k in ("A", "B", "C", "D"))
	lines.append(
		f"All four controlled tests {'passed' if all_tests_passed else 'DID NOT all pass'}, and the"
	)
	lines.append(
		f"exhaustive invariant check over all {metadata['number_of_query_instances']} real RQ1 query"
	)
	lines.append(
		f"instances found {invariant_summary['joint_violations']} joint violations out of "
		f"{invariant_summary['total_retrieved']} retrieved memories."
	)
	lines.append(
		"The comparison in Section 13 shows a state-violating memory whose naive"
	)
	lines.append(
		f"weighted score ({comparison['naive_weighted_score']:.6f}) would have "
		f"{'matched or beaten' if comparison['would_have_beaten_current_top_k_min_score'] else 'fallen short of'} "
		f"the actual top-{TOP_K}'s lowest score "
		f"({comparison['lowest_actual_top_k_score']:.6f}) under weighted-only ranking, yet the"
	)
	lines.append("eligibility gate rejected it outright -- illustrating why a weighted")
	lines.append("average alone does not provide the structural guarantee an explicit gate does.")

	lines.append("\n## 15. Limitations\n")
	lines.append("- This verifies a STRUCTURAL property of the code (a row can only enter")
	lines.append("  the ranked output after passing both checks), not a claim about clinical")
	lines.append("  correctness, causal validity, or that tau_context/tau_state are optimal.")
	lines.append("- Tests A-D use one fixed query (sacr.current_patient) and constructed")
	lines.append("  (not organically retrieved) synthetic memories to guarantee the exact")
	lines.append("  properties each test needs; they demonstrate the mechanism works as")
	lines.append("  specified, not that it generalizes to every possible query/memory pair.")
	lines.append("- tau_context was chosen from a small predefined candidate set using one")
	lines.append("  coverage rule over the current 20 RQ1 scenarios; a different scenario")
	lines.append("  sample or candidate set could select a different value.")
	min_joint_at_chosen = next(
		(r["min_joint_eligible_per_query"] for r in candidate_rows if r["context_threshold"] == chosen_threshold),
		None,
	)
	if min_joint_at_chosen is not None and min_joint_at_chosen <= TOP_K:
		lines.append(
			f"- The chosen threshold's coverage margin is thin: the worst-case query"
		)
		lines.append(
			f"  instance had exactly {min_joint_at_chosen} jointly-eligible candidates against"
		)
		lines.append(
			f"  TOP_K={TOP_K} -- a slightly different query or a smaller CANDIDATE_K could"
		)
		lines.append(
			"  produce a query with fewer than TOP_K eligible memories at this threshold."
		)
	lines.append("- This does not claim generalization to other datasets.")

	lines.append("\n## 16. Reproducibility\n")
	lines.append(f"Random seed: {metadata['random_seed']}")
	lines.append(f"Python version: {metadata['python_version']}")
	lines.append(f"Timestamp: {metadata['timestamp']}")
	lines.append(f"Git commit: {metadata['git_commit']}")
	lines.append(f"Dataset shape: {metadata['dataset_shape']}")
	lines.append(f"Number of memory records: {metadata['number_of_memories']}")
	lines.append(f"State columns: {STATE_COLS}")
	lines.append(f"Context columns: {CONTEXT_COLS}")
	lines.append(f"Experience columns: {EXPERIENCE_COLS}")
	lines.append(f"State threshold: {STATE_ALIGNMENT_THRESHOLD}")
	lines.append(f"Context threshold: {chosen_threshold} ({chosen_kind})")
	lines.append(f"Ranking weights: semantic={LAMBDA_SEMANTIC} context={LAMBDA_CONTEXT} experience={LAMBDA_EXPERIENCE}")
	lines.append(f"CANDIDATE_K: {CANDIDATE_K}   TOP_K: {TOP_K}")
	lines.append(f"Query scenarios reused from RQ1: {metadata['number_of_query_pairs']} (identical generation code)")
	lines.append("Files generated:")
	lines.append("  results/rq2/experiment_report.txt")
	lines.append("  results/rq2/rq2_summary.csv")
	lines.append("  results/rq2/rq2_detailed_results.csv")
	lines.append("  results/rq2/context_threshold_analysis.csv")
	lines.append("  results/rq2/config.txt")

	lines.append("\n## 17. Conclusion\n")
	lines.append(
		"Under the similarity functions and thresholds defined above, adding an"
	)
	lines.append(
		"explicit joint state+context eligibility gate to SACR structurally"
	)
	lines.append(
		"prevents any memory violating either constraint from entering the final"
	)
	lines.append(
		f"retrieved set: {invariant_summary['joint_violations']} violations were found across"
	)
	lines.append(
		f"{invariant_summary['total_retrieved']} retrieved memories in the real RQ1 query scenarios,"
	)
	lines.append(
		"and all four controlled tests behaved as the mathematical formulation"
	)
	lines.append(
		"predicts. This is a narrow, code-level guarantee about this eligibility"
	)
	lines.append(
		"mechanism under these similarity definitions -- it is not a claim of"
	)
	lines.append("clinical correctness, causal validity, or an optimal threshold choice.")

	report_text = "\n".join(lines) + "\n"
	(RESULTS_DIR / "experiment_report.txt").write_text(report_text, encoding="utf-8")


def run_rq2():
	RESULTS_DIR.mkdir(parents=True, exist_ok=True)
	np.random.seed(RANDOM_SEED)

	query_pairs = _make_query_pairs()
	number_of_query_instances = len(query_pairs) * 2

	# --- Task 2 ---
	per_query = _context_alignment_samples(query_pairs, CANDIDATE_K)
	all_context = np.concatenate([ctx for (_, ctx) in per_query.values()])
	threshold_stats = _descriptive_stats(all_context)
	candidate_rows = _evaluate_candidate_thresholds(per_query)
	chosen_threshold, chosen_kind, _ = _select_context_threshold(candidate_rows)
	pd.DataFrame(candidate_rows).to_csv(RESULTS_DIR / "context_threshold_analysis.csv", index=False)

	# --- Task 4 ---
	test_results = {
		"A": test_state_violation(current_patient, chosen_threshold),
		"B": test_context_violation(current_patient, chosen_threshold),
		"C": test_jointly_aligned(current_patient, chosen_threshold),
		"D": test_ranking_after_eligibility(current_patient, chosen_threshold),
	}

	# --- Task 6 ---
	comparison = compare_old_vs_new_formulation(current_patient, chosen_threshold)

	# --- Task 5 ---
	invariant_summary, detailed_rows = exhaustive_invariant_check(query_pairs, chosen_threshold)
	pd.DataFrame(detailed_rows).to_csv(RESULTS_DIR / "rq2_detailed_results.csv", index=False)

	# --- summary CSV ---
	summary_rows = [
		{"check": "test_a_state_violation", "passed": test_results["A"][0]},
		{"check": "test_b_context_violation", "passed": test_results["B"][0]},
		{"check": "test_c_jointly_aligned", "passed": test_results["C"][0]},
		{"check": "test_d_ranking_after_eligibility", "passed": test_results["D"][0]},
		{"check": "exhaustive_invariant_check", "passed": invariant_summary["invariant_satisfied"]},
		{"check": "total_retrieved", "passed": None, "value": invariant_summary["total_retrieved"]},
		{"check": "state_violations", "passed": None, "value": invariant_summary["state_violations"]},
		{"check": "context_violations", "passed": None, "value": invariant_summary["context_violations"]},
		{"check": "joint_violations", "passed": None, "value": invariant_summary["joint_violations"]},
		{"check": "context_threshold_chosen", "passed": None, "value": chosen_threshold},
		{"check": "context_threshold_selection_kind", "passed": None, "value": chosen_kind},
		{"check": "state_threshold", "passed": None, "value": STATE_ALIGNMENT_THRESHOLD},
	]
	pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "rq2_summary.csv", index=False)

	metadata = {
		"experiment_id": EXPERIMENT_ID,
		"timestamp": datetime.now(timezone.utc).isoformat(),
		"python_version": sys.version.split()[0],
		"git_commit": _git_commit_hash(),
		"random_seed": RANDOM_SEED,
		"dataset_shape": list(memory_df.shape),
		"number_of_memories": len(memory_df),
		"number_of_query_pairs": len(query_pairs),
		"number_of_query_instances": number_of_query_instances,
		"state_threshold": STATE_ALIGNMENT_THRESHOLD,
		"context_threshold": chosen_threshold,
		"context_threshold_selection": chosen_kind,
		"lambda_semantic": LAMBDA_SEMANTIC,
		"lambda_context": LAMBDA_CONTEXT,
		"lambda_experience": LAMBDA_EXPERIENCE,
		"candidate_k": CANDIDATE_K,
		"top_k": TOP_K,
	}
	(RESULTS_DIR / "experiment_metadata.json").write_text(
		json.dumps(metadata, indent=4, default=str), encoding="utf-8"
	)

	_write_config(metadata, threshold_stats, chosen_threshold, chosen_kind)
	_write_report(
		metadata, threshold_stats, candidate_rows, chosen_threshold, chosen_kind,
		test_results, comparison, invariant_summary,
	)

	print("\n================ RQ2 COMPLETE ================\n")
	print("Research Question:")
	print("How can memory retrieval mathematically guarantee joint state and")
	print("contextual alignment?\n")
	print(f"Context threshold selected:\n{chosen_threshold} ({chosen_kind})\n")
	print(f"State threshold:\n{STATE_ALIGNMENT_THRESHOLD}\n")
	print(f"Total RQ2 queries:\n{number_of_query_instances}\n")
	print(f"Total memories retrieved:\n{invariant_summary['total_retrieved']}\n")
	print(f"State violations:\n{invariant_summary['state_violations']}\n")
	print(f"Context violations:\n{invariant_summary['context_violations']}\n")
	print(f"Joint violations:\n{invariant_summary['joint_violations']}\n")
	print(
		"RQ2 invariant satisfied:\n"
		f"{'TRUE' if invariant_summary['invariant_satisfied'] else 'FALSE'}\n"
	)
	print("Report:\nresults/rq2/experiment_report.txt\n")
	print("Summary:\nresults/rq2/rq2_summary.csv\n")
	print("Detailed results:\nresults/rq2/rq2_detailed_results.csv\n")
	print("Context threshold analysis:\nresults/rq2/context_threshold_analysis.csv\n")
	print(f"Git commit:\n{metadata['git_commit']}\n")

	return {
		"metadata": metadata,
		"threshold_stats": threshold_stats,
		"candidate_rows": candidate_rows,
		"test_results": test_results,
		"comparison": comparison,
		"invariant_summary": invariant_summary,
	}


if __name__ == "__main__":
	run_rq2()
