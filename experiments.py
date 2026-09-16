"""RQ1: can state-aware retrieval (SACR) reduce retrieval of semantically
relevant but state-misaligned experiences, compared with conventional
semantic retrieval?

Two experiments are run against the same 20 base/shifted query scenarios:

  Experiment A -- NORMAL memory bank (state attached to its real owner).
  Experiment B -- STATE-SHUFFLED memory bank (STATE_COLS tuples permuted
                  across memories; CONTEXT_COLS/EXPERIENCE_COLS/outcome
                  columns untouched), which deliberately decouples semantic
                  relevance from state compatibility.

A secondary BASE->SHIFTED sensitivity analysis (comparing each scenario's
base and shifted structured state against the SAME fixed-text retrieval) is
also reported, but it is not the primary evidence for RQ1.
"""

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from baseline_rag import memory_df, semantic_retrieve, state_neutral_text
from columns import CONTEXT_COLS, EXPERIENCE_COLS, STATE_COLS
from sacr import (
	CANDIDATE_K,
	LAMBDA_CONTEXT,
	LAMBDA_EXPERIENCE,
	LAMBDA_SEMANTIC,
	STATE_ALIGNMENT_THRESHOLD,
	active_rules,
	calculate_state_alignment,
	sacr_retrieve,
)


TOP_K = 5
RANDOM_SEED = 42
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EXPERIMENT_ID = "RQ1_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "rq1"


def _native_value(value):
	"""Convert a pandas/numpy scalar to a plain Python type for JSON dumping."""
	if isinstance(value, np.integer):
		return int(value)
	if isinstance(value, np.floating):
		return float(value)
	if isinstance(value, np.bool_):
		return bool(value)
	return value


def _scenario_templates(n):
	"""Sample n distinct real patient rows to diversify context/experience.

	Each template supplies the STATE_COLS/CONTEXT_COLS/EXPERIENCE_COLS
	starting point for one query pair. Outcome/governance columns
	(readmitted, feedback, quality_score, quarantined, memory_id) are
	deliberately not copied, since they must never influence retrieval.
	"""
	template_cols = STATE_COLS + CONTEXT_COLS + EXPERIENCE_COLS
	sampled = memory_df[template_cols].sample(n=n, random_state=RANDOM_SEED)
	return [
		{col: _native_value(row[col]) for col in template_cols}
		for _, row in sampled.iterrows()
	]


def _state_neutral_query_text(patient):
	"""Return the CONTEXT_COLS/EXPERIENCE_COLS-only text used for retrieval.

	This is identical to the text sacr.patient_to_text produces; it is
	exposed here so base/shifted invariants can be checked directly against
	the query dicts this module builds.
	"""
	return state_neutral_text(patient)


def _make_query_pairs():
	"""Create 20 deterministic base/shifted patient-query pairs.

	Each pair starts from a distinct sampled scenario template (diversifying
	CONTEXT_COLS/EXPERIENCE_COLS across the 20 pairs) and then applies one
	shift dict that only ever touches STATE_COLS. Because the state-neutral
	text depends only on CONTEXT_COLS/EXPERIENCE_COLS, and those are copied
	unchanged from template to both base and shifted, the two versions of
	every pair share identical state-neutral text by construction.
	"""
	shifts = [
		{"number_inpatient": 0, "shifted_number_inpatient": 6},
		{"number_emergency": 0, "shifted_number_emergency": 4},
		{"number_diagnoses": 4, "shifted_number_diagnoses": 9},
		{"age": "[40-50)", "shifted_age": "[70-80)"},
		{"number_inpatient": 1, "shifted_number_inpatient": 5},
		{"number_emergency": 1, "shifted_number_emergency": 5},
		{"number_diagnoses": 5, "shifted_number_diagnoses": 9},
		{"age": "[50-60)", "shifted_age": "[80-90)"},
		{"number_inpatient": 2, "shifted_number_inpatient": 6},
		{"number_emergency": 2, "shifted_number_emergency": 6},
		{"number_diagnoses": 6, "shifted_number_diagnoses": 9},
		{"age": "[60-70)", "shifted_age": "[30-40)"},
		{
			"number_inpatient": 0,
			"shifted_number_inpatient": 6,
			"number_emergency": 0,
			"shifted_number_emergency": 4,
		},
		{
			"number_outpatient": 0,
			"shifted_number_outpatient": 6,
			"number_diagnoses": 4,
			"shifted_number_diagnoses": 9,
		},
		{
			"age": "[40-50)",
			"shifted_age": "[70-80)",
			"number_inpatient": 0,
			"shifted_number_inpatient": 6,
		},
		{
			"age": "[50-60)",
			"shifted_age": "[80-90)",
			"number_emergency": 0,
			"shifted_number_emergency": 4,
		},
		{
			"number_inpatient": 1,
			"shifted_number_inpatient": 6,
			"number_diagnoses": 5,
			"shifted_number_diagnoses": 9,
		},
		{
			"number_emergency": 0,
			"shifted_number_emergency": 6,
			"number_diagnoses": 4,
			"shifted_number_diagnoses": 9,
		},
		{
			"age": "[30-40)",
			"shifted_age": "[70-80)",
			"number_outpatient": 0,
			"shifted_number_outpatient": 6,
		},
		{
			"age": "[40-50)",
			"shifted_age": "[80-90)",
			"number_inpatient": 1,
			"shifted_number_inpatient": 6,
			"number_diagnoses": 5,
			"shifted_number_diagnoses": 9,
		},
	]

	templates = _scenario_templates(len(shifts))
	pairs = {}
	for index, (shift, template) in enumerate(zip(shifts, templates), start=1):
		base = deepcopy(template)
		shifted = deepcopy(base)
		for key, value in shift.items():
			if key.startswith("shifted_"):
				shifted[key.removeprefix("shifted_")] = value
			elif f"shifted_{key}" not in shift:
				base[key] = value
				shifted[key] = value
			else:
				base[key] = value

		pairs[f"Q{index:03d}"] = {"base": base, "shifted": shifted}
	return pairs


def _build_state_shuffled_memory_bank(seed=RANDOM_SEED):
	"""Return a copy of memory_df with the STATE_COLS tuple permuted across rows.

	For every memory, CONTEXT_COLS, EXPERIENCE_COLS, and outcome/governance
	columns (readmitted, feedback, quality_score, quarantined, memory_id)
	stay attached to their original row. Only the STATE_COLS tuple --
	(age, gender, race, number_outpatient, number_emergency,
	number_inpatient, number_diagnoses) -- is replaced, as a complete tuple,
	with another real memory's STATE_COLS tuple, via one deterministic
	random permutation over all rows (np.random.default_rng(seed)).

	Shuffling the whole tuple (rather than each STATE_COLS field
	independently) means every substituted state is still a real,
	internally-consistent patient state -- it simply now belongs to the
	wrong record -- which is what deliberately decouples state from
	semantic/contextual content without inventing impossible combinations.

	memory_df itself is never mutated: this function only ever reads from it
	and returns a new DataFrame.
	"""
	rng = np.random.default_rng(seed)
	permutation = rng.permutation(len(memory_df))
	base = memory_df.reset_index(drop=True)
	permuted_state = base[STATE_COLS].iloc[permutation].reset_index(drop=True)
	shuffled = base.copy()
	shuffled[STATE_COLS] = permuted_state

	non_state_cols = [c for c in base.columns if c not in STATE_COLS]
	assert shuffled[non_state_cols].equals(base[non_state_cols]), (
		"state shuffle must not alter any non-STATE_COLS column"
	)
	assert not shuffled[STATE_COLS].equals(base[STATE_COLS]), (
		"state shuffle did not actually change STATE_COLS"
	)
	return shuffled


def _state_scores(patient, retrieved, memory_bank):
	return np.array(
		[
			calculate_state_alignment(patient, row, memory_bank)
			for _, row in retrieved.iterrows()
		],
		dtype=float,
	)


def _metrics(patient, baseline_results, sacr_results, memory_bank):
	baseline_scores = _state_scores(patient, baseline_results, memory_bank)
	sacr_scores = sacr_results["state_alignment"].to_numpy(dtype=float)
	return {
		"baseline_avg_state_alignment": float(np.mean(baseline_scores)),
		"baseline_min_state_alignment": float(np.min(baseline_scores)),
		"baseline_misaligned_count": int(
			np.sum(baseline_scores < STATE_ALIGNMENT_THRESHOLD)
		),
		"baseline_retrieved_count": int(len(baseline_scores)),
		"sacr_avg_state_alignment": (
			float(np.mean(sacr_scores)) if len(sacr_scores) else float("nan")
		),
		"sacr_min_state_alignment": (
			float(np.min(sacr_scores)) if len(sacr_scores) else float("nan")
		),
		"sacr_misaligned_count": int(
			np.sum(sacr_scores < STATE_ALIGNMENT_THRESHOLD)
		),
		"sacr_retrieved_count": int(len(sacr_scores)),
	}


def _retrieved_memory_rows(
	query_id, query_version, method, retrieved, patient, memory_bank, bank_condition
):
	rows = []
	for rank, (_, result) in enumerate(retrieved.iterrows(), start=1):
		is_sacr = method == "sacr"
		rows.append(
			{
				"experiment_id": EXPERIMENT_ID,
				"bank_condition": bank_condition,
				"query_id": query_id,
				"query_version": query_version,
				"retrieval_method": method,
				"rank": rank,
				"memory_id": result["memory_id"],
				"semantic_similarity": float(
					result["semantic_similarity"] if is_sacr else result["similarity_score"]
				),
				"state_alignment": float(
					result["state_alignment"]
					if is_sacr
					else calculate_state_alignment(patient, result, memory_bank)
				),
				"context_alignment": (
					float(result["context_alignment"]) if is_sacr else None
				),
				"experience_alignment": (
					float(result["experience_alignment"]) if is_sacr else None
				),
				"final_ranking_score": (
					float(result["final_ranking_score"]) if is_sacr else None
				),
				"readmitted": result["readmitted"],
			}
		)
	return rows


def _run_condition(query_pairs, memory_bank, condition_name):
	"""Run baseline + SACR retrieval for all 40 query instances against one
	memory bank condition ("normal" or "shuffled"). Returns
	(query_rows, memory_rows) for that condition only.
	"""
	condition_query_rows = []
	condition_memory_rows = []
	metric_lookup = {}

	for query_id, versions in query_pairs.items():
		for query_version in ("base", "shifted"):
			patient = versions[query_version]
			baseline_results = semantic_retrieve(
				_state_neutral_query_text(patient), top_k=TOP_K, memory_bank=memory_bank
			)
			sacr_results = sacr_retrieve(
				patient,
				active_rules,
				top_k=TOP_K,
				candidate_k=CANDIDATE_K,
				memory_bank=memory_bank,
			)
			metrics = _metrics(patient, baseline_results, sacr_results, memory_bank)
			metric_lookup[(query_id, query_version)] = metrics
			condition_memory_rows.extend(
				_retrieved_memory_rows(
					query_id, query_version, "baseline", baseline_results,
					patient, memory_bank, condition_name,
				)
			)
			condition_memory_rows.extend(
				_retrieved_memory_rows(
					query_id, query_version, "sacr", sacr_results,
					patient, memory_bank, condition_name,
				)
			)
			condition_query_rows.append(
				{
					"experiment_id": EXPERIMENT_ID,
					"bank_condition": condition_name,
					"query_id": query_id,
					"query_version": query_version,
					**metrics,
					"alignment_improvement": (
						metrics["sacr_avg_state_alignment"]
						- metrics["baseline_avg_state_alignment"]
					),
					"misalignment_reduction": (
						metrics["baseline_misaligned_count"]
						- metrics["sacr_misaligned_count"]
					),
					"baseline_shift_drop": None,
					"sacr_shift_drop": None,
				}
			)

	for row in condition_query_rows:
		if row["query_version"] != "shifted":
			continue
		base_metrics = metric_lookup[(row["query_id"], "base")]
		row["baseline_shift_drop"] = (
			base_metrics["baseline_avg_state_alignment"]
			- row["baseline_avg_state_alignment"]
		)
		row["sacr_shift_drop"] = (
			base_metrics["sacr_avg_state_alignment"]
			- row["sacr_avg_state_alignment"]
		)

	return condition_query_rows, condition_memory_rows


def _condition_summary(query_results, condition, method):
	subset = query_results[query_results["bank_condition"] == condition]
	avg_col = f"{method}_avg_state_alignment"
	min_col = f"{method}_min_state_alignment"
	misaligned_col = f"{method}_misaligned_count"
	retrieved_col = f"{method}_retrieved_count"
	total_retrieved = int(subset[retrieved_col].sum())
	misaligned_count = int(subset[misaligned_col].sum())
	return {
		"bank_condition": condition,
		"method": method,
		"n_query_instances": int(len(subset)),
		"mean_state_alignment": float(subset[avg_col].mean()),
		"min_state_alignment": float(subset[min_col].min()),
		"misaligned_count": misaligned_count,
		"total_retrieved": total_retrieved,
		"misalignment_rate": (
			misaligned_count / total_retrieved if total_retrieved else float("nan")
		),
	}


def _paired_stats(baseline_values, sacr_values):
	"""Paired baseline-vs-SACR comparison on per-query-instance mean alignment.

	Reports a paired t-test (parametric) and a Wilcoxon signed-rank test
	(non-parametric, no normality assumption) since each query instance
	yields one baseline value and one SACR value -- a natural pairing.
	Neither test is used to claim significance beyond what its own p-value
	supports.
	"""
	baseline_values = np.asarray(baseline_values, dtype=float)
	sacr_values = np.asarray(sacr_values, dtype=float)
	diffs = sacr_values - baseline_values
	n = len(diffs)
	mean_diff = float(np.mean(diffs))
	median_diff = float(np.median(diffs))
	std_diff = float(np.std(diffs, ddof=1)) if n > 1 else float("nan")

	if n > 1 and std_diff > 0:
		se = std_diff / np.sqrt(n)
		t_crit = float(scipy_stats.t.ppf(0.975, df=n - 1))
		ci_low = mean_diff - t_crit * se
		ci_high = mean_diff + t_crit * se
		t_stat, t_pvalue = scipy_stats.ttest_rel(sacr_values, baseline_values)
	else:
		ci_low = ci_high = mean_diff
		t_stat, t_pvalue = float("nan"), float("nan")

	try:
		wilcoxon_stat, wilcoxon_pvalue = scipy_stats.wilcoxon(sacr_values, baseline_values)
	except ValueError:
		# raised when all paired differences are zero / too few non-zero diffs
		wilcoxon_stat, wilcoxon_pvalue = float("nan"), float("nan")

	return {
		"n": n,
		"mean_diff": mean_diff,
		"median_diff": median_diff,
		"std_diff": std_diff,
		"ci95_low": float(ci_low),
		"ci95_high": float(ci_high),
		"paired_t_stat": float(t_stat),
		"paired_t_pvalue": float(t_pvalue),
		"wilcoxon_stat": float(wilcoxon_stat),
		"wilcoxon_pvalue": float(wilcoxon_pvalue),
	}


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


def _format_metric(value):
	return f"{value:.6f}" if isinstance(value, float) else str(value)


def _write_report(query_results, aggregate_rows, summary_rows, paired_stats, metadata):
	aggregate = dict(aggregate_rows)
	summary_by_key = {(row["bank_condition"], row["method"]): row for row in summary_rows}

	def cond(bank_condition, method):
		return summary_by_key[(bank_condition, method)]

	a_base, a_sacr = cond("normal", "baseline"), cond("normal", "sacr")
	b_base, b_sacr = cond("shuffled", "baseline"), cond("shuffled", "sacr")

	lines = []
	lines.append("=" * 60)
	lines.append("RQ1 -- SACR STATE-AWARE RETRIEVAL EXPERIMENT")
	lines.append("=" * 60)
	lines.append("")
	lines.append("## Research Question")
	lines.append("")
	lines.append(
		"Can state-aware retrieval reduce the retrieval of semantically"
	)
	lines.append(
		"relevant but state-misaligned experiences compared with"
	)
	lines.append("conventional semantic retrieval?")
	lines.append("")
	lines.append("## Experimental Setup")
	lines.append("")
	lines.append("Dataset: UCI Diabetes 130-US Hospitals (1999-2008)")
	lines.append(f"Number of memories: {metadata['number_of_memories']}")
	lines.append(f"Number of query scenarios (pairs): {metadata['number_of_query_pairs']}")
	lines.append(
		"Number of query instances per condition (base+shifted): "
		f"{metadata['number_of_query_instances_per_condition']}"
	)
	lines.append(f"Retrieval K: {metadata['top_k']}")
	lines.append(f"SACR candidate pool size: {metadata['candidate_k']}")
	lines.append(f"State alignment threshold: {metadata['state_alignment_threshold']}")
	lines.append(
		"SACR weights: semantic="
		f"{metadata['semantic_weight']} context={metadata['context_weight']} "
		f"experience={metadata['experience_weight']}"
	)
	lines.append(f"Random seed: {metadata['random_seed']}")
	lines.append(f"State-shuffle seed: {metadata['state_shuffle_seed']}")
	lines.append(f"Embedding model: {metadata['embedding_model']}")

	def experiment_block(title, base_row, sacr_row, with_header=True):
		if with_header:
			lines.append("")
			lines.append("-" * 60)
			lines.append(title)
			lines.append("-" * 60)
			lines.append("")
		lines.append("Baseline:")
		lines.append(f"  Mean state alignment: {base_row['mean_state_alignment']:.6f}")
		lines.append(f"  Minimum state alignment: {base_row['min_state_alignment']:.6f}")
		lines.append(f"  Misaligned memories: {base_row['misaligned_count']}")
		lines.append(f"  Total retrieved: {base_row['total_retrieved']}")
		lines.append(f"  Misalignment rate: {base_row['misalignment_rate']:.2%}")
		lines.append("")
		lines.append("SACR:")
		lines.append(f"  Mean state alignment: {sacr_row['mean_state_alignment']:.6f}")
		lines.append(f"  Minimum state alignment: {sacr_row['min_state_alignment']:.6f}")
		lines.append(f"  Misaligned memories: {sacr_row['misaligned_count']}")
		lines.append(f"  Total retrieved: {sacr_row['total_retrieved']}")
		lines.append(f"  Misalignment rate: {sacr_row['misalignment_rate']:.2%}")
		lines.append("")
		lines.append(
			"SACR improvement (mean alignment): "
			f"{sacr_row['mean_state_alignment'] - base_row['mean_state_alignment']:+.6f}"
		)
		lines.append(
			"SACR improvement (misalignment rate): "
			f"{sacr_row['misalignment_rate'] - base_row['misalignment_rate']:+.2%}"
		)

	experiment_block("EXPERIMENT A -- NORMAL MEMORY BANK", a_base, a_sacr)

	lines.append("")
	lines.append("-" * 60)
	lines.append("EXPERIMENT B -- STATE-SHUFFLED MEMORY BANK")
	lines.append("-" * 60)
	lines.append("")
	lines.append(
		"What was shuffled: for every memory, CONTEXT_COLS and EXPERIENCE_COLS,"
	)
	lines.append(
		"plus outcome/governance columns (readmitted, feedback, quality_score,"
	)
	lines.append(
		"quarantined, memory_id), were left exactly as in the original record."
	)
	lines.append(
		"The STATE_COLS tuple (age, gender, race, number_outpatient,"
	)
	lines.append(
		"number_emergency, number_inpatient, number_diagnoses) was replaced,"
	)
	lines.append(
		"as a complete tuple, with another real memory's STATE_COLS tuple, via"
	)
	lines.append(
		"one fixed random permutation (np.random.default_rng(42)) over all"
	)
	lines.append(
		"memory rows. Every substituted state is therefore still a real,"
	)
	lines.append(
		"internally-consistent patient state -- it now just belongs to the"
	)
	lines.append(
		"wrong record -- deliberately decoupling state from the semantic/"
	)
	lines.append(
		"contextual content used for retrieval. memory_df is not mutated;"
	)
	lines.append(
		"embeddings are unaffected because they are built from state-neutral"
	)
	lines.append("text (CONTEXT_COLS + EXPERIENCE_COLS only).")
	lines.append("")

	experiment_block(None, b_base, b_sacr, with_header=False)

	lines.append("")
	lines.append("-" * 60)
	lines.append("SIDE-BY-SIDE SUMMARY")
	lines.append("-" * 60)
	lines.append("")
	lines.append(f"{'Condition':<30}{'Baseline':>14}{'SACR':>14}")
	lines.append(
		f"{'Normal mean alignment':<30}{a_base['mean_state_alignment']:>14.4f}"
		f"{a_sacr['mean_state_alignment']:>14.4f}"
	)
	lines.append(
		f"{'Normal misalignment rate':<30}{a_base['misalignment_rate']:>13.2%} "
		f"{a_sacr['misalignment_rate']:>13.2%}"
	)
	lines.append(
		f"{'Shuffled mean alignment':<30}{b_base['mean_state_alignment']:>14.4f}"
		f"{b_sacr['mean_state_alignment']:>14.4f}"
	)
	lines.append(
		f"{'Shuffled misalignment rate':<30}{b_base['misalignment_rate']:>13.2%} "
		f"{b_sacr['misalignment_rate']:>13.2%}"
	)
	lines.append("")
	lines.append("Paired comparison (per query instance, SACR - baseline mean alignment):")
	for condition in ("normal", "shuffled"):
		ps = paired_stats[condition]
		lines.append(
			f"  {condition}: n={ps['n']}  mean_diff={ps['mean_diff']:.6f}  "
			f"median_diff={ps['median_diff']:.6f}  std_diff={ps['std_diff']:.6f}  "
			f"95% CI=({ps['ci95_low']:.6f}, {ps['ci95_high']:.6f})"
		)
		lines.append(
			f"    paired t-test: t={ps['paired_t_stat']:.4f}, p={ps['paired_t_pvalue']:.4g}"
			f"   Wilcoxon signed-rank: W={ps['wilcoxon_stat']:.4f}, p={ps['wilcoxon_pvalue']:.4g}"
		)
	lines.append("")
	lines.append(
		"Test choice: a paired t-test (parametric) and a Wilcoxon signed-rank test"
	)
	lines.append(
		"(non-parametric, no normality assumption) are both reported because each"
	)
	lines.append(
		"query instance yields exactly one baseline and one SACR mean-alignment"
	)
	lines.append(
		"value -- a natural pairing. Agreement between the two is the more"
	)
	lines.append("conservative basis for any claim of a real difference.")
	lines.append(
		"No significance test is reported for misalignment COUNTS: SACR's hard"
	)
	lines.append(
		"threshold drives its count to (near) zero, which makes a count-based"
	)
	lines.append("test degenerate; raw counts/rates are reported above instead.")

	lines.append("")
	lines.append("-" * 60)
	lines.append("BASE -> SHIFTED ANALYSIS (secondary / sensitivity analysis)")
	lines.append("-" * 60)
	lines.append("")
	lines.append(
		"Caveat: the query's semantic text is state-neutral (CONTEXT_COLS +"
	)
	lines.append(
		"EXPERIENCE_COLS only), so BASE and SHIFTED share identical text and the"
	)
	lines.append(
		"baseline retriever returns the SAME memories for both versions of a"
	)
	lines.append(
		"pair. The 'shift drop' below therefore reflects only whether the"
	)
	lines.append(
		"shifted state happens to be farther from that fixed retrieved set, not"
	)
	lines.append(
		"a genuine re-retrieval failure. It is NOT direct causal evidence that"
	)
	lines.append(
		"the retriever 'got worse' -- Experiments A and B above are the primary"
	)
	lines.append("evidence for RQ1.")
	lines.append("")
	normal_results = query_results[query_results["bank_condition"] == "normal"]
	lines.append(
		f"Mean baseline shift drop (normal condition): "
		f"{_format_metric(aggregate['mean baseline shift drop'])}"
	)
	lines.append(
		f"Mean SACR shift drop (normal condition): "
		f"{_format_metric(aggregate['mean SACR shift drop'])}"
	)
	lines.append("")
	for query_id in sorted(normal_results["query_id"].unique()):
		lines.append(query_id)
		query_rows = normal_results[normal_results["query_id"] == query_id]
		for version in ("base", "shifted"):
			row = query_rows[query_rows["query_version"] == version].iloc[0]
			lines.append(f"  {version.upper()}")
			lines.append(
				f"    Baseline average alignment: {row['baseline_avg_state_alignment']:.6f}"
			)
			lines.append(
				f"    Baseline minimum alignment: {row['baseline_min_state_alignment']:.6f}"
			)
			lines.append(
				f"    Baseline misaligned: {int(row['baseline_misaligned_count'])}"
				f"/{int(row['baseline_retrieved_count'])}"
			)
			lines.append(
				f"    SACR average alignment: {row['sacr_avg_state_alignment']:.6f}"
			)
			lines.append(
				f"    SACR minimum alignment: {row['sacr_min_state_alignment']:.6f}"
			)
			lines.append(
				f"    SACR misaligned: {int(row['sacr_misaligned_count'])}"
				f"/{int(row['sacr_retrieved_count'])}"
			)
			lines.append("")
		shifted_row = query_rows[query_rows["query_version"] == "shifted"].iloc[0]
		lines.append(f"    Baseline shift drop: {shifted_row['baseline_shift_drop']:.6f}")
		lines.append(f"    SACR shift drop: {shifted_row['sacr_shift_drop']:.6f}")
		lines.append("")

	lines.append("-" * 60)
	lines.append("INTERPRETATION")
	lines.append("-" * 60)
	lines.append("")
	lines.append("1. Evidence supporting SACR's state-alignment property:")
	lines.append(
		f"   - Normal condition: SACR misalignment rate {a_sacr['misalignment_rate']:.2%} vs "
		f"baseline {a_base['misalignment_rate']:.2%} "
		f"({a_sacr['misaligned_count']} vs {a_base['misaligned_count']} misaligned out of "
		f"{a_base['total_retrieved']})."
	)
	lines.append(
		f"   - Shuffled condition: SACR misalignment rate {b_sacr['misalignment_rate']:.2%} vs "
		f"baseline {b_base['misalignment_rate']:.2%} "
		f"({b_sacr['misaligned_count']} vs {b_base['misaligned_count']} misaligned out of "
		f"{b_base['total_retrieved']})."
	)
	ps_normal, ps_shuffled = paired_stats["normal"], paired_stats["shuffled"]
	lines.append(
		f"   - Paired mean-alignment improvement is {ps_normal['mean_diff']:+.4f} "
		f"(normal, t p={ps_normal['paired_t_pvalue']:.4g}, "
		f"Wilcoxon p={ps_normal['wilcoxon_pvalue']:.4g}) and "
		f"{ps_shuffled['mean_diff']:+.4f} (shuffled, t p={ps_shuffled['paired_t_pvalue']:.4g}, "
		f"Wilcoxon p={ps_shuffled['wilcoxon_pvalue']:.4g})."
	)
	lines.append("")
	lines.append("2. Evidence about baseline vulnerability:")
	lines.append(
		f"   - Baseline misalignment rate: {a_base['misalignment_rate']:.2%} (normal) -> "
		f"{b_base['misalignment_rate']:.2%} (shuffled), i.e. when state was deliberately "
		"decoupled from semantic/contextual content."
	)
	if b_base["misalignment_rate"] > a_base["misalignment_rate"]:
		lines.append(
			"   - This is consistent with RQ1: a semantic-only retriever cannot detect that"
		)
		lines.append(
			"     state has been decoupled from content, so it keeps retrieving the same"
		)
		lines.append(
			"     semantically-plausible memories even once they are state-incompatible."
		)
	else:
		lines.append(
			"   - The shuffled condition did NOT increase baseline misalignment relative to"
		)
		lines.append(
			"     normal. This weakens the vulnerability claim at the misalignment-rate level"
		)
		lines.append(
			"     and should be reported as such; see limitations below for why."
		)
	lines.append("")
	lines.append("3. Limitations of this experiment:")
	lines.append(
		"   - STATE_COLS alignment is a graded average over 3 categorical and 4 numeric"
	)
	lines.append(
		"     fields; because most numeric STATE_COLS values cluster near zero across the"
	)
	lines.append(
		"     dataset, even a randomly assigned state can exceed the 0.70 threshold by"
	)
	lines.append(
		"     chance, which caps how large the shuffled-vs-normal gap can get."
	)
	lines.append(
		"   - The BASE->SHIFTED analysis is confounded by state-neutral query text (see"
	)
	lines.append(
		"     above) and is reported only as a secondary/sensitivity check."
	)
	lines.append(
		"   - The 20 query scenarios are sampled once (seed 42) from the real memory bank;"
	)
	lines.append(
		"     results may shift somewhat under a different sample, though the mechanism"
	)
	lines.append(
		"     being tested (hard state filter vs. no filter) does not depend on that sample."
	)
	lines.append("")
	lines.append("-" * 60)
	lines.append("REPRODUCIBILITY")
	lines.append("-" * 60)
	lines.append("")
	lines.append(f"Random seed: {metadata['random_seed']}")
	lines.append(f"State-shuffle seed: {metadata['state_shuffle_seed']}")
	lines.append("Files generated:")
	lines.append("  results/rq1/query_results.csv")
	lines.append("  results/rq1/retrieved_memories.csv")
	lines.append("  results/rq1/state_shuffle_results.csv")
	lines.append("  results/rq1/rq1_summary.csv")
	lines.append("  results/rq1/aggregate_results.csv")
	lines.append("  results/rq1/experiment_metadata.json")
	lines.append("  results/rq1/test_queries.json")
	lines.append("  results/rq1/experiment_report.txt")
	lines.append("Embedding file: data/baseline_embeddings.npy (state-neutral text)")
	lines.append("Dataset cache: data/diabetic_data_raw.pkl")
	lines.append(f"Timestamp: {metadata['timestamp']}")
	lines.append(f"Git commit: {metadata['git_commit']}")
	lines.append(f"Python version: {metadata['python_version']}")
	lines.append("")

	report_text = "\n".join(lines) + "\n"
	(RESULTS_DIR / "experiment_report.txt").write_text(report_text, encoding="utf-8")


def run_rq1():
	np.random.seed(RANDOM_SEED)
	RESULTS_DIR.mkdir(parents=True, exist_ok=True)
	query_pairs = _make_query_pairs()
	shuffled_bank = _build_state_shuffled_memory_bank(RANDOM_SEED)

	normal_query_rows, normal_memory_rows = _run_condition(query_pairs, memory_df, "normal")
	shuffled_query_rows, shuffled_memory_rows = _run_condition(
		query_pairs, shuffled_bank, "shuffled"
	)

	query_results = pd.DataFrame(normal_query_rows + shuffled_query_rows)
	retrieved_memories = pd.DataFrame(normal_memory_rows + shuffled_memory_rows)
	query_results.to_csv(RESULTS_DIR / "query_results.csv", index=False)
	retrieved_memories.to_csv(RESULTS_DIR / "retrieved_memories.csv", index=False)
	query_results[query_results["bank_condition"] == "shuffled"].to_csv(
		RESULTS_DIR / "state_shuffle_results.csv", index=False
	)

	normal_results = query_results[query_results["bank_condition"] == "normal"]
	shifted_normal_results = normal_results[normal_results["query_version"] == "shifted"]
	aggregate_rows = [
		("mean baseline state alignment", normal_results["baseline_avg_state_alignment"].mean()),
		("mean SACR state alignment", normal_results["sacr_avg_state_alignment"].mean()),
		("mean alignment improvement", normal_results["alignment_improvement"].mean()),
		("total baseline misaligned memories", normal_results["baseline_misaligned_count"].sum()),
		("total SACR misaligned memories", normal_results["sacr_misaligned_count"].sum()),
		("mean baseline shift drop", shifted_normal_results["baseline_shift_drop"].mean()),
		("mean SACR shift drop", shifted_normal_results["sacr_shift_drop"].mean()),
		(
			"number of queries where SACR has higher alignment",
			int(np.sum(normal_results["alignment_improvement"] > 0)),
		),
		(
			"number of queries where SACR has fewer misaligned memories",
			int(
				np.sum(
					normal_results["sacr_misaligned_count"]
					< normal_results["baseline_misaligned_count"]
				)
			),
		),
	]
	pd.DataFrame(aggregate_rows, columns=["metric", "value"]).to_csv(
		RESULTS_DIR / "aggregate_results.csv", index=False
	)

	summary_rows = [
		_condition_summary(query_results, "normal", "baseline"),
		_condition_summary(query_results, "normal", "sacr"),
		_condition_summary(query_results, "shuffled", "baseline"),
		_condition_summary(query_results, "shuffled", "sacr"),
	]
	pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "rq1_summary.csv", index=False)

	shuffled_results = query_results[query_results["bank_condition"] == "shuffled"]
	paired_stats = {
		"normal": _paired_stats(
			normal_results["baseline_avg_state_alignment"],
			normal_results["sacr_avg_state_alignment"],
		),
		"shuffled": _paired_stats(
			shuffled_results["baseline_avg_state_alignment"],
			shuffled_results["sacr_avg_state_alignment"],
		),
	}

	metadata = {
		"experiment": "RQ1",
		"description": (
			"Baseline semantic retrieval vs SACR under a normal memory bank "
			"(Experiment A) and a state-shuffled memory bank (Experiment B), "
			"plus a secondary BASE-SHIFTED sensitivity analysis."
		),
		"top_k": TOP_K,
		"candidate_k": CANDIDATE_K,
		"state_alignment_threshold": STATE_ALIGNMENT_THRESHOLD,
		"semantic_weight": LAMBDA_SEMANTIC,
		"context_weight": LAMBDA_CONTEXT,
		"experience_weight": LAMBDA_EXPERIENCE,
		"number_of_query_pairs": len(query_pairs),
		"number_of_query_instances_per_condition": len(normal_results),
		"random_seed": RANDOM_SEED,
		"state_shuffle_seed": RANDOM_SEED,
		"timestamp": datetime.now(timezone.utc).isoformat(),
		"number_of_memories": len(memory_df),
		"embedding_model": EMBEDDING_MODEL_NAME,
		"paired_statistics": paired_stats,
		"python_version": sys.version.split()[0],
		"git_commit": _git_commit_hash(),
	}
	(RESULTS_DIR / "experiment_metadata.json").write_text(
		json.dumps(metadata, indent=4), encoding="utf-8"
	)
	(RESULTS_DIR / "test_queries.json").write_text(
		json.dumps(query_pairs, indent=4), encoding="utf-8"
	)

	_write_report(query_results, aggregate_rows, summary_rows, paired_stats, metadata)

	print("========================================")
	print("RQ1 SUMMARY (full detail in results/rq1/experiment_report.txt)")
	print("========================================")
	for row in summary_rows:
		print(
			f"{row['bank_condition']:>9} / {row['method']:<8} "
			f"mean_alignment={row['mean_state_alignment']:.4f} "
			f"misalignment_rate={row['misalignment_rate']:.2%} "
			f"({row['misaligned_count']}/{row['total_retrieved']})"
		)
	print("\nResults saved to:")
	print("results/rq1/")


if __name__ == "__main__":
	run_rq1()
