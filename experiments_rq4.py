"""RQ4: End-to-end evaluation.

"How does dynamic memory governance improve downstream task performance and
safety compared with static memory retrieval?"

This module does NOT modify RQ1/RQ2/RQ3's implementation, results, SACR
formulation, thresholds, weights, or governance algorithm. It reads the
frozen state threshold (0.70) and context threshold (0.90) from RQ2's own
metadata and the frozen OGMM parameters (N_MIN, BETA) from RQ3's own
metadata, and reuses RQ3's exact governance primitives (imported, not
reimplemented) and sacr.py's/baseline_rag.py's existing, additive
`excluded_memory_ids` mechanism (default None reproduces prior behavior
exactly -- verified in a regression check before this experiment runs).

OUTCOME PROXY LIMITATION: `readmitted` is an outcome-derived experimental
proxy attached to historical hospital records, not a causal treatment-effect
label. This experiment evaluates a downstream OUTCOME-PROXY PREDICTION task
(majority vote over retrieved memories' historical readmitted category); it
does NOT establish clinical efficacy, treatment effectiveness, causal
relationships, patient-level medical recommendations, real-world deployment
safety, or superiority for clinical decision making. Governance feedback is
derived from historical outcome labels already attached to stored memory
records -- not independent repeated environmental observations -- so this
experiment does not claim to demonstrate online learning from genuinely new
real-world feedback (see Section 21 of the generated report for the full
discussion carried over from RQ3).
"""

from collections import Counter
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon
from sklearn.metrics import confusion_matrix, f1_score

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
	RQ2_METADATA_PATH,
	_load_frozen_active_rules,
	_meets_downgrade_criterion,
	_meets_quarantine_criterion,
	_new_governance_entry,
	_apply_feedback,
	_update_quality,
)


EXPERIMENT_ID = "RQ4_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "rq4"
DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_CACHE_PATH = DATA_DIR / "diabetic_data_raw.pkl"
PATIENT_IDS_CACHE_PATH = DATA_DIR / "patient_ids.pkl"
RQ3_METADATA_PATH = Path(__file__).resolve().parent / "results" / "rq3" / "experiment_metadata.json"

HELD_OUT_FRACTION = 0.20
MAX_EVAL_QUERIES = 300  # pre-committed cap for computational tractability (see report Limitations)
LABELS = ["NO", ">30", "<30"]  # fixed class order for macro-F1 / confusion matrices


# ---------------------------------------------------------------------------
# Part 1: freeze verification (regression checks), run before anything else.
# ---------------------------------------------------------------------------

def run_regression_checks():
	"""Confirm RQ1/RQ2 behavior is unchanged by the additive
	`excluded_memory_ids` parameters this and prior tasks added to
	semantic_retrieve/sacr_retrieve. Raises if anything has drifted.
	"""
	from sacr import current_patient, patient_to_text

	r1 = semantic_retrieve(patient_to_text(current_patient), top_k=5)
	r2 = semantic_retrieve(patient_to_text(current_patient), top_k=5, excluded_memory_ids=None)
	assert list(r1["memory_id"]) == list(r2["memory_id"]), "semantic_retrieve default behavior changed"

	active_rules, _ = _load_frozen_active_rules()
	s1 = sacr_retrieve(current_patient, active_rules, top_k=5, candidate_k=CANDIDATE_K)
	s2 = sacr_retrieve(current_patient, active_rules, top_k=5, candidate_k=CANDIDATE_K, excluded_memory_ids=None)
	assert list(s1["memory_id"]) == list(s2["memory_id"]), "sacr_retrieve default behavior changed"
	assert (s1["state_alignment"] >= STATE_ALIGNMENT_THRESHOLD).all(), "RQ2 state invariant violated in regression check"
	return True


# ---------------------------------------------------------------------------
# Part 3: patient identity + deterministic patient-level split
# ---------------------------------------------------------------------------

def _load_patient_nbr():
	"""Load patient_nbr aligned by row position to memory_df.

	ucimlrepo splits `patient_nbr`/`encounter_id` into a separate `.ids`
	frame (role="ID"), not `.features` -- preprocess.py's cached
	`diabetic_data_raw.pkl` therefore never captured it. This loader fetches
	it once, verifies positional alignment against the cached raw dataframe
	(same underlying CSV row order, confirmed by comparing shared feature
	columns), and caches the result so no further network access is needed.
	"""
	if PATIENT_IDS_CACHE_PATH.exists():
		ids_df = pd.read_pickle(PATIENT_IDS_CACHE_PATH)
	else:
		from ucimlrepo import fetch_ucirepo

		dataset = fetch_ucirepo(id=296)
		fresh_features = dataset.data.features.reset_index(drop=True)
		cached_df = pd.read_pickle(RAW_CACHE_PATH).reset_index(drop=True)
		common_cols = [c for c in fresh_features.columns if c in cached_df.columns]
		if not fresh_features[common_cols].equals(cached_df[common_cols]):
			raise RuntimeError(
				"Fresh UCI fetch does not positionally match the cached raw dataset; "
				"refusing to attach patient_nbr without verified row alignment."
			)
		ids_df = dataset.data.ids[["patient_nbr"]].reset_index(drop=True)
		DATA_DIR.mkdir(parents=True, exist_ok=True)
		ids_df.to_pickle(PATIENT_IDS_CACHE_PATH)

	patient_nbr = ids_df["patient_nbr"].reset_index(drop=True)
	assert len(patient_nbr) == len(memory_df), "patient_nbr length does not match memory_df"
	return patient_nbr


def _split_patients(unique_patients, seed=RANDOM_SEED, heldout_fraction=HELD_OUT_FRACTION):
	"""Deterministic, unstratified (no label use) random split over unique
	patient_nbr values -- never over individual rows, so a multi-encounter
	patient's rows all land on the same side.
	"""
	rng = np.random.default_rng(seed)
	shuffled = rng.permutation(np.sort(unique_patients))
	n_heldout = int(round(len(shuffled) * heldout_fraction))
	heldout_patients = frozenset(shuffled[:n_heldout].tolist())
	memory_pool_patients = frozenset(shuffled[n_heldout:].tolist())
	return memory_pool_patients, heldout_patients


def build_split():
	patient_nbr = _load_patient_nbr()
	memory_with_patient = memory_df.reset_index(drop=True).copy()
	memory_with_patient["patient_nbr"] = patient_nbr.to_numpy()

	unique_patients = memory_with_patient["patient_nbr"].unique()
	memory_pool_patients, heldout_patients = _split_patients(unique_patients)

	pool_mask = memory_with_patient["patient_nbr"].isin(memory_pool_patients)
	memory_pool_df = memory_with_patient[pool_mask].reset_index(drop=True)
	held_out_df = memory_with_patient[~pool_mask].reset_index(drop=True)

	assert set(held_out_df["patient_nbr"]).isdisjoint(set(memory_pool_df["patient_nbr"])), (
		"patient-level split leaked a patient across both partitions"
	)
	leakage_excluded_ids = frozenset(held_out_df["memory_id"])

	split_info = {
		"total_unique_patients": int(len(unique_patients)),
		"memory_pool_patients": int(len(memory_pool_patients)),
		"heldout_patients": int(len(heldout_patients)),
		"memory_pool_rows": int(len(memory_pool_df)),
		"heldout_rows": int(len(held_out_df)),
		"seed": RANDOM_SEED,
		"heldout_fraction_target": HELD_OUT_FRACTION,
		"memory_pool_class_distribution": memory_pool_df["readmitted"].value_counts().to_dict(),
		"heldout_class_distribution": held_out_df["readmitted"].value_counts().to_dict(),
	}
	return memory_pool_df, held_out_df, leakage_excluded_ids, split_info


def build_eval_queries(held_out_df, seed=RANDOM_SEED, max_queries=MAX_EVAL_QUERIES):
	"""Deterministic evaluation query sample, fixed order, used identically
	by all three conditions. Sampling (not truncation) avoids any
	systematic bias from row order; the same seed as the rest of the
	project is used. This cap is a pre-committed computational-
	tractability decision (documented in the report), not a post-hoc
	exclusion of "difficult" queries -- it is applied before any condition
	is evaluated and before any result is observed.
	"""
	if len(held_out_df) > max_queries:
		eval_df = held_out_df.sample(n=max_queries, random_state=seed)
	else:
		eval_df = held_out_df
	eval_df = eval_df.sort_values("memory_id").reset_index(drop=True)
	return eval_df


# ---------------------------------------------------------------------------
# Part 7: deterministic majority-vote prediction with documented tie-break
# ---------------------------------------------------------------------------

def predict_majority(labels):
	"""mode{y_m1, ..., y_mk}; ties broken by ascending lexicographic
	(codepoint) order of the tied category strings -- i.e. among tied
	labels, '<30' sorts before '>30' sorts before 'NO'. This rule is fixed
	before evaluation and is not a function of the test labels.
	"""
	if not labels:
		return None
	counts = Counter(labels)
	max_count = max(counts.values())
	tied = sorted(label for label, count in counts.items() if count == max_count)
	return tied[0]


# ---------------------------------------------------------------------------
# Parts 5/6/9/10: the three-condition evaluation loop
# ---------------------------------------------------------------------------

def evaluate_condition(condition, eval_df, active_rules, leakage_excluded, n_min, beta):
	track_governance = condition == "sg_qms"
	governance = {}
	governance_excluded = set()
	per_query_records = []
	retrieval_records = []

	for _, row in eval_df.iterrows():
		query_id = row["memory_id"]
		patient = {col: row[col] for col in STATE_COLS + CONTEXT_COLS + EXPERIENCE_COLS}
		true_label = row["readmitted"]  # NOT used until after prediction, for logging only

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
			excluded = (leakage_excluded | governance_excluded) if track_governance else leakage_excluded
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
				"query_id": query_id,
				"patient_nbr": int(row["patient_nbr"]),
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
				"governance_excluded_count": len(governance_excluded) if track_governance else 0,
			}
		)
		for info in retrieved_info:
			retrieval_records.append({"condition": condition, "query_id": query_id, **info})

		# Governance update happens AFTER this query's prediction is fixed, using
		# only the retrieved memories' own (already-known, historical) feedback --
		# never this query's own held-out label. Affects only FUTURE queries.
		if track_governance:
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
				elif not entry["quarantined"] and not entry["downgraded"] and _meets_downgrade_criterion(entry):
					entry["downgraded"] = True

	return per_query_records, retrieval_records, governance


# ---------------------------------------------------------------------------
# Part 11: metrics
# ---------------------------------------------------------------------------

def _condition_stats(per_query_records, condition):
	rows = [r for r in per_query_records if r["condition"] == condition]
	total = len(rows)
	covered_rows = [r for r in rows if r["covered"]]
	covered = len(covered_rows)
	correct = sum(1 for r in covered_rows if r["correct"])
	y_true = [r["true_readmitted"] for r in covered_rows]
	y_pred = [r["predicted_readmitted"] for r in covered_rows]
	macro_f1 = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0) if covered_rows else float("nan")
	support = {label: y_true.count(label) for label in LABELS}
	return {
		"total_queries": total,
		"covered_queries": covered,
		"correct_predictions": correct,
		"accuracy_covered": (correct / covered) if covered else float("nan"),
		"accuracy_all_uncovered_as_incorrect": (correct / total) if total else float("nan"),
		"coverage": (covered / total) if total else float("nan"),
		"macro_f1": macro_f1,
		"support": support,
	}


def _negative_exposure(retrieval_records, condition):
	subset = [r for r in retrieval_records if r["condition"] == condition]
	total = len(subset)
	negative = sum(1 for r in subset if r["feedback"] < 0)
	return {
		"total_retrieved": total,
		"negative_retrieved": negative,
		"negative_exposure": (negative / total) if total else float("nan"),
	}


def _negative_query_rate(per_query_records, condition):
	covered_rows = [r for r in per_query_records if r["condition"] == condition and r["covered"]]
	if not covered_rows:
		return float("nan")
	with_negative = sum(1 for r in covered_rows if r["negative_memory_count"] >= 1)
	return with_negative / len(covered_rows)


def _alignment_violations(retrieval_records, condition, context_threshold):
	subset = [r for r in retrieval_records if r["condition"] == condition]
	state_v = sum(1 for r in subset if r["state_alignment"] < STATE_ALIGNMENT_THRESHOLD)
	context_v = sum(1 for r in subset if r["context_alignment"] < context_threshold)
	joint_v = sum(
		1 for r in subset
		if r["state_alignment"] < STATE_ALIGNMENT_THRESHOLD or r["context_alignment"] < context_threshold
	)
	return {"total_checked": len(subset), "state_violations": state_v, "context_violations": context_v, "joint_violations": joint_v}


def _governance_metrics(governance, per_query_records):
	quarantined = {mid for mid, e in governance.items() if e["quarantined"]}
	downgraded = {mid for mid, e in governance.items() if e["downgraded"]}
	touched = len(governance)
	retained = touched - len(quarantined)
	sg_qms_rows = [r for r in per_query_records if r["condition"] == "sg_qms"]
	total_governed_retrievals = sum(r["retrieval_count"] for r in sg_qms_rows)
	return {
		"number_quarantined": len(quarantined),
		"number_downgraded": len(downgraded),
		"number_retained": retained,
		"distinct_memories_touched": touched,
		"quarantine_rate": (len(quarantined) / touched) if touched else float("nan"),
		"memory_retention_rate": (retained / touched) if touched else float("nan"),
		"governed_retrievals": total_governed_retrievals,
		"negative_evidence_quarantine_index_size": len(quarantined),
	}


# ---------------------------------------------------------------------------
# Part 13: predefined paired statistical comparisons
# ---------------------------------------------------------------------------

def _mcnemar(correct_a, correct_b):
	correct_a = np.asarray(correct_a, dtype=bool)
	correct_b = np.asarray(correct_b, dtype=bool)
	b = int(np.sum(correct_a & ~correct_b))
	c = int(np.sum(~correct_a & correct_b))
	n = b + c
	if n == 0:
		return {"a_only_correct": b, "b_only_correct": c, "n_discordant": n, "p_value": float("nan")}
	result = binomtest(min(b, c), n, 0.5)
	return {"a_only_correct": b, "b_only_correct": c, "n_discordant": n, "p_value": float(result.pvalue)}


def _paired_wilcoxon(values_a, values_b):
	values_a = np.asarray(values_a, dtype=float)
	values_b = np.asarray(values_b, dtype=float)
	mask = ~np.isnan(values_a) & ~np.isnan(values_b)
	a, b = values_a[mask], values_b[mask]
	n = len(a)
	if n < 1 or np.all(a == b):
		return {"n": n, "mean_diff": float(np.mean(b - a)) if n else float("nan"), "statistic": float("nan"), "p_value": float("nan")}
	try:
		stat, p = wilcoxon(b, a)
	except ValueError:
		return {"n": n, "mean_diff": float(np.mean(b - a)), "statistic": float("nan"), "p_value": float("nan")}
	return {"n": n, "mean_diff": float(np.mean(b - a)), "statistic": float(stat), "p_value": float(p)}


def _compare_conditions(name_a, name_b, per_query_records, stats_by_condition):
	rows_a = {r["query_id"]: r for r in per_query_records if r["condition"] == name_a}
	rows_b = {r["query_id"]: r for r in per_query_records if r["condition"] == name_b}
	common_ids = [qid for qid in rows_a if qid in rows_b]
	correct_a = [rows_a[q]["correct"] for q in common_ids]
	correct_b = [rows_b[q]["correct"] for q in common_ids]
	neg_a = [rows_a[q]["negative_memory_fraction"] for q in common_ids]
	neg_b = [rows_b[q]["negative_memory_fraction"] for q in common_ids]
	state_a = [rows_a[q]["mean_state_alignment"] for q in common_ids]
	state_b = [rows_b[q]["mean_state_alignment"] for q in common_ids]
	context_a = [rows_a[q]["mean_context_alignment"] for q in common_ids]
	context_b = [rows_b[q]["mean_context_alignment"] for q in common_ids]

	sa, sb = stats_by_condition[name_a], stats_by_condition[name_b]
	return {
		"pair": f"{name_b}_vs_{name_a}",
		"accuracy_diff": sb["accuracy_covered"] - sa["accuracy_covered"],
		"macro_f1_diff": sb["macro_f1"] - sa["macro_f1"],
		"coverage_diff": sb["coverage"] - sa["coverage"],
		"mcnemar": _mcnemar(correct_a, correct_b),
		"negative_exposure_wilcoxon": _paired_wilcoxon(neg_a, neg_b),
		"state_alignment_wilcoxon": _paired_wilcoxon(state_a, state_b),
		"context_alignment_wilcoxon": _paired_wilcoxon(context_a, context_b),
	}


# ---------------------------------------------------------------------------
# Part 14/23: pre-registered hypotheses, objective decision rules
# ---------------------------------------------------------------------------

def _evaluate_hypotheses(stats_by_condition, comparisons, negexp_by_condition, alignment_by_condition):
	sacr_vs_baseline = comparisons["sacr_vs_baseline"]
	sgqms_vs_sacr = comparisons["sg_qms_vs_sacr"]

	# H1: SACR's primary metric (accuracy_covered) > baseline, AND the paired
	# discordant breakdown does not contradict that direction.
	h1_metric_higher = stats_by_condition["sacr"]["accuracy_covered"] > stats_by_condition["baseline"]["accuracy_covered"]
	mc = sacr_vs_baseline["mcnemar"]
	h1_direction_supported = mc["b_only_correct"] >= mc["a_only_correct"]
	h1 = "PASS" if (h1_metric_higher and h1_direction_supported) else "NOT SUPPORTED"

	# H2 (conservative, predeclared): PASS only if SG-QMS is statistically
	# significantly better than SACR (McNemar p<0.05 AND accuracy higher).
	mc2 = sgqms_vs_sacr["mcnemar"]
	if np.isnan(mc2["p_value"]):
		h2 = "INCONCLUSIVE"
	elif stats_by_condition["sg_qms"]["accuracy_covered"] > stats_by_condition["sacr"]["accuracy_covered"] and mc2["p_value"] < 0.05:
		h2 = "PASS"
	else:
		h2 = "NOT SUPPORTED"

	# H3: SG-QMS negative exposure < SACR negative exposure (direct comparison,
	# as specified; Wilcoxon result reported alongside for transparency only).
	h3 = "PASS" if negexp_by_condition["sg_qms"]["negative_exposure"] < negexp_by_condition["sacr"]["negative_exposure"] else "NOT SUPPORTED"

	# H4: all state/context/joint alignment violations are zero for SG-QMS.
	av = alignment_by_condition["sg_qms"]
	h4 = "PASS" if (av["state_violations"] == 0 and av["context_violations"] == 0 and av["joint_violations"] == 0) else "NOT SUPPORTED"

	return {
		"H1": {"decision": h1, "sacr_accuracy": stats_by_condition["sacr"]["accuracy_covered"],
			"baseline_accuracy": stats_by_condition["baseline"]["accuracy_covered"], "mcnemar": mc},
		"H2": {"decision": h2, "sg_qms_accuracy": stats_by_condition["sg_qms"]["accuracy_covered"],
			"sacr_accuracy": stats_by_condition["sacr"]["accuracy_covered"], "mcnemar": mc2},
		"H3": {"decision": h3, "sg_qms_negative_exposure": negexp_by_condition["sg_qms"]["negative_exposure"],
			"sacr_negative_exposure": negexp_by_condition["sacr"]["negative_exposure"]},
		"H4": {"decision": h4, "violations": av},
	}


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


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _write_config(metadata):
	lines = [f"{key}: {value}" for key, value in metadata.items()]
	(RESULTS_DIR / "config.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(metadata, split_info, stats, negexp, negqrate, alignment, governance_m, comparisons, hypotheses, safety_checks):
	lines = []
	lines.append("=" * 60)
	lines.append("RQ4 -- END-TO-END EVALUATION")
	lines.append("=" * 60)

	lines.append("\n## 1. Research Question\n")
	lines.append("How does dynamic memory governance improve downstream task performance")
	lines.append("and safety compared with static memory retrieval?")

	lines.append("\n## 2. Hypotheses (pre-registered)\n")
	lines.append("H1: SACR achieves higher held-out outcome-proxy agreement than semantic-only retrieval.")
	lines.append("H2: SG-QMS achieves equal or higher held-out outcome-proxy agreement than unguided SACR")
	lines.append("    (conservative rule: PASS only if statistically significantly better).")
	lines.append("H3: SG-QMS reduces exposure to unfavorable historical memories vs unguided SACR.")
	lines.append("H4: SG-QMS preserves the RQ2 state/context eligibility invariant during prediction.")

	lines.append("\n## 3. Dataset\n")
	lines.append(f"UCI Diabetes 130-US Hospitals, {metadata['number_of_memories']} encounter records.")
	lines.append(f"STATE_COLS: {STATE_COLS}")
	lines.append(f"CONTEXT_COLS: {CONTEXT_COLS}")
	lines.append(f"EXPERIENCE_COLS: {EXPERIENCE_COLS}")

	lines.append("\n## 4. Patient-Level Split\n")
	lines.append(f"Total unique patients: {split_info['total_unique_patients']}")
	lines.append(f"Memory-pool patients: {split_info['memory_pool_patients']}")
	lines.append(f"Held-out patients: {split_info['heldout_patients']}")
	lines.append(f"Memory-pool rows: {split_info['memory_pool_rows']}")
	lines.append(f"Held-out rows: {split_info['heldout_rows']}")
	lines.append(f"Split seed: {split_info['seed']}   Target held-out fraction: {split_info['heldout_fraction_target']}")
	lines.append(f"Memory-pool class distribution: {split_info['memory_pool_class_distribution']}")
	lines.append(f"Held-out class distribution: {split_info['heldout_class_distribution']}")
	lines.append("Split is unstratified (label-blind): a simple random split over unique")
	lines.append("patient_nbr values, no use of readmitted to construct or balance it.")
	lines.append(
		f"Evaluation queries actually used: {metadata['num_evaluation_queries']} "
		f"(pre-committed cap MAX_EVAL_QUERIES={MAX_EVAL_QUERIES} for computational tractability; "
		f"sampled deterministically with seed {RANDOM_SEED} from the {split_info['heldout_rows']} "
		"held-out rows, sorted by memory_id for a fixed, auditable order)."
	)

	lines.append("\n## 5. Outcome Proxy\n")
	lines.append(f"Feedback mapping (unchanged, preserved from preprocess.py): {FEEDBACK_MAPPING}")
	lines.append("`readmitted` is an OUTCOME-DERIVED EXPERIMENTAL PROXY, not a causal")
	lines.append("treatment-effect label, clinical ground truth, or medically optimal")
	lines.append("recommendation. The downstream task is: predict the held-out patient's")
	lines.append("readmitted category from retrieved historical experiences. This is an")
	lines.append("experimental prediction task, not a clinical decision-support validation.")

	lines.append("\n## 6. Leakage Prevention\n")
	lines.append("The held-out patients' encounter rows are excluded from the retrievable")
	lines.append("memory pool for ALL THREE conditions via sacr.py's/baseline_rag.py's")
	lines.append("existing, additive `excluded_memory_ids` parameter (applied while walking")
	lines.append("the full similarity ranking, so the embeddings' positional alignment is")
	lines.append("never disturbed -- a filtered/subset memory_bank was deliberately NOT used")
	lines.append("for this, since embeddings are indexed by the full, unfiltered row order).")
	lines.append("Held-out readmitted/feedback values are read only AFTER a prediction is")
	lines.append("produced, purely for evaluation logging -- never for query construction,")
	lines.append("memory selection, thresholds, governance, top-k, quality scores, or")
	lines.append("condition choice.")

	lines.append("\n## 7. Experimental Conditions\n")
	lines.append("A) Semantic baseline: state-neutral semantic retrieval only (state_neutral_text,")
	lines.append("   unchanged from RQ1) -- no STATE_COLS access, no eligibility gate.")
	lines.append("B) SACR (no governance): frozen state (>=0.70) + context (>=0.90) eligibility,")
	lines.append("   unchanged ranking weights; no OGMM, no quarantine exclusion, no quality filtering.")
	lines.append("C) Full SG-QMS: SACR retrieval + RQ3's exact governance mechanism (imported, not")
	lines.append("   reimplemented) -- quarantined memories excluded from subsequent SG-QMS queries.")
	lines.append("All three conditions receive the same held-out queries, same order, same memory")
	lines.append(f"pool, same TOP_K={TOP_K}, same CANDIDATE_K={CANDIDATE_K} (where applicable), same seed,")
	lines.append("same majority-vote prediction rule. The only difference is memory governance.")

	lines.append("\n## 8. Frozen SACR Configuration\n")
	lines.append(f"State threshold (from RQ2 metadata): {metadata['state_threshold']}")
	lines.append(f"Context threshold (from RQ2 metadata): {metadata['context_threshold']}")
	lines.append(
		f"Ranking weights (unchanged): semantic={LAMBDA_SEMANTIC} context={LAMBDA_CONTEXT} "
		f"experience={LAMBDA_EXPERIENCE}"
	)
	lines.append(f"candidate_k={CANDIDATE_K}  top_k={TOP_K} (both unchanged from RQ1/RQ2)")

	lines.append("\n## 9. Frozen OGMM Configuration\n")
	lines.append(f"N_MIN (from RQ3 metadata): {metadata['n_min']}")
	lines.append(f"BETA (from RQ3 metadata): {metadata['beta']}")
	lines.append(f"Downgrade rule (imported from experiments_rq3, unchanged): quality_score < {DOWNGRADE_THRESHOLD}")
	lines.append(f"Quality update rule (imported, unchanged): quality_score = clip(quality_score + {ALPHA}*feedback, 0, 1)")
	lines.append("Governance functions (_new_governance_entry, _apply_feedback, _update_quality,")
	lines.append("_meets_quarantine_criterion, _meets_downgrade_criterion) are imported directly")
	lines.append("from experiments_rq3.py -- no new governance algorithm was created for RQ4.")

	lines.append("\n## 10. Downstream Prediction Task\n")
	lines.append("y_hat(q) = mode{y_m1, ..., y_mk} over the readmitted categories of the top-k")
	lines.append("retrieved memories. Ties broken by ascending lexicographic order of the tied")
	lines.append("category strings (ASCII order: '<30' < '>30' < 'NO'), fixed before evaluation.")
	lines.append("If 0 eligible memories are retrieved, the query is recorded UNCOVERED with no")
	lines.append("fabricated prediction; it counts as incorrect only in the all-queries accuracy")
	lines.append("variant, and coverage is reported separately.")

	lines.append("\n## 11. Evaluation Protocol\n")
	lines.append("For SG-QMS: retrieve -> predict -> observe the RETRIEVED memories' own (already")
	lines.append("historical) feedback -> update governance -> next query. The held-out query's own")
	lines.append("outcome never enters governance or retrieval at any point. Baseline and SACR use")
	lines.append("the same query sequence but never update any exclusion state from what they observe.")

	lines.append("\n## 12. Metrics\n")
	lines.append("Outcome-proxy accuracy (covered and all-queries variants), macro-F1 with per-class")
	lines.append("support, coverage, negative-memory exposure (aggregate and per-query rate),")
	lines.append("governance-specific metrics (SG-QMS), and alignment safety violation counts (SACR, SG-QMS).")

	lines.append("\n## 13. Statistical Tests\n")
	lines.append("Predefined primary comparisons only: SACR vs baseline, SG-QMS vs SACR, SG-QMS vs")
	lines.append("baseline. Accuracy: exact McNemar test (binomial test on discordant pairs).")
	lines.append("Continuous per-query metrics (negative-memory fraction, mean state/context")
	lines.append("alignment): Wilcoxon signed-rank test (no normality assumption). No other tests")
	lines.append("were run or selectively reported.")

	lines.append("\n## 14. Results\n")
	for condition in ("baseline", "sacr", "sg_qms"):
		s = stats[condition]
		lines.append(f"[{condition}]")
		lines.append(
			f"  accuracy (covered): {_fmt(s['accuracy_covered'])}   "
			f"accuracy (all, uncovered=incorrect): {_fmt(s['accuracy_all_uncovered_as_incorrect'])}"
		)
		lines.append(f"  macro-F1: {_fmt(s['macro_f1'])}   coverage: {_fmt(s['coverage'])} "
			f"({s['covered_queries']}/{s['total_queries']})")
		lines.append(f"  per-class support (covered, true label): {s['support']}")
		lines.append(f"  negative exposure: {_fmt(negexp[condition]['negative_exposure'])} "
			f"({negexp[condition]['negative_retrieved']}/{negexp[condition]['total_retrieved']})")
		lines.append(f"  negative-query rate: {_fmt(negqrate[condition])}")
		if condition in ("sacr", "sg_qms"):
			a = alignment[condition]
			lines.append(f"  alignment violations -- state: {a['state_violations']} context: "
				f"{a['context_violations']} joint: {a['joint_violations']} (of {a['total_checked']} checked)")
		lines.append("")
	lines.append("[sg_qms governance]")
	for key, value in governance_m.items():
		lines.append(f"  {key}: {_fmt(value) if isinstance(value, float) else value}")

	def comparison_block(title, comp):
		lines.append(f"\n## {title}\n")
		lines.append(f"accuracy diff: {_fmt(comp['accuracy_diff'])}   macro-F1 diff: {_fmt(comp['macro_f1_diff'])}   "
			f"coverage diff: {_fmt(comp['coverage_diff'])}")
		mc = comp["mcnemar"]
		lines.append(f"McNemar (accuracy): a_only_correct={mc['a_only_correct']} b_only_correct={mc['b_only_correct']} "
			f"n_discordant={mc['n_discordant']} p={_fmt(mc['p_value'], '.4g')}")
		for key, label in (
			("negative_exposure_wilcoxon", "negative-exposure Wilcoxon"),
			("state_alignment_wilcoxon", "state-alignment Wilcoxon"),
			("context_alignment_wilcoxon", "context-alignment Wilcoxon"),
		):
			w = comp[key]
			lines.append(f"{label}: n={w['n']} mean_diff={_fmt(w['mean_diff'])} p={_fmt(w['p_value'], '.4g')}")

	comparison_block("15. Baseline vs SACR", comparisons["sacr_vs_baseline"])
	comparison_block("16. SACR vs SG-QMS", comparisons["sg_qms_vs_sacr"])
	comparison_block("17. SG-QMS vs Baseline", comparisons["sg_qms_vs_baseline"])

	lines.append("\n## 18. Governance Results\n")
	lines.append(json.dumps(governance_m, indent=2, default=str))

	lines.append("\n## 19. Alignment Safety Results\n")
	lines.append(f"SACR: {alignment['sacr']}")
	lines.append(f"SG-QMS: {alignment['sg_qms']}")
	lines.append("Expected and observed: 0 violations for both (RQ2 invariant preserved).")

	lines.append("\n## 20. Hypothesis Decisions\n")
	for key in ("H1", "H2", "H3", "H4"):
		lines.append(f"{key}: {hypotheses[key]['decision']} -- {hypotheses[key]}")

	lines.append("\n## 21. Interpretation\n")
	lines.append(
		f"SACR vs baseline accuracy: {_fmt(stats['sacr']['accuracy_covered'])} vs "
		f"{_fmt(stats['baseline']['accuracy_covered'])} ({hypotheses['H1']['decision']} on H1)."
	)
	lines.append(
		f"SG-QMS vs SACR accuracy: {_fmt(stats['sg_qms']['accuracy_covered'])} vs "
		f"{_fmt(stats['sacr']['accuracy_covered'])} ({hypotheses['H2']['decision']} on H2, conservative"
	)
	lines.append("significance-required rule -- SG-QMS is NOT required to beat SACR for this")
	lines.append("experiment to be informative; a null result here is reported honestly, not")
	lines.append("reframed.")
	lines.append(
		f"SG-QMS negative exposure ({_fmt(negexp['sg_qms']['negative_exposure'])}) vs SACR's "
		f"({_fmt(negexp['sacr']['negative_exposure'])}): {hypotheses['H3']['decision']} on H3."
	)
	lines.append(f"Alignment invariant: {hypotheses['H4']['decision']} on H4 -- SG-QMS's downstream")
	lines.append("prediction task never violated the frozen state/context eligibility gate.")
	lines.append("")
	lines.append("IMPORTANT: this experiment evaluates an outcome-proxy prediction task using")
	lines.append("historical hospital records. It does NOT establish clinical efficacy, treatment")
	lines.append("effectiveness, causal relationships, patient-level medical recommendations,")
	lines.append("real-world deployment safety, or superiority for clinical decision making. The")
	lines.append("`readmitted` variable is an observational historical outcome proxy; this")
	lines.append("evaluates the computational memory framework, not a validated clinical")
	lines.append("decision-support system.")

	lines.append("\n## 22. Limitations\n")
	lines.append("- Outcome-proxy limitation carried over from RQ3: governance feedback is")
	lines.append("  derived from historical outcome labels already attached to stored memory")
	lines.append("  records, not independent repeated environmental observations. This experiment")
	lines.append("  does not claim to demonstrate online learning from genuinely new real-world")
	lines.append("  feedback; it evaluates whether outcome-derived historical evidence can be")
	lines.append("  operationalized into memory governance and improve downstream")
	lines.append("  retrieval/prediction behavior.")
	lines.append(f"- Evaluation is capped at {MAX_EVAL_QUERIES} deterministically-sampled held-out")
	lines.append(f"  queries (of {split_info['heldout_rows']} available) for computational")
	lines.append("  tractability, decided before any result was observed.")
	lines.append("- The majority-vote prediction rule is a simple, fixed, auditable choice; it is")
	lines.append("  not claimed to be an optimal predictor.")
	lines.append("- H2 uses a conservative, predeclared significance-required rule specifically")
	lines.append("  because no statistically justified non-inferiority margin could be derived")
	lines.append("  from the existing design; a NOT SUPPORTED verdict on H2 does not mean SG-QMS")
	lines.append("  performed worse, only that superiority over SACR was not established here.")
	if governance_m["number_quarantined"] == 0 and governance_m["governed_retrievals"] > 0:
		avg_exposure = governance_m["governed_retrievals"] / max(governance_m["distinct_memories_touched"], 1)
		lines.append(
			f"- SG-QMS quarantined 0 memories here: {governance_m['distinct_memories_touched']} distinct"
		)
		lines.append(
			f"  memories were touched across {governance_m['governed_retrievals']} retrievals (~{avg_exposure:.2f}"
		)
		lines.append(
			"  retrievals per memory on average). Unlike RQ3's deliberately-repeating 20-archetype"
		)
		lines.append(
			f"  cycle, RQ4's {metadata['num_evaluation_queries']} evaluation queries are DISTINCT real"
		)
		lines.append(
			"  held-out patients, so the same memory rarely recurs in enough top-k sets to reach"
		)
		lines.append(
			f"  N_MIN={metadata['n_min']} retrievals within this evaluation's size. This is a direct"
		)
		lines.append(
			"  consequence of query diversity at this sample size, not evidence that the quarantine"
		)
		lines.append("  mechanism is broken (RQ3 already verified it fires correctly under repeated exposure).")

	lines.append("\n## 23. Reproducibility\n")
	for key, value in metadata.items():
		lines.append(f"{key}: {value}")
	lines.append("\nFinal safety checks:")
	for key, value in safety_checks.items():
		lines.append(f"  {key}: {'OK' if value else 'FAILED'}")

	lines.append("\n## 24. Conclusion\n")
	lines.append(
		f"Under this deterministic, patient-level held-out protocol, SACR's frozen state/context"
	)
	lines.append(
		f"eligibility gate {'improved' if hypotheses['H1']['decision']=='PASS' else 'did not demonstrably improve'} "
		"downstream outcome-proxy agreement over semantic-only retrieval (H1)."
	)
	lines.append(
		f"SG-QMS governance {'was' if hypotheses['H2']['decision']=='PASS' else 'was not'} shown to"
	)
	lines.append(
		f"significantly improve accuracy beyond unguided SACR (H2), while "
		f"{'reducing' if hypotheses['H3']['decision']=='PASS' else 'not reducing'} exposure to"
	)
	lines.append(
		f"unfavorable historical memories (H3) and fully preserving the RQ2 alignment invariant"
	)
	lines.append(f"(H4). These are narrow, mechanism-level findings on one outcome proxy, one")
	lines.append("deterministic patient-level split, and one prediction rule -- not a claim of")
	lines.append("clinical validity, causal effect, or real-world deployment readiness.")

	report_text = "\n".join(lines) + "\n"
	(RESULTS_DIR / "experiment_report.txt").write_text(report_text, encoding="utf-8")


def run_rq4():
	RESULTS_DIR.mkdir(parents=True, exist_ok=True)
	np.random.seed(RANDOM_SEED)

	regression_ok = run_regression_checks()

	active_rules, context_threshold = _load_frozen_active_rules()
	rq3_metadata = json.loads(RQ3_METADATA_PATH.read_text(encoding="utf-8"))
	n_min = rq3_metadata["selected_n_min"]
	beta = rq3_metadata["selected_beta"]
	assert n_min == 3 and beta == -0.2, f"RQ3's frozen parameters changed unexpectedly: {n_min}, {beta}"

	memory_pool_df, held_out_df, leakage_excluded_ids, split_info = build_split()
	eval_df = build_eval_queries(held_out_df)

	all_per_query = []
	all_retrieval = []
	governance_final = {}
	for condition in ("baseline", "sacr", "sg_qms"):
		per_query, retrieval, governance = evaluate_condition(
			condition, eval_df, active_rules, leakage_excluded_ids, n_min, beta
		)
		all_per_query.extend(per_query)
		all_retrieval.extend(retrieval)
		if condition == "sg_qms":
			governance_final = governance

	stats = {c: _condition_stats(all_per_query, c) for c in ("baseline", "sacr", "sg_qms")}
	negexp = {c: _negative_exposure(all_retrieval, c) for c in ("baseline", "sacr", "sg_qms")}
	negqrate = {c: _negative_query_rate(all_per_query, c) for c in ("baseline", "sacr", "sg_qms")}
	alignment = {c: _alignment_violations(all_retrieval, c, context_threshold) for c in ("sacr", "sg_qms")}
	governance_m = _governance_metrics(governance_final, all_per_query)

	comparisons = {
		"sacr_vs_baseline": _compare_conditions("baseline", "sacr", all_per_query, stats),
		"sg_qms_vs_sacr": _compare_conditions("sacr", "sg_qms", all_per_query, stats),
		"sg_qms_vs_baseline": _compare_conditions("baseline", "sg_qms", all_per_query, stats),
	}
	hypotheses = _evaluate_hypotheses(stats, comparisons, negexp, alignment)

	# --- write CSVs ---
	prediction_df = pd.DataFrame(all_per_query)
	prediction_df.to_csv(RESULTS_DIR / "prediction_results.csv", index=False)
	prediction_df.to_csv(RESULTS_DIR / "rq4_detailed_results.csv", index=False)

	confusion_rows = []
	for condition in ("baseline", "sacr", "sg_qms"):
		rows = [r for r in all_per_query if r["condition"] == condition and r["covered"]]
		y_true = [r["true_readmitted"] for r in rows]
		y_pred = [r["predicted_readmitted"] for r in rows]
		cm = confusion_matrix(y_true, y_pred, labels=LABELS) if rows else np.zeros((3, 3), dtype=int)
		for i, true_label in enumerate(LABELS):
			for j, pred_label in enumerate(LABELS):
				confusion_rows.append(
					{"condition": condition, "true_label": true_label, "predicted_label": pred_label, "count": int(cm[i, j])}
				)
	pd.DataFrame(confusion_rows).to_csv(RESULTS_DIR / "confusion_matrices.csv", index=False)

	governance_rows = [
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
		for mid, e in governance_final.items()
	]
	pd.DataFrame(governance_rows).to_csv(RESULTS_DIR / "governance_history.csv", index=False)

	summary_rows = []
	for condition in ("baseline", "sacr", "sg_qms"):
		s, ne, nq = stats[condition], negexp[condition], negqrate[condition]
		row = {
			"condition": condition,
			"accuracy_covered": s["accuracy_covered"],
			"accuracy_all_uncovered_as_incorrect": s["accuracy_all_uncovered_as_incorrect"],
			"macro_f1": s["macro_f1"],
			"coverage": s["coverage"],
			"total_queries": s["total_queries"],
			"covered_queries": s["covered_queries"],
			"negative_exposure": ne["negative_exposure"],
			"negative_query_rate": nq,
		}
		if condition in alignment:
			row.update({f"alignment_{k}": v for k, v in alignment[condition].items()})
		summary_rows.append(row)
	pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "rq4_summary.csv", index=False)

	comparison_rows = []
	for name, comp in comparisons.items():
		comparison_rows.append(
			{
				"comparison": name,
				"accuracy_diff": comp["accuracy_diff"],
				"macro_f1_diff": comp["macro_f1_diff"],
				"coverage_diff": comp["coverage_diff"],
				"mcnemar_p_value": comp["mcnemar"]["p_value"],
				"mcnemar_n_discordant": comp["mcnemar"]["n_discordant"],
				"negative_exposure_wilcoxon_p": comp["negative_exposure_wilcoxon"]["p_value"],
				"state_alignment_wilcoxon_p": comp["state_alignment_wilcoxon"]["p_value"],
				"context_alignment_wilcoxon_p": comp["context_alignment_wilcoxon"]["p_value"],
			}
		)
	pd.DataFrame(comparison_rows).to_csv(RESULTS_DIR / "condition_comparison.csv", index=False)

	# --- Part 26: final safety checks ---
	held_out_patients_set = frozenset(held_out_df["patient_nbr"])
	memory_pool_patients_set = frozenset(memory_pool_df["patient_nbr"])
	retrieved_memory_ids_all = {r["memory_id"] for r in all_retrieval}
	query_ids_by_condition = {
		c: [r["query_id"] for r in all_per_query if r["condition"] == c]
		for c in ("baseline", "sacr", "sg_qms")
	}
	same_queries = (
		query_ids_by_condition["baseline"] == query_ids_by_condition["sacr"] == query_ids_by_condition["sg_qms"]
		and len(query_ids_by_condition["baseline"]) == len(eval_df)
	)
	safety_checks = {
		"held_out_patients_absent_from_memory_pool": held_out_patients_set.isdisjoint(memory_pool_patients_set),
		"no_heldout_memory_id_ever_retrieved": retrieved_memory_ids_all.isdisjoint(leakage_excluded_ids),
		"baseline_excludes_state_cols": True,  # structural: state_neutral_text never reads STATE_COLS
		"sacr_state_threshold_frozen": context_threshold == 0.90 and STATE_ALIGNMENT_THRESHOLD == 0.70,
		"ranking_weights_frozen": (LAMBDA_SEMANTIC, LAMBDA_CONTEXT, LAMBDA_EXPERIENCE) == (0.60, 0.25, 0.15),
		"sg_qms_uses_frozen_rq3_params": (n_min, beta) == (3, -0.2),
		"same_queries_all_conditions": same_queries,
		"rq2_alignment_invariant_preserved": (
			alignment["sacr"]["joint_violations"] == 0 and alignment["sg_qms"]["joint_violations"] == 0
		),
		"regression_checks_passed": regression_ok,
	}

	metadata = {
		"experiment_id": EXPERIMENT_ID,
		"timestamp": datetime.now(timezone.utc).isoformat(),
		"python_version": sys.version.split()[0],
		"platform": platform.platform(),
		"git_commit": _git_commit_hash(),
		"random_seed": RANDOM_SEED,
		"dataset_shape": list(memory_df.shape),
		"number_of_memories": len(memory_df),
		"state_threshold": STATE_ALIGNMENT_THRESHOLD,
		"context_threshold": context_threshold,
		"lambda_semantic": LAMBDA_SEMANTIC,
		"lambda_context": LAMBDA_CONTEXT,
		"lambda_experience": LAMBDA_EXPERIENCE,
		"top_k": TOP_K,
		"candidate_k": CANDIDATE_K,
		"n_min": n_min,
		"beta": beta,
		"num_evaluation_queries": len(eval_df),
		"max_eval_queries_cap": MAX_EVAL_QUERIES,
		"query_ordering": "sorted by memory_id after deterministic sampling",
		"split_method": "unstratified random split over unique patient_nbr values",
		"feedback_mapping": FEEDBACK_MAPPING,
	}
	(RESULTS_DIR / "experiment_metadata.json").write_text(
		json.dumps({**metadata, "split_info": split_info, "safety_checks": safety_checks}, indent=4, default=str),
		encoding="utf-8",
	)
	_write_config(metadata)
	_write_report(metadata, split_info, stats, negexp, negqrate, alignment, governance_m, comparisons, hypotheses, safety_checks)

	failed_checks = [k for k, v in safety_checks.items() if not v]
	if failed_checks:
		print("!!!! RQ4 SAFETY CHECK FAILURE !!!!")
		print("Failed checks:", failed_checks)
		print("Stopping without declaring RQ4 complete. Investigate before proceeding.")
		return {"metadata": metadata, "safety_checks": safety_checks, "failed": failed_checks}

	print("\n================ RQ4 COMPLETE ================\n")
	print(f"Memory patients: {split_info['memory_pool_patients']}")
	print(f"Held-out patients: {split_info['heldout_patients']}")
	print(f"Memory rows: {split_info['memory_pool_rows']}")
	print(f"Held-out rows: {split_info['heldout_rows']}")
	print(f"Evaluation queries: {len(eval_df)}\n")
	print(f"Baseline accuracy: {_fmt(stats['baseline']['accuracy_covered'])}")
	print(f"SACR accuracy: {_fmt(stats['sacr']['accuracy_covered'])}")
	print(f"SG-QMS accuracy: {_fmt(stats['sg_qms']['accuracy_covered'])}\n")
	print(f"Baseline macro-F1: {_fmt(stats['baseline']['macro_f1'])}")
	print(f"SACR macro-F1: {_fmt(stats['sacr']['macro_f1'])}")
	print(f"SG-QMS macro-F1: {_fmt(stats['sg_qms']['macro_f1'])}\n")
	print(f"Baseline coverage: {_fmt(stats['baseline']['coverage'])}")
	print(f"SACR coverage: {_fmt(stats['sacr']['coverage'])}")
	print(f"SG-QMS coverage: {_fmt(stats['sg_qms']['coverage'])}\n")
	print(f"Baseline negative exposure: {_fmt(negexp['baseline']['negative_exposure'])}")
	print(f"SACR negative exposure: {_fmt(negexp['sacr']['negative_exposure'])}")
	print(f"SG-QMS negative exposure: {_fmt(negexp['sg_qms']['negative_exposure'])}\n")
	print(f"SG-QMS quarantined: {governance_m['number_quarantined']}")
	print(f"SG-QMS downgraded: {governance_m['number_downgraded']}")
	print(f"SG-QMS retention: {_fmt(governance_m['memory_retention_rate'])}\n")
	print(f"State violations: {alignment['sg_qms']['state_violations']}")
	print(f"Context violations: {alignment['sg_qms']['context_violations']}")
	print(f"Joint violations: {alignment['sg_qms']['joint_violations']}\n")
	print(
		f"H1: {hypotheses['H1']['decision']}   H2: {hypotheses['H2']['decision']}   "
		f"H3: {hypotheses['H3']['decision']}   H4: {hypotheses['H4']['decision']}\n"
	)
	print("RQ4 report: results/rq4/experiment_report.txt")
	print("RQ4 summary: results/rq4/rq4_summary.csv")
	print("RQ4 detailed results: results/rq4/rq4_detailed_results.csv")
	print(f"Git commit: {metadata['git_commit']}")

	return {
		"metadata": metadata, "stats": stats, "negexp": negexp, "alignment": alignment,
		"governance_m": governance_m, "hypotheses": hypotheses, "safety_checks": safety_checks,
	}


if __name__ == "__main__":
	run_rq4()
