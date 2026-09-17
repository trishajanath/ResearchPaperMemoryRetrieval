# SG-QMS: State-Gated, Quality-Managed Memory Retrieval

SG-QMS is a memory-augmented retrieval framework designed to make retrieved past experiences **state-compatible, contextually relevant, and quality-aware**. It is evaluated on the **UCI Diabetes 130-US Hospitals (1999–2008)** dataset (101,766 hospital encounters) through four research questions, moving progressively from **retrieval → safety → memory governance → downstream evaluation**:

1. **RQ1 — State-aware retrieval:** Does state-aware retrieval reduce the retrieval of experiences incompatible with the current patient state?
2. **RQ2 — Retrieval safety:** Can state and contextual compatibility be enforced as explicit mathematical eligibility constraints, not just a statistical tendency?
3. **RQ3 — Memory governance:** Can the memory bank automatically downgrade or quarantine repeatedly negative experiences using outcome feedback?
4. **RQ4 — Downstream effect:** Does dynamic memory governance affect downstream prediction performance and negative-memory exposure?

> **Experimental proxy:** `readmitted` is converted into an outcome-derived feedback signal purely for experimentation. It is **not** causal evidence, clinical ground truth, or evidence of treatment effectiveness. Nothing here establishes clinical effectiveness or deployment readiness.

## Dataset & Preprocessing

**Raw encounters → missing-value handling → feature selection → state/context/experience construction → outcome-feedback construction → patient-level split → memory pool + held-out set**

| | |
|---|---|
| Raw size | 101,766 encounters × 48 columns |
| Missing values | only `race`, `medical_specialty` → filled `"Unknown"` |
| Features used | 17 of 48 (see table below); columns such as `discharge_disposition_id`, `payer_code`, `weight`, and the diagnosis/medication codes are left out — several of these sit close to the discharge/outcome pathway itself, so keeping them out of the retrieval features avoids a plausible leakage channel |
| Patient split (seed 42) | 71,518 unique patients → 57,214 (memory pool, 81,394 rows) / 14,304 held out (20,372 rows) |

The split is done **at the patient level**: every encounter belonging to a held-out patient is excluded from the memory pool, so the retrieval system can never see a held-out patient's own history during evaluation.

### Feature roles

| Group | Features | Used for |
|---|---|---|
| **State** | `age`, `gender`, `race`, `number_outpatient`, `number_emergency`, `number_inpatient`, `number_diagnoses` | *State alignment* — is this past patient compatible with the current one? (RQ1/RQ2 eligibility gate) |
| **Context** | `admission_type_id`, `admission_source_id`, `time_in_hospital`, `medical_specialty`, `num_lab_procedures`, `num_procedures`, `num_medications` | *Context alignment* — is the care setting comparable? (RQ2 eligibility gate, retrieval ranking) |
| **Experience** | `insulin`, `change`, `diabetesMed` | *Experience match* — what treatment action did this memory involve? (retrieval ranking) |
| **Outcome** | `readmitted` (`NO` / `>30` / `<30`) | Downstream prediction target, and mapped to **feedback** for governance: `NO`→+1.0, `>30`→0.0, `<30`→−1.0 |

```
Encounter → [STATE, CONTEXT, EXPERIENCE] → SACR → Retrieved memories
                                                        │
                                              outcome-derived feedback
                                                        ▼
                                                      OGMM
                                                 ┌──────┴──────┐
                                              Retain     Downgrade / Quarantine
```

State and context govern **retrieval safety and alignment** (is this memory even eligible?); experience describes **what happened** in an eligible memory. This separation lets the system check *relevance* before it uses *history* — instead of leaning on text similarity alone.

## Metrics & Formulas

Every number in the tables below comes from one of these definitions — no metric here is a black box.

**Alignment scores** (used for eligibility gates and ranking): for a query `q` and memory `m`,
```
similarity(q_i, m_i) = 1 if q_i == m_i                        (categorical field)
                      = 1 − |q_i − m_i| / range(field)          (numeric field)
state_alignment(q, m)   = mean(similarity over the 7 STATE fields)
context_alignment(q, m) = mean(similarity over the 7 CONTEXT fields)
```
**Misalignment rate** = share of retrieved memories with `state_alignment < 0.70` (the frozen threshold).
**Eligibility (RQ2)** = a memory is even retrievable only if `state_alignment ≥ 0.70` **and** `context_alignment ≥ 0.90`.
**Coverage** = share of queries that had ≥1 eligible memory retrieved at all. **Accuracy (covered)** = accuracy computed only over those covered queries.

**Negative exposure** = `(# retrieved memories with feedback < 0) / (total memories retrieved)` — measured over every retrieval, across all queries.
**Negative-query rate** = `(# queries with ≥1 negative memory in their retrieved set) / (# covered queries)` — measured per query instead of per retrieval.

**Quality score** (OGMM, per memory, starts at 0.5): updated every time the memory is retrieved,
```
quality_score ← clip(quality_score + ALPHA × feedback, 0, 1),   ALPHA = 0.1
```
**Downgrade**: a memory is downgraded the first time `quality_score < 0.4`.
**Average utility** (per memory) = `(sum of feedback over every retrieval of that memory) / (retrieval_count)`.
**Quarantine**: a memory is quarantined — permanently excluded from future retrieval — the first time both hold:
```
retrieval_count ≥ N_MIN (3)   AND   average_utility ≤ BETA (−0.2)
```
**Retention** = `(# memories touched by governance − # quarantined) / (# memories touched)` — the share of touched memories still fully available.
**Quarantine effectiveness** = `|quarantined ∩ oracle-eligible| / |oracle-eligible|` — of the memories an unconstrained (no-governance) run would also have flagged, what fraction did we actually catch?
**False-quarantine rate** = `|quarantined − oracle-eligible| / |quarantined|` — of what we quarantined, what fraction the oracle run would *not* have flagged?

## Research Questions

### RQ1 — Does state-aligned retrieval (SACR) beat semantic-only retrieval?

SACR adds a hard eligibility filter (mean state similarity ≥ 0.70) before ranking; baseline is text-only semantic similarity.

| | Baseline | SACR |
|---|---|---|
| Mean state alignment (normal bank) | 0.78 | **0.85** |
| Mean state alignment (shuffled bank) | 0.70 | **0.81** |
| Misalignment rate (normal / shuffled) | 38.5% / 61% | **0% / 0%** |

Both improvements significant, p < 10⁻⁹ (paired t-test & Wilcoxon, n=40).

**Result: SACR retrieves better-matched memories, with zero misalignment in both conditions.**

### RQ2 — Is state+context eligibility a guarantee, not just a tendency?

Added a joint gate (state ≥ 0.70 **and** context ≥ 0.90) checked before ranking.

| Checked retrievals | State violations | Context violations | Joint violations |
|---|---|---|---|
| 200 | **0** | **0** | **0** |

Context threshold (0.90) was fixed in advance as the largest value keeping ≥5 eligible candidates for every query — not tuned to any downstream result.

**Result: the eligibility gate is a structural invariant — it cannot be violated by construction, and testing confirmed it.**

### RQ3 — Can outcome-gated governance (OGMM) identify and quarantine low-utility memories?

Compared static (no governance), history-deletion, and OGMM over 300 rounds (20 recurring scenarios × 15 cycles).

| Policy | Negative retrieval rate | Quarantined | Retention |
|---|---|---|---|
| Static | 13.0% | 0 | 100% |
| History-deletion | 3.4% | 17 | 85.3% |
| OGMM | 3.4% | 17 | 85.3% |

**Result: OGMM matches deletion's exposure reduction, but preserves an auditable `quarantine_reason` for every flagged memory instead of silently erasing it.**

### RQ4 — Does end-to-end governance improve downstream prediction?

Two evaluations, same frozen split/mechanism:

**A. Naturalistic (N=3,000 real held-out patients)**

| Condition | Accuracy | Macro-F1 | Quarantined |
|---|---|---|---|
| Baseline | 0.469 | 0.349 | — |
| SACR | 0.482 | 0.363 | — |
| SG-QMS | 0.482 | 0.363 | 17 |

SACR and SG-QMS are numerically identical. McNemar test between them is undefined (0 discordant pairs).

**B. Recurrence-controlled (50 fixed queries × 20 cycles, to force governance exposure)**

- Governance activated by cycle 3; retrieval sets diverged from SACR (overlap 1.0 → 0.835)
- Negative exposure fell from 11.1% (SACR) to 0% (SG-QMS) by cycle 7 (Wilcoxon p = 0.0001)
- **Correctness: 34 queries flipped wrong→correct, 34 flipped correct→wrong — net zero.**

**Result: governance is mechanically active and safe (0 eligibility violations throughout), and measurably reduces exposure to bad memories — but has not been shown to improve prediction accuracy, in either test.**

## Overall Inference

| Claim | Status |
|---|---|
| SACR improves retrieval state-alignment over semantic-only search | ✅ Supported |
| State+context eligibility is a structural safety guarantee | ✅ Supported |
| OGMM correctly identifies and quarantines low-utility memories, auditably | ✅ Supported |
| Governance mechanism activates safely under sufficient exposure | ✅ Supported |
| SG-QMS improves downstream prediction accuracy over SACR | ❌ Not supported (null result in both RQ4 tests) |

The null result on accuracy is reported as-is, not reframed — the system's retrieval and governance mechanisms work exactly as designed, but that has not (yet) translated into a measured accuracy gain.

## Repository Structure

```
preprocess.py, baseline_rag.py, sacr.py, columns.py   # core pipeline + SACR retrieval
experiments.py            # RQ1
experiments_rq2.py        # RQ2
experiments_rq3.py        # RQ3 (OGMM)
experiments_rq4.py        # RQ4 original (300-query)
experiments_rq4_sequential.py  # sample-size sensitivity sweep (300 → 20,372)
experiments_rq4_final.py  # RQ4 primary result (N=3,000)
experiments_rq4b.py       # RQ4-B recurrence-controlled experiment
results/<name>/           # every experiment's report, CSVs, and metadata (one folder per run)
```

## Running It

```bash
source myenv/bin/activate   # or your own venv with pandas, numpy, scipy, scikit-learn, sentence-transformers, torch, ucimlrepo

python3 experiments.py                    # RQ1 (first run downloads the dataset + builds embeddings)
python3 experiments_rq2.py                # RQ2 — needs RQ1's results
python3 experiments_rq3.py                # RQ3 — needs RQ2's threshold
python3 experiments_rq4.py                # RQ4 original — needs RQ3's parameters
python3 experiments_rq4_sequential.py     # sample-size sweep (~70 min)
python3 experiments_rq4_final.py          # RQ4 primary, N=3,000 (~10 min)
python3 experiments_rq4b.py               # RQ4-B recurrence-controlled (~3 min)
```

Run in this order — each script reads back frozen parameters from the previous one's `results/` output. `data/*.pkl` and `data/*.npy` (raw dataset cache, embeddings) are gitignored and regenerate automatically on first run.

## Limitations

- `readmitted`/feedback is a historical outcome proxy, not causal ground truth, in every RQ.
- RQ1–RQ3 use small, fixed, deterministic query sets (20–40 instances / 300 rounds).
- RQ4-B's repeated exposure is deterministic given each memory's fixed historical feedback — it demonstrates mechanism behavior under forced recurrence, not learning from new evidence.
- RQ3's and RQ4's false-quarantine rates are measured against an unconstrained oracle (static/SACR) whose own trajectory can diverge from the governed condition's for reasons beyond the quarantine decision itself.
