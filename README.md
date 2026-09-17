# SG-QMS: State-Gated, Quality-Managed Memory Retrieval

A memory-augmented retrieval system for outcome-proxy prediction on the **UCI Diabetes 130-US Hospitals** dataset (101,766 encounters). Four research questions test, in order: whether *smarter retrieval* beats semantic-only retrieval, whether that retrieval can be made *provably safe*, whether the memory bank can *govern itself* based on outcome feedback, and whether all of it together improves *downstream prediction*.

> `readmitted` is used throughout as an **outcome-derived experimental proxy**, not a causal or clinical ground truth. No claim here is a claim of clinical effectiveness or deployment readiness.

## Dataset

| | |
|---|---|
| Raw size | 101,766 encounters × 48 columns |
| Features used | 17: 7 state (age, gender, race, prior visit counts) + 7 context (admission, labs, meds) + 3 experience (insulin, med change, diabetes med) |
| Missing values | only `race`, `medical_specialty` → filled "Unknown" |
| Outcome → feedback | `NO`→+1.0, `>30`→0.0, `<30`→−1.0 |
| Patient split (seed 42) | 71,518 unique patients → 57,214 (memory pool, 81,394 rows) / 14,304 held out (20,372 rows, never in the pool) |

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
