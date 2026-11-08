# Submission #2 — targeted rerun analysis (2026-11-08, ~13:30 IST)

Analysed `data/answers.jsonl` (600 rows, Qwen3-1.7B) joined to `data/Test.csv`.
First submission y7yUg5eh scored public 0.221709104, rank 9 (leader 0.36913).

## Correction to the earlier summary
The "76 abstentions" figure is a substring count, not full abstentions:
- Only **1** answer is exactly the insufficient-information sentence, and only
  **7** start with it. The other 69 mention "don't have enough information"
  inside a longer answer (hedging / partial answer).
- So the lever is **not** 76 empty answers. It is 109 rows with no model
  evidence, of which 108 still gave an answer — those got fallback evidence
  (first excerpt source) in `build_submission.py`, which likely scores low on
  evidence support. Metric is 0.8*Correctness + 0.2*(Correctness*Evidence):
  a correct answer with no supporting evidence caps at 0.8.

## Failure counts (verified 2026-11-08)
| group | n | by domain |
|---|---|---|
| No model evidence | 109 / 600 (18.2%) | transport 33/148 (22.3%), urban_greenery 37/191 (19.4%), buildings_energy 39/261 (14.9%) |
| Hit 512-token cap (`NUM_PREDICT=512`) | 56 | buildings_energy 29, transport 14, urban_greenery 13 |
| Mention "not enough information" | 76 | transport 24, greenery 23, buildings 29 |
| Exact abstention | 1 | urban_greenery |
| Raw evidence SIDs not in that Q's listed `Source_IDs` | 114 of 2,426 lines (4.7%) — already dropped by `build_submission.clean_evidence`, which dedupes to max 3 SIDs and drops invalid ones, so the submitted file does not carry them |

Overlap: capped & no-evidence 9; abstain-mention & no-evidence 65.
**Union targeted set: 156 IDs** (buildings 62, greenery 48, transport 46).
Full list: `research/submission2_target_ids.txt` (156 lines, sorted).

## Recommendation for submission #2 (window opens Mon 9 Nov, 1/day)
Do NOT rerun all 600. Rerun only the 156 targeted IDs (~24 min at the
first run's ~9 s/Q, vs ~91 min full) with two changes:
1. `NUM_PREDICT` 512 -> 1024 for the 56 capped IDs only (limit is 12,000/Q,
   so this stays legal; the cap truncated evidence blocks mid-line).
2. For the 109 no-evidence IDs, a second pass (allowed: max 15 calls/Q,
   currently max 2 used) with the prompt restated: cite the exact source
   lines that support each number, one `SRC_x | locator | claim` line per
   claim, only from the listed sources.
Keep the other 444 answers untouched; rebuild with `build_submission.py`
and diff the 156 rows before submitting.

## What `colab_run.py` needs first (small patch, not yet applied)
- An `--ids-file` / `--only-ids` filter: currently only `--limit N` /
  `--smoke` exist, which take the first N questions — a targeted rerun
  cannot be expressed without it. Resume logic (skip IDs already in
  answers.jsonl) already exists, so the patch is: load the ID set,
  filter `records` to it, and pass a per-question `num_predict`.
- Do not change model, temperature, seed, or `num_ctx` (32768).

## Still undecided (needs the user)
- Teammate: solo is disqualified under IEEE Rules §8 (min 2, max 4);
  late teaming deducts solo submissions from the team's 20. Candidates
  from the user's own context: Gaurav Jadhav or the Zindi-notes friend.
  This decision is independent of the rerun — the rerun can be prepared
  today and submitted tomorrow either way, but teaming should happen
  before submission #2 is spent if possible.
- Funding Portal narrative application (exactly one use case: Pristina
  transit / Rivne energy / Skopje green) also due 13 Dec — not started.
