# Submission #2 — diagnosis and build notes (2026-11-08)

Supersedes research/06-submission1-diagnosis.md where it conflicts.
Submission #1: ID y7yUg5eh, score **0.221709104, rank 9**; leader
alexchilt 0.369126157; gap **0.1474**. 19/20 submissions remain.

## Diagnosis (measured locally from data/answers.jsonl, data/contexts.jsonl, data/contexts_log.json, Test.csv)

### Failure-mode counts (600 questions)

| Failure mode | Count | By domain (BE / UG / TR) |
|---|---|---|
| Mentions "don't have enough information" | 76 | 29 / 23 / 24 |
| — of which start with the abstain sentence | 7 | — |
| — of which the exact abstain sentence only | 1 | — |
| Answers capped at 512 generated tokens | 56 | 29 / 13 / 14 |
| Rows with no evidence lines | 109 | 39 / 37 / 33 |
| Evidence lines citing sources never listed for that Q | 114 lines | — |
| Union (mention \| capped \| no-evidence) | 165 | — |
| Targeted rerun set (union minus 9 mention-only IDs that have evidence) | 156 | research/submission2_target_ids.txt |

Overlaps: capped & no-evidence 9; mention & has-evidence 11;
mention & capped 2. Contexts: 286 sources skipped (budget
exhausted), 1,495 excerpts truncated. Total generated tokens
128,491 (mean ~214/Q); 119 Qs used the 2-call length retry.

### Where the gap comes from — correctness, not evidence

Per-question score = 0.8×Correctness + 0.2×(Correctness×Evidence).
If average evidence support on correct answers is e, implied
correctness C = 0.2217 / (0.8 + 0.2e):

| Assumed evidence support e on correct answers | Implied correctness |
|---|---|
| 0.0 | 0.277 |
| 0.5 | 0.246 |
| 1.0 (perfect) | 0.222 |

So correctness is roughly **0.22–0.28**. Perfect evidence on every
currently-correct answer adds at most ~0.05 (0.2217 → ~0.277).
Reaching the leader's 0.369 with perfect evidence needs
correctness 0.369 — about **88 more fully-correct answers** out of
600 (fewer if some become partially correct). The gap is therefore
mostly a correctness gap: the model is getting the numbers / facts
wrong, or abstaining, not merely failing to cite.

### Why the model gets numbers wrong (context audit)

- CSVs showed only the first 50 rows; a question about row 900 was
  answered from an unrelated head sample. No full-file counts/sums.
- JSON arrays showed only the first 5 items (DCC.json: 15,962 trees).
- GeoJSON showed the first 20 features with NO areas — the model had
  to compute hectares from raw lon/lat coordinates in its head.
- SQLite (162 MB Helsinki graph) and GeoPackage were metadata-only;
  TIFFs were metadata-only (no raster stats at all).
- Spreadsheets showed the first 60 rows per sheet, no column stats.
- 12,000-char/question budget: 286 listed sources never reached the
  model at all; 1,495 excerpts cut mid-file.
- 56 answers hit the 512-token ceiling mid-answer (capped answers
  are usually also wrong/incomplete).
- 109 rows had no evidence lines at all (correct-with-no-evidence
  scores only 0.8, and uncited answers are more likely wrong).
- The model wrote literal "supported claim" as its evidence text
  in abstention answers, and cited sources not listed for the
  question (114 lines) — both are evidence-score losses.

## Fixes implemented (v2, this build)

### src/extract.py
- **COMPUTED summaries in every structured extractor** (verified
  against the source file itself, never guessed): CSV/JSON/
  spreadsheet/SQLite now carry per-column stats over the WHOLE file
  (n, blanks, distinct, numeric min/max/sum/mean, top values);
  GeoJSON carries per-feature projected areas in hectares (pyproj
  UTM, equirectangular fallback) plus a total; TIFF carries
  per-frame min/max/mean/nonzero over the whole raster.
  Summaries sit at the TOP of each extraction so they survive
  excerpt truncation.
- **SQLite / GeoPackage are now read natively**: schema, row counts,
  SQL aggregates per column, sample rows, WKB geometry decode
  (proper GeoPackage header parse: bit 0 = binary type, bits 1–3 =
  envelope indicator; Point/LineString/Polygon area/length/centroid).
- **TIFFs read via Pillow + numpy**: size, mode, frames, tags,
  per-frame stats. (Median dropped: too slow on 16M-cell frames.)
- **Focus-aware excerpts**: rows/items/features matching the
  question's tokens are shown first — ranked by IDF weighting
  (rare identifiers outweigh common words; a naive boolean match
  failed because "zone"/"Liverpool" match nearly every row).
  Focus tokens drop question-phrasing stopwords. Small files only
  (<10 MB); large files keep the generic cached extraction.
- **Spreadsheet header-row skip** for stats (a text header no
  longer turns a numeric column into a text column).
- **3D coordinates handled** in ring-area computation ([x, y, z]).

### src/build_contexts.py
- Budget 12,000 → **20,000 chars/question**; per-excerpt 8,000 →
  10,000; per-source 20,000 → 30,000. Token check: ~20k chars ≈
  5,000 input tokens; prompt + output (≤2,048) stays well under
  the 12,000-token/question rule limit.
- Source priority: .sqlite/.gpkg → 2, .tif/.tiff → 3 (they now
  carry real content).
- Focus tokens passed per question (two-tier cache, see above).

### src/colab_run.py
- **num_predict 512 → 1024** (56 capped answers in #1).
- **Evidence second-pass call**: if the answer is not an
  abstention and has no evidence lines, one extra evidence-only
  call asks the model to cite the claims it just made. At most
  3 calls/question (≤15 allowed).
- **Stronger evidence prompt**: evidence lines must carry the
  ACTUAL claim (a number, a name, a value) — never the literal
  words "supported claim". System prompt tells the model to use
  [COMPUTED ...] values exactly as given and NOT recompute them.
- **--ids-file**: run only the listed question IDs (targeted
  rerun of the 156-ID failure set).

### Not changed
- **build_submission.py**: unchanged — it already parses the
  "evidence" field, drops literal "supported claim" filler, drops
  SIDs not listed for the question, and falls back to the first
  excerpt source. Verified it still builds a valid 600-section
  file from the old answers.jsonl (format check to /tmp only).
- **Model / rules**: still exactly one model, Qwen3-1.7B via
  Ollama, temp 0, seed 42, num_ctx 32768, ≤3B params asserted.
  No model >3B anywhere, no paid APIs, no training on test data.
- **.laz / .ifc**: still metadata-only. No LAZ height or IFC
  geometry computation this round (parsers not available;
  out of scope — see "Not yet done" below).
- **No graph routing**: the Helsinki SQLite graph gives schema +
  aggregates + sample rows, but no shortest-path computation.
  No per-cell raster values, no cross-source joins, no
  multi-column spreadsheet formulas in code.

## Verification (smoke tests, local VM, no GPU)

- All four scripts compile (py_compile).
- Train Q1 (Liverpool zone 1912/A1): focus ranking puts the
  feature first with **area_ha=3.4798 → 3.48 ha**, matching the
  Train answer. (Before the IDF fix it was buried: "Liverpool"
  matches 38 features, "1912/A1" only 1.)
- CSV computed summary verified on tpo-groups.csv (713 rows);
  DCC.json summary over all 15,962 items; SQLite aggregates on
  helsinki_graph (65,040 nodes / 130,352 edges); TIFF frame
  stats on surface.tif (4000×4000, 5 frames); GPKG points decode
  to BNG metres (e.g. 342473.85, 387380.03).
- Spreadsheet: col B numeric min 59,840 / max 125,304 /
  sum 860,454 / mean 86,045.4 after header skip.
- build_contexts 5-question test to /tmp: 5/5 records, computed
  blocks present, resume works (1 skipped-done on re-run).
- build_submission.py on old answers.jsonl → /tmp: 600 sections.
- Bugs found and fixed by the smoke tests: WKB header parse
  (envelope indicator is bits 1–3, not 0–2), WKB 4-tuple vs
  2-tuple mismatch, 3D ring coordinates, blob values in SQLite
  top-values, focus boolean-match swamping, TIFF median slowness.
- Full contexts rebuild (v2): backed up v1 to
  data/contexts_v1.jsonl + data/contexts_log_v1.json first.
  [rebuild stats to be filled in when it finishes]

## Full Colab run — cost and honest expected gain

- **Full rerun (recommended)**: 600 Qs at num_predict 1024,
  ~10–12 s/Q on Colab T4 → **~100–120 min**, plus ~1 extra
  evidence call on ~15% of questions. Tokens: ~128k generated
  in #1; expect ~180–250k generated tokens at 1024 cap, input
  ~5,000 tokens/Q worst case — every question stays under the
  12,000-token limit.
- **Targeted rerun (156 IDs only)**: ~25 min, same per-Q cost.
  Use `--ids-file research/submission2_target_ids.txt` with a
  fresh answers file, then merge with the #1 answers for the
  other 444 Qs before build_submission.py.
- **Honest expectation**: the public score is LLM-judged and
  cannot be predicted precisely. The fixes remove the clearest
  mechanical losses (capped answers, missing evidence, missing
  sources, wrong arithmetic from head samples), which plausibly
  converts a share of the 165 failure-mode questions. A
  reasonable range is **+0.02 to +0.08** on a full rerun
  (0.24–0.30). The targeted 156-ID rerun alone cannot close
  the 0.147 gap — it touches at most 165 questions and most
  will not all flip to correct. The leader (0.369) is likely
  still out of reach on this round alone; the remaining gap
  needs the "not yet done" items below (graph routing, LAZ/IFC,
  raster cell values) plus possibly a second iteration on the
  new failure modes.

## Not yet done (candidates for submission #3)

- LAZ heights / IFC geometry (need laspy / ifcopenshell; not
  installed; parsers not attempted).
- Shortest-path / graph routing over helsinki_graph.sqlite.
- Per-cell raster values from TIFFs (stats only for now).
- Multi-column spreadsheet formulas and cross-source joins.
- Teammate decision (solo = prize-disqualified) and the Funding
  Portal narrative application (due 13 Dec 2026) — separate
  user decisions, not this build.
