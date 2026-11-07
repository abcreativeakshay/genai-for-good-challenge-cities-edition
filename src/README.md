# GenAI for Good Challenge: Cities Edition — submission pipeline

Phase 1 Technical Assessment on Zindi. Deadline **13 Dec 2026 23:59 EST**.
1 submission/day, 20 total. Metric (LLM-judged, per question):
**score = 0.8 × Correctness + 0.2 × (Correctness × Evidence support)**,
Correctness / Evidence each 1 / 0.5 / 0. A wrong answer scores 0 even
with perfect evidence; a correct answer with no evidence scores 0.8.
Top 100 go to panel evaluation; final ranking uses the better
full-Test score of the team's 2 selected submissions.

**Teams must be 2–4 people, all 18+** (IEEE Rules §8). Solo
submission may be ineligible for prizes — a teammate is still needed.

## The one model

Exactly one generative model, everywhere: **Qwen3-1.7B via Ollama
(`qwen3:1.7b`)**, the organiser baseline from the StarterNotebook.
- 1 chat call per question, `think=False`, temperature 0, seed 42,
  `num_ctx` 8192, `num_predict` 512 (≤ 12,000 generated tokens/question).
- `src/colab_run.py` asserts parameter count ≤ 3,000,000,000 before
  generating, exactly as the notebook does.
- No other generative model anywhere — not as judge, reranker or
  self-check. No paid APIs. Retrieval / parsing / arithmetic are
  plain code, not generative models.
- **Never train or fine-tune on test materials.** Extraction and
  inference-time answering are the only uses of Test data.

## Files

| File | What it does |
|---|---|
| `src/extract.py` | Readers by suffix, each returning plain text with locators. `.txt/.md/.json/.csv` (head + row counts, header names kept), `.geojson` (features + properties), `.xlsx/.ods` (pandas, per sheet, cell refs like `sheet Tulokset, cell AA9`), `.pdf` (pypdf), `.html` (tags stripped), `.xml/.xsd/.gml` (raw text), `.zip` (member list), `.docx` (text via document.xml). `.gpkg/.sqlite/.tif/.laz/.ifc/.jpg` are **metadata only** — size + a note, never faked values. |
| `src/build_contexts.py` | Per Test question, reads **only its listed Source_IDs**, prioritises Analysis/SourceGuide files first, then small text/json/csv, geojson, spreadsheets, pdf/html/xml; caps total excerpt text at ~12,000 chars/question. Writes `data/contexts.jsonl` + `data/contexts_log.json`. Deterministic, resumable (skips IDs already written). |
| `src/colab_run.py` | Runs on Google Colab (free T4): installs/starts Ollama, pulls `qwen3:1.7b`, verifies ≤3B params, 1 call/question, writes `answers.jsonl` **after every question** (per-question checkpoint) and resumes from it. `--smoke` = 5 questions. |
| `src/build_submission.py` | Validates `answers.jsonl` against `Test.csv` (exactly 600 IDs, in order, non-empty answers, no `## ` inside answers, single-line evidence) and writes `submissions/Submission.md` in the SampleSubmission.md format. Refuses to write if any ID is missing. If the model cited nothing, it falls back to the question's first excerpt source and **logs** the fallback. |

## How to run

```bash
# 0. Set up (once, local VM)
cd ~/workspace/zindi-genai-for-good
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 1. Build contexts (local VM, no GPU needed; ~minutes)
.venv/bin/python -m src.build_contexts
#    -> data/contexts.jsonl, data/contexts_log.json

# 2. Generate answers (Google Colab, free T4 GPU -- NOT on the VM)
#    Runtime > Change runtime type > T4 GPU.
#    Upload data/contexts.jsonl to the Colab session, then:
#    !python3 src/colab_run.py --input contexts.jsonl --output answers.jsonl
#    Smoke test first:  !python3 src/colab_run.py --smoke
#    Download answers.jsonl back to data/answers.jsonl locally.
#    Expected: ~10 s/question, ~100 min for 600 questions.

# 3. Build + validate the submission (local VM, no GPU needed)
.venv/bin/python -m src.build_submission
#    -> submissions/Submission.md

# 4. Submit submissions/Submission.md on Zindi (1/day limit).
#    Do NOT submit placeholders; the validator refuses incomplete files.
```

## What is NOT yet implemented

- **Native geometry / raster / point-cloud computation:** `.gpkg`,
  `.sqlite` (162 MB Helsinki graph), `.tif` (surface/terrain rasters),
  `.laz` (LiDAR survey), `.ifc` (building model) are metadata-only.
  Questions needing areas from GeoPackage geometry, shortest paths
  from the SQLite graph, raster values, LAZ heights or IFC geometry
  will therefore get only a "not yet implemented" note in their
  context, and the model is instructed to say it does not have
  enough information rather than guess. These are the biggest
  known accuracy gap: the data analysis (research/05-data-analysis.md)
  found the Test set is calculation-heavy structured-data QA, so
  native readers + deterministic Python computation are the main
  upgrade path for submission #2.
- **Spreadsheet arithmetic in code:** the context builder hands the
  model cell values with cell refs, but does not yet compute sums /
  percentages / date differences deterministically in Python. The
  model does the arithmetic from the excerpts, which is error-prone
  for multi-cell calculations.
- **Cross-source joins** (e.g. CSV reference vs GeoPackage reference
  sets) are not computed in code yet.

## Data & git rules

- **Never commit data files.** `data/` (Train/Test, SourceManifest,
  sources/, contexts, answers), `submissions/` and `*.jsonl` are
  gitignored. Only code, research notes and docs go in the repo.
  Challenge data stays local only (it was removed from the public
  repo history on 7 Nov 2026).
- Research notes: `research/01-rules-scoring-data.md`,
  `research/02-model-rag.md`, `research/03-winning-approaches.md`,
  `research/04-zindi-data-page.md`, `research/05-data-analysis.md`.
