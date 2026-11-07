# Data analysis — read from the actual files, 7 Nov 2026 ~21:55 IST

## Files in data/ (6 of 7; source archive pending)
- Train.csv 9.8 KB, Test.csv 386 KB, SampleSubmission.md 71 KB, StarterNotebook.ipynb 43 KB, SourceManifest.csv 122 KB, AllowedDatasets.csv 32 KB
- NOTE: Data page sizes changed between browser runs (StarterNotebook was 3.5 MB, now 42.4 KB; SourceManifest was 5.6 MB, now 121.8 KB; AllowedDatasets was 4.2 KB, now 31.7 KB). Organisers updated files on 7 Nov. The files in data/ are the current versions.

## Test: 600 questions, 155 cases
- Columns: ID, Case_ID, Domain, Question, Context, Source_IDs
- Domains: buildings_energy 261, urban_greenery 191, transport 148
- Questions per case: mean 3.87, min 1, max 12
- Avg question 160 chars, avg context 284 chars, avg 5.45 sources/question, max 12
- Distinct Test Source_IDs: 161, all present in manifest (0 missing)
- Most-used sources cluster by case (36x each): Accra/Burgos/Kenya HTML+PDF+Analysis.md; 32x each: anlagenbaeume/strassenbaeume GML/GeoJSON/GPKG/XSD

## Train: only 9 rows, 3 per domain
- Columns: ID, Case_ID, Domain, Question, Context, Source_IDs, Answer, Evidence
- Answers are SHORT, numeric, precise, with evidence locators like "bus01.ods, sheet BUS01a, cell H23" / "zones-2026-06-23.geojson, properties.reference=1912/A1" / "trees-dataset-2026-06-23.csv, reference=TPO453, made_date and confirmed_date"
- This is structured-data QA (spreadsheets, GeoJSON, CSV, GTFS, PDFs), NOT free-text RAG. Many questions require calculation in code (areas from geometry, percentages from workbook cells, date differences, sums). The model alone will get numbers wrong; compute deterministically in Python and let the model write the cited answer.

## SampleSubmission.md: 600 sections, Markdown
- Format per question: `## <Test ID>` then answer text, then `### Evidence` then `- SOURCE_ID | locator | supported claim` lines. No length limit stated in the template itself. Keep IDs exactly once, in supplied order (notebook save_submission asserts this).

## SourceManifest: 188 sources, 1,204 MB total uncompressed
- Formats: .txt 33, .json 31, .csv 29, .md 16, .geojson 14, .ods 13, .xlsx 9, .pdf 8, .tif 7, .xml 6, .html 5, .gpkg 3, .gml 2, .laz 2, .zip 2, .jpg 2, .xsd 2, .docx 2, .sqlite 1, .ifc 1
- Largest: helsinki_graph.sqlite 162.7 MB, urban_building_stock_datasets 139.2 MB, current.csv 136.0 MB, survey.laz 92.0 MB, surface.tif 80.5 MB, terrain.tif 77.9 MB, shapes.txt 69.5 MB, inventory.geojson 65.6 MB, stop_times.txt 50.3 MB, history.csv 46.4 MB, register.geojson 46.2 MB
- 20 sources >10 MB, 44 >1 MB, 108 <100 KB. Archive for all: SourceMaterials.zip (per manifest + notebook).

## Source archive mystery
- Data page lists a 437.5 MB file SRC_535ab8cab643__3dbag_page_a.json, but manifest says SRC_535ab8cab643 is 13,324 bytes (3dbag_page_a.json). The 437.5 MB file may be the full SourceMaterials.zip under a misleading name (1.2 GB of mostly text/CSV/GeoJSON could compress to ~437 MB) — verify by magic bytes after download. Browser task asked 7 Nov ~21:56 IST to transfer it via the row-button trick + try "Download all as .zip".

## StarterNotebook (17 cells) — organiser baseline, read fully
- Workflow: question + context → relevant source locations → tools/calculations → one compact model → answer + evidence. "The baseline illustrates the workflow; it is not a complete reader for every native format."
- Expects SourceMaterials.zip beside CSVs, extracts to sources/, verifies manifest sizes + SHA.
- Model: Qwen3-1.7B via Ollama, model "qwen3:1.7b", think=False, temp 0, seed 42, num_ctx 8192, num_predict 512; asserts total parameter_count <= 3,000,000,000. This settles the compliance question for the baseline: Qwen3-1.7B is the organiser-endorsed model. (OCC-RAG-1.7B remains an untested challenger.)
- Metric (cell 13): element score 0.8*c + 0.2*c*e; question score = mean element score; aggregate within cases, then topics, with the three topics weighted equally. "Automatic checks validate file structure and IDs... They do not independently verify arbitrary calculations in free text."
- Submission writer (cell 16): refuses to export unless predictions cover exactly all 600 Test IDs; asserts answer non-empty, no "## " inside answers, evidence lines single-line strings; IDs in supplied order.
- Source extraction demo (cell 9): PDF via pypdf, ODS via content.xml text, XLSX via pandas head(60) per sheet, else strip HTML tags from text.

## Plan for submission #1 (today)
1. Get source archive (437 MB file / zip) — blocker for grounded answers.
2. Build readers for the formats actually used by Test questions (ODS/XLSX/CSV/GeoJSON/JSON/TXT/MD/PDF first; GPKG/GML/SQLite/TIF/LAZ later or skip).
3. Per question: extract only its listed Source_IDs, compute deterministically in Python where the question is arithmetic, then 1 Qwen3-1.7B call (temp 0, <=500 tokens) to write the short answer + evidence lines.
4. Run generation on Colab T4 (600 Q x ~10s = ~100 min) with per-question checkpointing; CPU fallback is 1-2 min/Q = 10-20 h, not viable today.
5. Validate with the notebook's save_submission logic (600 IDs, order, non-empty), submit via Zindi (1/day).
