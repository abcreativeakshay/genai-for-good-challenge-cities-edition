# Zindi Data/Info pages — verified 7 Nov 2026 (browser task 21:38–21:46 IST)

User was already signed in to Zindi. Browser visited Info/Overview, Data and Team tabs and the Zindi Team Rules page. Did NOT submit anything, did NOT join/create a team, did NOT accept rules.

## 1. Dates, prize, metric, test size, teams

- Opens: 7 Nov 2026 (Zindi header; IEEE says entry period 5 Nov). **Closes: 13 Dec 2026 — no time of day published on Info/Data pages (header shows date only).** IEEE rules say 23:59 EST 13 Dec. Reveals: 14 Dec 2026. Semi-finalists notified early February; finalists before 28 Feb. Phase 3 (pilot co-design) Mar–May 2027; Phase 4 (funding) 15 May–Dec 2027. Dataset proposals must be sent by 30 Nov 2026.
- Prize/funding: up to **$40,000 USD total** — up to $10,000 prototype funding + up to $30,000 pilot funding per winning team; 3 funded winners. The technical assessment itself has **no financial reward**; there are 5,000 Zindi points available. 15–20 semi-finalists, 9 finalists (page also mentions provisional top 100 for code review).
- **METRIC — FOUND (was unknown in Research 1): LLM-judged against checked reference answers and source evidence; equivalently-worded answers are accepted. Per-question score = 0.8 × Correctness + 0.2 × (Correctness × Evidence support). Correctness: 1 = correct, 0.5 = partially correct, 0 = incorrect/missing. Evidence support: 1 = complete supporting reference, 0.5 = incomplete, 0 = missing/irrelevant/incorrect (evidence credit is multiplied by correctness). Final leaderboard scores range 0 to 1. Evidence references are worth 20% and are forfeited entirely if the answer is wrong.**
  - Implication: short, correct, directly-evidenced answers. A wrong answer with perfect evidence still scores 0. A correct answer with no evidence scores 0.8. Partial correctness caps at 0.5 + 0.1 = 0.6 max.
- Test size: exact number of Test questions is **not stated on Info/Data pages**. Public leaderboard is a **hidden subset** of Test questions; after close, **withheld** questions are evaluated for the **two submissions the user selects** for final evaluation, and the better of the two full-Test scores determines final ranking. Finalist solutions are also re-run on previously unseen cases under the published model/compute limits.
- Submission limits: **1 submission/day, 20 submissions overall**; select 2 submissions for final evaluation.
- Teams: **maximum 4 members**. All members must complete their Zindi profile. A team shares the submission limits (1/day, 20 total). Once registered with a team for this competition you cannot continue to submit as an individual. Team leader splits prizes; all team members receive points. No additional members in final 5 days. Teams can merge (up to max size), but not after a submission has been made.
- Model/data constraints: one generative model ≤3B params; max 15 model calls and max 12,000 generated tokens per question (incl. thinking, retries, self-checks); quantisation does not reduce param count. Only provided datasets and those in AllowedDatasets.csv; extra datasets only after organiser review (proposals by 30 Nov 2026, list updated Mondays). Test questions/sources/derivatives/generated answers must NOT be used for training, tuning or adapted evaluation; processing Test sources at inference time (OCR, parsing, indexing, retrieval, spatial analysis, calculations) is permitted.

## 2. Data tab — 7 files

| File | Size | Description |
|---|---|---|
| Train.csv | 9.7 KB | Development examples with checked reference answers and supporting evidence |
| Test.csv | 385.3 KB | Complete set of Test questions and associated source materials; questions to answer, with context and source references |
| SampleSubmission.md | 70.3 KB | Template showing required Markdown submission format, field structure and answer length limits; no example answers |
| StarterNotebook.ipynb | 3.5 MB | Compact QA baseline that loads the data, explores source materials and prepares a submission file |
| SourceManifest.csv | 5.6 MB | Canonical register of all source materials: source IDs, descriptions, publishers, licences, attribution, file sizes, SHA-256 checksums, file paths in source archive |
| AllowedDatasets.csv | 4.2 KB | Register of approved external datasets/packages, distinguishing training/development resources from inference-only Test sources |
| SRC_535ab8cab643__3dbag_page_a.json | 437.5 MB | Source documents, tables and spatial data; see SourceManifest.csv |

Discrepancy: Data page About text refers to a "SourceMaterials.zip" to extract into a sources folder, but **no .zip file is listed** — only the single 437.5 MB JSON above.

## 3. Downloads — NOT in workspace yet

Browser triggered downloads of all 7 files (Train.csv and SampleSubmission.md via tabs t2/t3; other five via download arrows), but could **not** save them into the workspace: the browser's file-transfer tool cannot target these download controls (icon-only links, zero area in accessibility tree), and "Download all as .zip" was not usable. Direct download URLs follow `https://api.zindi.world/v1/competitions/GenAI-for-Good-Challenge/files/<filename>?auth_token=<per-session token>` — tabs t2/t3 still hold live examples; token is a session credential, not reproduced. Checked 7 Nov ~21:47 IST: no GenAI files in `~/workspace/browser_downloads/` (only older Nov 1–5 files). **Next step: get the files into `~/workspace/zindi-genai-for-good/data/` — continue the same browser task and have it transfer/read the small files (Train.csv 9.7 KB, SampleSubmission.md 70.3 KB, Test.csv 385.3 KB, AllowedDatasets.csv 4.2 KB) first; the 437.5 MB source JSON last.**

## 4. Sample submission — Markdown, not CSV

SampleSubmission.md is a **Markdown template, not a CSV** — no CSV columns/rows. Browser could not open/read its contents (Zindi serves it as a download), so exact internal structure, row count and first rows are **unverified** until the downloaded file is read. Data page states: include **every Test ID exactly once**, followed by an answer in natural English and evidence identifying the original source pages, cells, records, features or model elements. **Invalid file structure is rejected; a structurally valid blank answer scores zero.** Answers must be short and directly evidenced; the template defines answer length limits (numeric limit inside the template, not on web pages — do not rely on an unverified 280-character figure). Train.csv has questions with checked reference answers + supporting evidence; Test.csv has the complete set of Test questions. About section describes five use cases (including Colombo, Skopje green infrastructure, Skopje air quality, Bogotá and San José) and says application also requires a separate narrative proposal through the funding portal — Zindi is only the technical assessment half.

Retained tabs: t1 (Info/Data/Team), t2 and t3 (Train.csv and SampleSubmission.md download URLs).
