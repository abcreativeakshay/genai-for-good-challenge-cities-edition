# GenAI for Good Challenge: Cities Edition — Phase 1 Technical Assessment (Zindi)

- Competition: https://zindi.world/competitions/GenAI-for-Good-Challenge
- Closes: 13 Dec 2026, 23:59 EST. Prize: up to $40,000.
- Rules (verified 2026-11-07): 1 submission/day, 20 total. One generative model ≤3B params, max 15 model calls and 12,000 generated tokens per question. Open-source only, no paid services. Top 100 go to panel evaluation; final ranking uses the better full-Test score of the two selected submissions.
- Teams: min 2, max 4, all 18+. Solo submission may be ineligible for prizes — teammate still to be confirmed.
- Status 2026-11-07: RESEARCH FIRST — no model built, nothing submitted yet. Three research strands running (rules/scoring/data; ≤3B model + RAG; winning approaches), plus a Zindi browser task pulling the Data page.

## Layout
- `data/` — Zindi downloads (train/test/sample, AllowedDatasets.csv, SourceManifest.csv, source docs). Not yet downloaded.
- `research/` — research reports (to be added as they land).
- `src/` — RAG + generation pipeline (to be built after research).
- `submissions/` — validated submission CSVs + submission log.
- `notes.md` — running log.

## Research-first rule
No submission is built until the research strands confirm: exact metric, sample-submission format, test size, and the best ≤3B model for grounded city-document QA.
