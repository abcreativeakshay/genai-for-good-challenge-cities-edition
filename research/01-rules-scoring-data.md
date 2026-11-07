# Deep Research 1 — Rules, Scoring, Data (7 Nov 2026)

Research only. Nothing built or submitted. All sources fetched 7 Nov 2026.
Subagent report saved verbatim below.

## 1. Official rules & structure — VERIFIED

Sources:
- Official Rules PDF: https://ieeeht.org/wp-content/uploads/2026/10/GenAI-for-Good-Challenge_-Cities-Edition-2026-Official-Rules.pdf
- IEEE challenge page: https://ieeeht.org/get-involved/funding-opportunities/genai-for-good/
- Zindi page: https://zindi.world/competitions/GenAI-for-Good-Challenge

Entry period: 00:00 EST 5 Nov 2026 – 23:59 EST 13 Dec 2026 (IEEE server is the official clock).

What the $40,000 is — correction to the Zindi headline: it is **per winning team, not a $40k total pool, and not a Zindi cash prize.**
- 9 finalists (3 per city) each get a fixed **$10,000 IEEE grant** for prototype development (aggregate $90,000).
- 3 winners (1 per city) are then eligible for an innovation award of **up to $30,000** to develop/pilot the solution, under a separate agreement, through 31 Dec 2027.
- So a winning team can access up to $10k + $30k = $40k.

Phases:
- Phase 1 (Open Call): narrative application (Funding Portal) + technical assessment (Zindi). Expert panel selects up to 20 semi-finalists.
- Phase 2 (Use Case Sprint): workshops, revised proposals → 9 finalists (3 per city).
- Phase 3 (Prototype): finalists build prototypes, pitch live; 3 winners (1 per city) → Phase 4 piloting through end-2027.
- Zindi's role is only Step 1 of Phase 1. A Zindi submission alone is not a complete entry under IEEE Rules §6–7.

Phase 1 panel criteria (IEEE Rules §11): Use Case Alignment 20%, Impact Potential 25%, Team Capability 20%, Technical Approach 35%.

Technical-assessment filter: top 100 technical performers move to panel evaluation. Zindi rules: provisional top 100 after final evaluation are emailed for code review.

Support events: Technical Office Hours 10 Nov 2026, 9:00 AM ET (GENIE.AI Q&A); live Q&A 19 Nov 2026, 11 AM ET. GENIE.AI replica repo is read-only reference only; full access/support opens for selected finalists in December. Phase 1 is about shaping the concept and outlining how it could connect to the GENIE AI framework — no prototype yet.

## 2. Zindi technical assessment — PARTLY VERIFIED, METRIC UNKNOWN

Verified from the Zindi page (crawled 7 Nov 2026):
- 1 submission/day, 20 submissions overall.
- One generative model, ≤3B parameters (quantisation does not reduce the count), max 15 model calls and 12,000 generated tokens per question, including thinking, retries and self-checks.
- Other tools for retrieval, OCR, embeddings, reranking, spatial analysis and calculations are permitted provided they do not invoke additional generative models.
- Open-source languages/tools/packages only; no paid services or free trials requiring a credit card; no AutoML.
- Public leaderboard scores a subset of Test questions. Before close, each team selects 2 submissions for final evaluation (else the 2 best public scores are auto-selected). After close, organisers evaluate the withheld questions for those 2 and compute full-Test scores "using the same scoring method"; the better full-Test score determines final ranking. Tiebreak: earlier submission wins.
- A submission that fails processing gets no score and does not count against the daily limit.
- Test-material ban: test questions, test source materials, their public copies and derivatives, and any answers generated from them must not be used for training, fine-tuning, distillation, self-training, pseudo-labelling or other learned adaptation. Test sources are inference-only. OCR/parsing/indexing/retrieval/spatial analysis/calculations are permitted without training on test materials.
- Additional datasets only if listed in AllowedDatasets.csv on the Data page; anything else needs organiser approval. All sources/licences must be declared; each source retains its own licence per SourceManifest.csv.
- Code must be released under an approved permissive licence (Apache 2.0 / MIT or equivalent); entrants must be legally entitled to assign copyright in the winning code to Zindi.
- Code review: provisional top 100 are emailed and have 48 hours to submit code (Submissions section). Conflicting line in Monitoring section: code requested mid-challenge must be supplied within 24 hours. Code that does not reproduce the score → rank adjusted to the reproduced score; code that does not run → possible disqualification. Review checks end-to-end reproducibility, source eligibility, Train/Test separation, and model-call/token-limit compliance. Organisers may run the solution on fresh cases. Custom packages in the submission notebook are not accepted; set random seeds; document all model versions, settings, dependencies and data sources.
- Cheating: 1st offence = no prizes for 6 months + 2,000 points removed; 2nd offence = account banned. Code/information must not be shared privately outside a team — shared code must be posted publicly on the discussion boards.

NOT VERIFIED — the scoring metric is not stated anywhere public:
- No public source names the metric. The Zindi page, the IEEE rules PDF, and the IEEE page all say "scored" / "same scoring method" without defining it (no exact-match, F1, ROUGE, BERTScore, or LLM-judge formula found).
- No sample-submission file, its columns, row count, answer-length limit, test-set size, or question format found in any public source. Those live on the logged-in Zindi Info/Data pages only.
- Do not import the metric from another Zindi GenAI challenge: a published solution doc for the unrelated "Multilingual Health QA" Zindi challenge used 0.37·ROUGE-1 + 0.37·ROUGE-L + 0.26·LLM-judge, with three separately-scored answer columns — that is a different competition and there is no evidence this one uses it.

## 3. Team rules — VERIFIED, solo = ineligible for prizes

- IEEE Rules §8: "Each team must have at least 2 members… If any team is discovered to have less participants, that team will be immediately disqualified." Each member must be 18+ (or age of majority in their jurisdiction, whichever is older), a legal resident where permitted by local law, not on US/UN sanctions lists, not an IEEE employee/immediate family. One person may join only one team; being on multiple teams disqualifies that person from all teams and may disqualify the teams.
- Zindi rules: teams must be informal groups of at least 2, max 4; corporate entities ineligible; every member 18+; only one teammate needs to submit on Zindi. "Until a second data scientist accepts an invitation to join a team, the data scientist who initiated a team remains an individual on the leaderboard" — i.e. a solo account can appear/submit on Zindi, but under IEEE rules it is disqualified as a team of <2, so it cannot advance or win.
- Forming a team late costs submissions: "When creating a team, the team must have a total submission count less than or equal to the maximum allowable submissions as of the formation date. A team will be allowed the maximum number of submissions minus the total number of submissions among team members at team formation." So solo submissions made before teaming are deducted from the team's 20.
- No additional members in the final 5 days. A team can be disbanded only if it has not yet submitted; once a submission is made, members cannot leave. Teams can merge only if the combined size ≤ max team size. Multiple accounts per user = immediate disqualification.

## 4. Data & use cases — VERIFIED (briefs); Zindi dataset contents NOT public

The three use cases (one per team; a team must pick exactly one — multiple applications / multiple use cases = immediate disqualification, IEEE Rules §7):

1. Pristina, Kosovo — Transit Route Planning AI Assistant. https://ieeeht.org/wp-content/uploads/2026/11/GAI4GC-Cities-Use-Case-Transit-Route-Planning-AI-Assistant-Pristina-Kosovo.pdf — Multilingual (Albanian, Serbian, English) GenAI assistant consolidating routes/schedules, traffic data, GPS tracking, IoT/Smart City Platform data and resident feedback into evidence summaries with source attribution for route reviews. GIS layers and unstructured planning documents are out of MVP scope. Records mostly in Albanian, some Serbian/English.
2. Rivne, Ukraine — Energy Advisory AI Assistant. https://ieeeht.org/wp-content/uploads/2026/11/GAI4GC-Cities-Edition-Use-Case-Energy-Advisory-AI-Assistant-Rivne-Ukraine.pdf — Building profile (type, age, heating, consumption) → recommended measures, indicative costs/payback, financing, exportable report. Primary basis: Ukrainian state standards, energy audits, past-project data (PDF/DOCX/XLSX). Estimates must be labelled preliminary, not a substitute for an audit; verified records vs user-reported vs assumptions must be distinguished.
3. Skopje, North Macedonia — Green Infrastructure AI Assistant. https://ieeeht.org/wp-content/uploads/2026/11/GAI4GC-Cities-Edition-Use-Case-Green-Infrastructure-AI-Assistant-Skopje-North-Macedonia.pdf — Reads construction permits, tree-removal approvals, landscaping plans, scanned PDFs/Word files (mostly Macedonian, some Albanian, technical English) and proposes georeferenced updates to the Green Cadastre, with expert review before any official update. No autonomous permitting/enforcement decisions.

Implication for the Zindi test: questions are very likely multilingual (Albanian/Serbian/Macedonian/Ukrainian/English) and grounded in city documents — retrieval quality and source attribution matter, and a model weak in those languages will underperform. But the actual Zindi train/test structure (file names, question types, languages, counts) is not public: AllowedDatasets.csv and SourceManifest.csv are named on the Zindi rules page as living on the Data page, and their contents, plus Train/Test/sample-submission files, require the logged-in Zindi Data page to confirm.

GENIE.AI framework: ITU open-source framework (built on OPEA, Apache 2.0, GitLab release pending) for public-sector GenAI chatbots/assistants; 100+ early adopters, 10 countries, 30 use cases. https://www.itu.int/itu-d/sites/genieai/ — Phase 1 does not require building on it yet.

First edition (2025–26) — do not confuse with this one: health/agriculture/climate, >300 submissions from 79 countries, 9 finalists, 3 winners, up to $25,000.

## 5. Zindi discussion / starter notebooks / organiser clarifications

None found for this competition. Searches returned only LinkedIn announcements and unrelated Zindi discussions. No discussion thread, starter notebook, leaderboard snapshot, or organiser clarification for this competition is indexed publicly as of 7 Nov 2026 — expected, since it opened only 5 Nov 2026. Any discussion content will need the logged-in Zindi Discussions tab.

## Still uncertain — needs the logged-in Zindi Data/Info pages
1. Scoring metric and its exact formula (ROUGE? LLM judge? human? composite weights?).
2. Sample submission format: exact columns, row count (= number of test questions), answer format/length limits.
3. Train/Test file list, sizes, question types, languages, and whether test questions span all 3 cities or one.
4. Public vs withheld split proportion.
5. Whether any discussion thread / starter notebook / organiser note already exists on Zindi.

## 3 biggest risks
1. Optimising blind on the metric. With 1 submission/day and 20 total, each submission is expensive. A verbose, judge-friendly answer and a short extractive answer score very differently under ROUGE vs an LLM judge; building the pipeline before reading the Info-tab metric risks burning the daily slots on the wrong answer style.
2. Treating Zindi as the whole challenge. A top-100 Zindi score alone advances nothing without (a) a team of ≥2, (b) a narrative application on the Funding Portal for exactly one use case, and (c) both by 13 Dec 2026 23:59 EST. A solo entrant is disqualified under IEEE Rules §8 even with a top score, and late teaming deducts prior solo submissions from the team's 20.
3. Breaking the model/data rules during "research". Using any model >3B parameters — including as a judge, reranker, or self-check — breaches the "one generative model" limit; training/fine-tuning on test sources or their derivatives (including pseudo-labelling) is banned; using a dataset not in AllowedDatasets.csv without organiser approval, or a paid API, risks disqualification and the 6-month / 2,000-point penalty. Code must reproduce the score within 48h of the top-100 email, so the pipeline must be deterministic, seeded, and fully documented from the first submission.
