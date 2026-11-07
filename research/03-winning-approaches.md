# Deep research — winning approaches for GenAI / RAG QA competitions

Research only. No building or submission done. Sources are web search/open results from 7 Nov 2026; anything not found there is marked **UNVERIFIED**.

## What this edition actually is — and what is NOT yet known

- **Cities Edition, Phase 1 Technical Assessment (Zindi):** opened 5 Nov 2026, closes **13 Dec 2026, 23:59 EST**, up to $40,000. 1 submission/day, 20 total. One generative model ≤3B params, max 15 model calls and 12,000 generated tokens per question. Open-source only, no paid services. Top 100 go to panel evaluation; final ranking is the better full-Test score of the team's 2 selected submissions.
  - Source: Zindi competition rules page, fetched by parent agent; IEEE HT Cities Edition page: https://ieeeht.org/get-involved/funding-opportunities/genai-for-good/
- **UNVERIFIED — do not assume:** the scoring metric for *this* edition, the exact submission columns, the number of public vs withheld test questions, and the train/test file list. Those must come from the Zindi Data/Info pages (parent browser task in progress). The ROUGE+judge metric below belongs to a *different* Zindi challenge, not this one.

## 1. Prior GenAI for Good — no leaderboard to copy

The 1st edition (IEEE + ITU, launched 7 Oct 2025) was **not** a Zindi QA leaderboard. It was narrative/prototype + expert panel: >300 submissions from 79 countries, 9 finalists, 3 winners announced July 2026, up to $25,000 each:
- Agriculture: **AgriConnect Lesotho — Keletso** (farmer guidance)
- Health: **Innov8AI — ASante** (Gambia chronic-disease wayfinder)
- Climate: **Polisense AI — MEWA** (Bengali weather advisor for Bangladesh farmers)

Lesson: the 1st edition rewarded *local grounding, multilingual delivery, and a deployable prototype* (GENIE.AI framework, Apache 2.0, open-source), not a single metric. That panel logic still applies in Cities Edition Phase 2 — a high Zindi score alone does not win the $40k; the narrative proposal (separate Funding Portal submission) and deployability matter.

Sources:
- https://ieeeht.org/get-involved/funding-opportunities/genai-for-good/first-edition/
- https://www.itu.int/en/ITU-D/ICT-Applications/Pages/Initiatives/ITU_OSPO/Open-Source_AI_for_Public_Services/GenAI-for-Good-Community-Challenge.aspx
- https://ieeeorg-stg.ieee.org/about/news/2026/ieee-and-itu-announce-challenge-winners
- GENIE.AI: https://www.itu.int/hub/2026/10/genie-ai-empowering-countries-to-shape-their-ai-future/ — modular, open-source, model-agnostic, built on OPEA, GovStack-aligned; 100+ early adopters, 10 countries, 30 use cases. **UNVERIFIED:** whether Zindi Phase 1 scoring rewards GENIE.AI use at all — the rules only require open-source code.

## 2. The closest Zindi GenAI wins — what actually moved the score

### (a) Multilingual Health QA (Zindi / HASH / ITU) — winning private 0.694836
The best-documented recent Zindi generative-QA win. Metric there was **0.37·ROUGE-1 F1 + 0.37·ROUGE-L F1 + 0.26·LLM-judge (1–5, normalised)**, with three answer columns scored independently.
- Baseline retrieval: ~0.485 → winning 0.697 public / 0.695 private. Public→private gap only ~0.0024.
- **Per-subset winner map was the big lever:** duplicate-rich subsets (near-identical questions in train) — *retrieval + selection of a real human answer beat generation*. Generation-bound subsets — cross-base ensembling + self-consensus (MBR) pick won.
- Retrieval: **e5-large + bge-m3 + TF-IDF, fused; bge-reranker-v2-m3**; per-subset selector (LGBMRanker / cross-encoder).
- Generation: 4 QLoRA bases (Qwen2.5-14B, Qwen3-14B, Gemma-3-12B, Mistral-Nemo), MBR sampling n=16–32, self-consensus = max mean unigram-F1 vs other samples. Gains came from *cross-base diversity, not one stronger base*; diminishing returns past ~4 bases; a weak base drags consensus down.
- Per-column optimisation: ROUGE columns got the consensus pick; the judge column got a *separate* pick from a 7B judge — lifted the 26% judge component at zero ROUGE cost.
- **Transferability warning:** those bases were 12–14B. **Not allowed here** (≤3B, one generative model). The transferable parts are: fused retrieval, reranker, self-consistency *within the 15-call budget*, per-question-type routing, and optimising each scored column separately *if* this edition has multiple columns (UNVERIFIED).
- Operational discipline: build scripts asserted 2,618 rows, IDs match SampleSubmission, no empty cells; `reproduce.sh` staged A→E, seed=42 everywhere; judge deltas <0.003 treated as noise.
- Source: https://hf-mirror.net/davidlowjw/multilingual-health-qa-crossbase-qlora/resolve/main/DOCUMENTATION.pdf?download=true

### (b) Unifi / financial-PDF RAG (Zindi) — 10th place, koleshjr
- Pipeline was deliberately simple: ingest → embed → retrieve → answer. **No reranking**, and it still placed 10th. "Proprietary models definitely won" in that edition — GPT-4 and Gemini-pro, GPT-4 <$11 total, Gemini-pro <$1, near-identical accuracy. **Not transferable here:** proprietary models are banned in Cities Edition; the point is that *retrieval quality + prompt, not model size, separated the field once a capable generator was in place*.
- Two tricks that moved score: (1) **passing previous-year answers in the prompt** for year-disambiguation — the model stopped confusing 2022 vs prior years; (2) **adding company_name to chunk metadata** — some documents had ambiguous names.
- Two failure modes that cost the most: **units/magnitude** (PDF says 8.8Rbn, expected answer 8,800,000,000 — model kept returning 8.8 despite prompting); and **composite answers** (Impala = refineries + Rustenburg + Marula summed; SSW = SA PGM+Gold only, ignore Europe) — "75% of competitors never even noticed."
- Warning in the write-up: manually reading PDFs, extracting values to CSV, and embedding the CSV "gets amazing accuracies" but is not scalable. **In Cities Edition this is also a rule risk:** test sources are inference-only — no training, fine-tuning, distillation, self-training or pseudo-labelling on test materials. Hand-extracting test answers would violate that.
- Source: https://zindi.africa/discussions/19871 ; code: https://github.com/koleshjr/Unifi_Instruct_Rag

### (c) Telecom 5G MCQ on Zindi (cited in QMOS paper)
GPT-3.5 56.97, GPT-4 64.78, Phi-2 42.07, **Phi-2+RAG 66.39, +fine-tuning 76.90, +batch-shuffle 84.65**. A small open model + RAG + fine-tuning beat GPT-4 on a narrow domain. Caveat: MCQ, not open QA; and fine-tuning on *test* materials is banned here — fine-tuning is only legitimate on train materials / allowed datasets.
- Source: https://lacuna.dev.tiptreesystems.com/pdf/art_b161b7b8cf0e4f12b94ddfc7eb3f8f81

## 3. General RAG best practice, 2026 — what moves a leaderboard vs noise

- **Hybrid retrieval is the 2026 baseline, not an upgrade.** BM25 + dense, fused with Reciprocal Rank Fusion, beats either alone on BEIR/MTEB; a cross-encoder reranker on top adds **5–15 MRR points** on hard sets.
- **Chunking is the most under-estimated lever.** 2026 default: **512–1024 tokens, 50–100 tokens overlap**, respecting section boundaries; 256–512 is the common starting point for fact-dense corpora. Too large = coarse match + noise; too small = right sentence, no framing. Semantic / late chunking and **contextual retrieval** (prefix each chunk with a short parent-document summary) help: Anthropic's benchmark — contextual embeddings −35% retrieval failures, + contextual BM25 −49%, **+ reranker −67%**.
- **Fix the question, not just the index:** HyDE, decomposition, step-back prompting. **Self-checking retrieval** (CRAG grades the fetch, Self-RAG reflects, agentic re-query instead of guessing once) beats one-shot retrieval. GraphRAG/RAPTOR only pays for global / multi-hop / cross-document questions — over-building it is noise for factoid QA. "The wins are early" — retrieval/chunking first, late stages last.
- **Evaluation reality:** ROUGE-L Pearson with human scores ~0.32, BLEU ~0.52; LLM-as-judge reaches >80% human agreement but carries position, verbosity and self-preference bias and run-to-run variance. For RAG, the load-bearing metric is **faithfulness = supported claims / total claims**, measured claim-by-claim with an NLI model, not answer-level "is it grounded?". Token F1 still matters for short answers.
- **Answer length/style is a scoring lever, not cosmetics.** ROUGE rewards lexical overlap with a single reference — verbose paraphrases, extra facts, and wrong units all dilute it. Match the reference style/length seen in train (if train answers exist — UNVERIFIED for this edition). In the health win, a per-question word-budget hint was part of both training and decoding prompts.

## 4. Zindi-specific pitfalls

- **Public ≈ 20–30% of test; private ≈ 70% decides the final ranking** (standard Zindi pattern; exact split for this edition UNVERIFIED). Small public gains are often noise — the health winner treated judge deltas <0.003 as noise and validated every method choice on a held-out val proxy, not on the public board.
- **Tiebreak: identical scores → earlier submission wins.** A good score submitted early beats the same score later.
- **Failed processing ≠ a used submission** in this edition's rules: "if your submission fails during processing, it will not be counted and not receive a score; nor will it count against your daily limit." So a format error costs a day, not a slot — but only if it fails processing; a *valid* but bad file burns 1 of 20.
- **Code review for top 100 (this edition; top 10 in most Zindi challenges), 48 hours to respond.** Code must reproduce the submitted score or the rank is adjusted to the reproduced score; if it does not run, disqualification. Seeds set, README required, no manual intervention, only competition datasets + permitted pretrained models. Keep every submission immutable + its exact generating script + requirements + model version from day 1.
- **Licensing/attribution:** each source retains its own licence (SourceManifest.csv); only datasets in AllowedDatasets.csv may be added, and additional datasets need organiser approval *before* use. All sources/licences must be declared. Multiple accounts, or sharing code/information across accounts not in a team, = disqualification.
- **This edition's hard constraints (from the rules page):** test materials inference-only; OCR, parsing, indexing, retrieval, spatial analysis and calculations are permitted *without* training on test materials; retrieval/OCR/embeddings/reranking/spatial/calculation tools are permitted **provided they do not invoke additional generative models**. That means: a cross-encoder reranker or NLI verifier is allowed; a *second LLM as judge* is not (it would be a second generative model). The 15-call / 12,000-token budget includes thinking, retries and self-checks — budget self-consistency accordingly.

## 5. The three use cases — public context, language and data traps

- **Pristina, Kosovo — transit route planning:** Multilingual assistant (Albanian, Serbian, English) for planners; bring together routes, traffic data, residents' feedback; flag routes that no longer match demand; link every finding to its source. City runs Smart City Platform, GPS bus tracking, traffic monitoring, Trafiku Urban app, but planners compare systems by hand. Trap: stale timetables/maps in the corpus will be treated as fact by a naive RAG. Date/version metadata per chunk is essential; answer in the *question's* language.
- **Rivne, Ukraine — energy advisory:** Assistant takes residents / HOAs / small businesses from a building profile to indicative costs, financing, and a ready-to-share report. Trap: numbers go stale fast; a retrieved 2025 grant rate presented as current is a wrong answer. Cost figures need source + date in the answer; never present an indicative cost as a quote. Ukrainian-language sources + Cyrillic.
- **Skopje, North Macedonia — green infrastructure:** Connect records scattered across formats/departments to the digital **Green Cadastre** to safeguard trees/green spaces. Most files in Macedonian, many scanned. Trap: **Do not confuse Green Cadastre with the real-estate cadastre** (katastar.gov.mk / GCIS) — different system, different data. Core skill is Macedonian-Cyrillic **OCR + spatial joining** (parcel/tree location), not generic text RAG. Scanned PDFs need OCR quality checks per document.

Language note (from the health win, directly transferable): **use a multilingual embedder (bge-m3 / e5-large / LaBSE) and a Unicode-aware tokenizer** — the default ROUGE tokenizer dropped Ge'ez/diacritic tokens and collapsed Amharic to ~0. Same risk for Albanian diacritics, Serbian, Macedonian and Ukrainian Cyrillic. Keep original script in chunks; do not transliterate; answer in the question language.

## 7 actionable lessons, ranked by expected impact

1. **Get a valid baseline submitted on day 1 — format first, quality second.** A scored entry locks in the tiebreak clock, validates the exact columns/IDs against Zindi processing, and reveals the real metric behaviour. A perfect pipeline submitted on day 10 with a format error is worth less than a simple RAG file scored on day 1. Validate locally first: row count = sample submission, IDs match exactly, no empty cells, no extra columns.
2. **Retrieval decides the leaderboard; generation polishes it.** Build hybrid BM25 + dense (bge-m3 or e5-large, multilingual) with RRF, then a cross-encoder reranker (bge-reranker-v2-m3) over top-k. This is the single biggest documented jump over dense-only, and it is allowed (rerankers are non-generative tools). Do not start with model choice.
3. **Chunk with metadata, not just text.** 512-token chunks with overlap, respecting section/table boundaries, each prefixed with document title + section + date/version + language + city/use-case. The Unifi tricks (entity name in metadata, year disambiguation in prompt) and the Pristina stale-map / Rivne stale-grant traps are the same lesson: *the model cannot disambiguate what the index does not carry*.
4. **Normalise numbers and units deterministically, before generation.** Convert magnitudes, currencies, kWh/m², dates and year references in a post-processing step with explicit rules from the corpus — do not leave it to the ≤3B model in free text. This was the largest single failure class in the closest prior Zindi RAG competition.
5. **Match the reference answer's length, language and format.** If the metric is ROUGE-like (UNVERIFIED for this edition — confirm first), verbosity and paraphrase are penalised; if it is judge-based, unsupported extra claims are penalised. Either way: short, sourced, same-language answers, with "not found in the provided sources" instead of guessing, win over long confident ones. Use the 15-call budget for 3–5 self-consistency samples + a non-generative consistency pick, not for 15 different prompts.
6. **Route by question type, and check for near-duplicates in train first.** If a test question near-duplicates a train Q&A (check once the data lands), selecting/adapting that train answer beats fresh generation on overlap metrics — the health win's largest single gain. If no near-duplicate exists, generate. Decide the routing threshold on a held-out val split, never on the public board.
7. **Keep every submission reproducible from day 1.** Immutable submission files, pinned requirements, seeds, model name/version, exact prompt, retrieval config, and a one-command reproduce script. Top 100 get 48 hours to hand over code; a score that cannot be reproduced is adjusted down or disqualified. Log public score + config per submission; never re-submit the same file.

## Suggested 20-submission sequence (1/day; ~36 days to 13 Dec — no wasted days)

- **#1 (today):** Simple hybrid-RAG baseline, exact format. Purpose: validate processing + get a public score.
- **#2–3:** Retrieval A/B, one change at a time: #2 hybrid+reranker, #3 chunk size/metadata variant. Keep the winner.
- **#4–5:** Prompt + answer-format tuning: length cap, same-language enforcement, units/date normalisation. Keep only if offline val improves clearly.
- **#6–8:** Self-consistency within 15 calls (n=3–5 samples + consensus pick) and abstention handling ("not in sources"). #8 = best-of-so-far re-run deterministically.
- **#9–12:** Per-type routing (duplicate → select; else generate), multilingual handling, city-specific fixes (Pristina dates, Rivne financing dates, Skopje OCR quality). One lever per submission.
- **#13–16:** Refinements of the current best only — no new architecture. Submit only if offline val gain exceeds a pre-set threshold (e.g. >0.005 on the local proxy); otherwise hold the slot.
- **#17–19:** Hold. Re-run the best pipeline end-to-end, verify reproducibility, prepare code + documentation for top-100 review. Submit a re-run only if it beats the incumbent public score on the local proxy.
- **#20 (by ~11 Dec, not the last day):** Final best. Then **select the 2 submissions for final evaluation before the close** — do not rely on auto-selection of the 2 best public scores; verify the selected files are the reproducible ones.

## What I could NOT verify — needs the Zindi Data/Info pages

- The scoring metric, submission columns, test size, and public/withheld split for Cities Edition.
- The train/test/sample file list, AllowedDatasets.csv contents, and SourceManifest.csv licences.
- Whether train Q&A pairs exist at all (the routing lesson #6 depends on this).
- Whether the test questions span all three cities or are split per use case, and in which languages.
- Any official starter notebook / baseline score.

No past winners or scores for this edition were invented; the only numeric wins quoted above are from the named prior competitions, with their sources.
