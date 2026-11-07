# Deep research — ≤3B model + RAG pipeline for GenAI for Good (Cities Edition)

Research only. No build, no submission. All numbers below are as reported by the cited source; where two sources report the same model differently (different MMLU variants/protocols), I say so rather than picking one. Date of research: 7 Nov 2026.

## 0. The compliance trap to resolve first — "≤3 billion parameters"

Most models marketed as "3B" are **over 3.00B in total parameters**:

| Model | Total params | Verdict |
|---|---|---|
| Qwen3-4B-Instruct-2507 | **4.02B** (3.6B non-embedding), 262k ctx | **Excluded** |
| Gemma-3-4B-it | **4B**, 128k ctx, 140+ langs | **Excluded** |
| Phi-4-mini-instruct | **3.8B**, 128k ctx, MMLU 67.3 | **Excluded, despite being the best "small reasoner" in many 2026 round-ups** |
| Qwen2.5-3B-Instruct | **3.09B**, 32,768 ctx, 29 langs | Nominally "3B" but **3.09 > 3.00 strictly** |
| Llama-3.2-3B-Instruct | **3.21B**, 128k ctx, 8 official langs | Nominally "3B" but **3.21 > 3.00 strictly** |
| SmolLM3-3B | marketed 3B, 64k native / 128k via YaRN, Apache 2.0 | Exact total param count **not verified**; assume ~3.0–3.1B |
| Qwen3-1.7B-Instruct | **1.7B**, 32k ctx, 119 langs | **Strictly safe** |
| OCC-RAG-1.7B | **1.7B** (Qwen3-1.7B base) — arXiv:2606.00683, 30 May 2026 | **Strictly safe** |
| Gemma-3-1B-it | **1B**, 32k ctx | Strictly safe, but **English-only** — weak for this task |

**Implication:** if the organisers count total parameters literally, only the 1.7B-and-under models are safe. If they count the marketed size class (the usual reading in these challenges — "a 3B model"), Qwen2.5-3B / Llama-3.2-3B / SmolLM3-3B are intended to be eligible. **This must be checked against the Zindi FAQ/discussion before submitting a 3B-class model.** Default to a strictly-safe model unless that check clears the nominal-3B reading.

## 1. Ranked generative-model recommendation

### #1 — Qwen3-1.7B-Instruct (non-thinking mode) — primary pick
- **Why:** strictly ≤3B; best multilingual coverage in the class (**119 languages/dialects**, 36T pretraining tokens — Qwen3 tech report); strong general scores for its size.
- **Evidence (Qwen3 tech report, Table 20, non-thinking):** MMLU-Redux **64.4** (ties Qwen2.5-3B-Instruct 64.4), LiveBench **35.6** vs Qwen2.5-3B 23.8, IFEval strict **68.2** vs 58.2, Arena-Hard **36.9** vs 23.7, GPQA-Diamond 28.6 vs 30.3. Multilingual: MMMLU (14 langs) and Multi-IF favour Qwen3 over Qwen2.5.
- **Caveats:** the *thinking* variant scores higher but thinking tokens count toward the 12,000-token-per-question budget, so **use non-thinking / `enable_thinking=False`** and keep outputs short. 1.7B is weaker than 3B-class on raw knowledge — but for grounded QA the retriever carries the facts; instruction-following and multilingual reading matter more, and Qwen3-1.7B wins those.
- Licence Apache 2.0; GGUF Q4_K_M widely available; runs on both CPU and T4.

### #1 challenger (test head-to-head, promote if it wins) — OCC-RAG-1.7B (`occ-ai/OCC-RAG-1.7B`, GGUF available)
- **Why this is the most task-aligned model found:** a June 2026 paper (arXiv:2606.00683, submitted 30 May 2026) mid-trained Qwen3-1.7B-Base on **3M+ synthetic multi-hop QA examples** specifically for faithful, context-grounded QA, with structured reasoning traces, **literal-quote citations, and calibrated abstention** (`ANSWERABLE` / `UNANSWERABLE`).
- **Evidence (paper abstract + HF GGUF card):** "match or exceed general-purpose models 2–6× their size" on HotpotQA, MuSiQue, TAT-QA, ConFiQA faithfulness and MuSiQue-Un refusal; "best faithfulness across all evaluated scales (up to 32B)"; outperforms its Qwen3 counterparts on all datasets and "exceed Gemma3 (1B and 4B) and SmolLM3-3B on every benchmark". Output format uses `<|query_start|>…<|source_start|><|source_id|>N…` special tokens, five sections (query analysis → source analysis → reasoning → status → answer in `<|answer_start|>…`), greedy decoding (`temp 0`) recommended.
- **Caveats (serious):** single paper, **no independent replication found**; synthetic training data; reasoning capped at 3 hops; context-grounded only (useless without retrieval); language coverage of its training is **unverified** — it inherits Qwen3's tokenizer but its QA training is presumably English. **Must be A/B-tested against Qwen3-1.7B-Instruct on the train QA set before trusting it.** If the test questions are multi-hop or contain unanswerable traps, this could be the winning edge; if they are single-hop factoid, the general model may tie it.

### #2 — Qwen2.5-3B-Instruct — only if nominal-3B is cleared
- **Evidence:** MMLU 65.6 / IFEval 58.3 / GSM8K 79.1 in the Edge/Mobile leaderboard (different protocol from the Qwen3 table — do not compare those MMLU numbers directly with the 64.4 MMLU-Redux above); in the SmolLM3 comparison table (lighteval, zero-shot): IFEval 65.6, Global MMLU 50.54, MMLU-CF 42.93, GSM-Plus 74.1. 29 languages, 32k ctx. Apache 2.0.
- **Why #2 not #1:** 3.09B total (compliance risk above); only 29 languages — Albanian, Ukrainian and Macedonian are **not in its listed set**; older generation than Qwen3.

### #3 — SmolLM3-3B-Instruct — best instruction-following in the 3B class, wrong languages for this task
- **Evidence (HF blog, Jul 2025, lighteval zero-shot):** IFEval **76.7** (best in 3B class, vs Qwen2.5-3B 65.6, Llama-3.1-3B 71.6, Qwen3-1.7B 74.0), Global MMLU 53.5, MMLU-CF 44.13, HellaSwag 76.15, ARC-CF 65.61, BoolQ 78.99, MATH 46.10 (vs Llama-3.2-3B 7.44). With extended thinking: AIME 2025 9.3→36.7, GPQA 35.7→41.7, Global MMLU 53.5→64.1. Apache 2.0, fully open training recipe, 128k via YaRN.
- **Why #3 not higher:** native support is **only 6 languages — English, French, Spanish, German, Italian, Portuguese**. Albanian/Ukrainian/Macedonian are outside that set. Also its thinking mode burns tokens.

### Also evaluated, not recommended
- **Llama-3.2-3B-Instruct:** MMLU 63.4, IFEval 77.4, MGSM 58.2, 128k ctx — respectable, best long-context retention at 128k in the SmolLM3 tables (RULER 128k 71.30). But only **8 official languages**, Dec 2023 cutoff, weakest MATH/GSM8K in class, and 3.21B total. Faithfulness 0.81 on MS MARCO / 0.80–0.81 on TriviaQA single-hop — decent grounding, but the language gap rules it out as primary.
- **Gemma-3-1B-it:** 1B, English-only, MGSM 2.04 — far too weak multilingually.
- **Qwen3-0.6B / TinyLlama-1.1B / Qwen2.5-0.5B:** too weak as sole generator.
- **Ministral-3B:** mentioned in 2025–26 papers as a 3B comparator; no verified scores retrieved that would displace the top 3 — **unverified, do not assume**.
- **Qwen3.5 / Gemma 4 / Ministral 3 (Aug–Sep 2026 round-ups):** exist in secondary sources but no like-for-like benchmarks were retrieved; do not chase without verifying param count, licence and HF availability.

### No direct RAG/QA benchmark exists for this exact task
No source reported HotpotQA-style scores for Qwen2.5-3B vs Qwen3-1.7B vs SmolLM3 on the same protocol. The ranking above is therefore an inference from general + multilingual + instruction-following + faithfulness evidence, and **must be settled empirically on the train set**.

## 2. Retrieval stack (does NOT count as a generative model)

The challenge rules explicitly permit retrieval, OCR, embeddings, reranking, spatial analysis and calculations "provided they do not invoke additional generative models". Embeddings + reranker are therefore legal and are where most of the score will be won or lost.

**Recommended stack:**
1. **Parsing — Docling (layout-aware) first, PyMuPDF fallback.** EMNLP 2025 LobbyMap paper (arXiv:2509.08907): Docling gave the best trade-off between chunk precision and recall on multilingual corporate documents; PyMuPDF with layout chunking was best on non-English in one configuration; semantic chunking optimised retriever effectiveness. Use Docling for PDFs with tables/layout; PyMuPDF for fast plain-text extraction. OCR (Tesseract/ocrmypdf) only for scanned pages — non-generative, allowed.
2. **Chunking — 512–1,024 tokens, 10–20% overlap, sentence-aware, keep headings/tables intact.** Do not split tables mid-row. Add metadata per chunk: source file, page, section, city/use-case. A 2025 Arabic-RAG systematic study found sentence-aware chunking beat all other segmentation methods on RAGAS metrics.
3. **Embedding — Qwen3-Embedding-0.6B (`Qwen/Qwen3-Embedding-0.6B`, Apache 2.0).** MMTEB multilingual mean **64.33** and retrieval **64.64**, vs bge-m3 59.56 / 54.60 (Qwen3 model card, leaderboard snapshot 24 May 2025; corroborated by independent Aug 2026 bake-off notes). 32k context, 100+ languages, MRL dims 32–1024, GGUF/ONNX available. **Requires instruction-prefixed queries and last-token pooling — forgetting the prefix silently degrades quality.** Alternative if Qwen3-Embedding proves awkward: `BAAI/bge-m3` (568M, MIT, 8k ctx, dense+sparse+multi-vector in one model) — older (Feb 2024) and now ~5 points behind on MMTEB, but battle-tested and its sparse output pairs naturally with BM25-style hybrid.
4. **Hybrid retrieval — dense + BM25, reciprocal-rank fusion.** Retrieve top 30–50 by hybrid, not top 5 directly — dense alone misses exact names/numbers/dates (street names, bus route numbers, kWh figures) that BM25 catches, and vice versa for paraphrased questions.
5. **Reranker — Qwen3-Reranker-0.6B** (paired with the embedder; in the Aug 2026 research note's MTEB-R table it scored 65.80 vs bge-reranker-v2-m3 57.03 — secondary source, treat as indicative not definitive). Fallback: `BAAI/bge-reranker-v2-m3` (0.6B, multilingual, well documented). Rerank the 30–50 candidates down to **top 5–8 passages** for the prompt. Cross-encoder reranking is consistently reported as the single biggest precision win after retrieval.
6. **Index — FAISS (CPU) or plain NumPy/Chroma; no hosted vector DB.** Keep the corpus small enough to embed once and reuse for every submission (20 max, 1/day — retrieval quality compounds across submissions).

## 3. Compute: Colab T4 wins decisively; CPU is a fallback only

| Route | Realistic speed (sourced) | Per-question estimate* | Verdict |
|---|---|---|---|
| **VM CPU, llama.cpp Q4_K_M, 3B** | ~**7.8 tok/s** decode on 6-core i7-9750H/16GB (measured, Aug 2026 speculative-decoding study); ~8 tok/s on Raspberry Pi 5; ~5 tok/s quoted for an 8B model on CPU | Prompt 2,000–4,000 tokens + 150–300 output tokens → **~1–2 min/question** (prefill is the uncertain part; CPU prefill for 3B is typically 100–300 tok/s, unverified on this VM) | Feasible for ≤50 questions with per-question checkpointing; **100+ questions = 2–4 h, exceeding the ~2 h VM restart interval unless checkpointed per question** |
| **Colab free T4, llama.cpp/Ollama Q4_K_M, 3B** | **81.5 tok/s** decode for Llama-3.2-3B Q4 on a T4 (Ollama, tinycomputers.io); **84.6–103.7 tok/s** decode, 2,637–5,391 tok/s prefill in brax.me Sep 2026 T4 benchmarks | **~5–15 s/question** including prefill | **Recommended.** 100 questions ≈ 15–25 min + ~10 min setup. 3B Q4_K_M is ~2 GB, Q8_0 ~3.4 GB — both fit trivially in 15 GB T4 VRAM, leaving room for embeddings + reranker on the same GPU |
| Colab T4, HF transformers BF16 1.7–3B | Slower than llama.cpp, ~30–60 tok/s (unverified for this exact setup) | ~10–30 s/question | Acceptable if GGUF path fails; vLLM is better for batching but heavier to set up on a free T4 |

\* Per-question estimates are my arithmetic from the sourced tok/s figures, not measurements — **verify with a 5-question timing run before committing**.

**Colab free-tier limits (multiple 2026 sources, consistent):** 12 h max session, ~90 min idle disconnect, GPU not guaranteed (T4 typical, K80 possible at peak), 15–30 h/week dynamic quota, pre-emptible. Mitigations: write every answered question to Google Drive immediately (checkpoint per question, same resilient-jobs discipline as the VM); keep the notebook executing (a cell that logs progress counts as activity); do retrieval/embedding on the VM beforehand and ship only the compact index + prompts to Colab, so a lost Colab session costs minutes, not the whole pipeline. Kaggle Notebooks (30 GPU-h/week, 2×T4, 12 h session, 20-min idle) is a free backup if Colab refuses a GPU — but the user already has a working Colab login (`knowledgeminees@gmail.com`), so Colab first.

**Hugging Face Inference API — do NOT use.** As of Sep–Oct 2026 checks, free accounts get **no monthly Inference Provider credits** (older $0.10/mo credit is gone or exhausted on first use; PRO $9/mo includes $2). HF's own serverless service serves no chat models. Routing through a third-party provider (Groq/Together/etc. free tier) would also be hard to square with "one generative model", "no paid services", "tools available to everyone", and reproducibility/code-review (the code must reproduce the score). **Self-host the model on Colab or the VM.**

## 4. Prompting for grounded answers (within 15 calls / 12,000 tokens per question)

Budget reality: 12,000 generated tokens per question is generous if outputs are capped at ~400–600 tokens. **Plan for 1 generation call per question; reserve calls 2–3 only for parse failures or a single self-check on low-confidence answers.** Never use thinking/reasoning mode — thinking tokens count as generated tokens.

**Template sketch (adapt to the sample submission's required answer format first — if it wants a bare short answer, drop citations from the submitted text and keep them in a sidecar log):**

```
SYSTEM: You answer ONLY from the CONTEXT below. Do not use prior
knowledge, do not invent numbers, names, dates or sources. Answer
in English. Keep the answer under 150 words unless the question
needs a list. After every factual claim, cite the passage it came
from as [S1], [S2]...

If the context does not contain enough information to answer,
reply exactly: "I don't have enough information in the provided
sources to answer that."

USER:
CONTEXT:
[S1] (source_file, p.3) <chunk text>
[S2] (source_file, p.7) <chunk text>
...
QUESTION: <question>
ANSWER:
```

- **Decoding:** temperature 0, top_p 1, max_tokens 500, stop on end-of-turn. Greedy is the OCC-RAG training default and the right choice for reproducibility.
- **Grounding levers (sourced):** explicit "answer ONLY using the Context" + "If unsure, say 'I don't know'" + per-claim citations + "Do NOT use information outside the Context". One RAG guide reports faithfulness 0.65 → 0.91 from temp 0.7→0.0 plus this prompt (single-source anecdote — treat as directional, not a guarantee). A grounding prompt **without a refusal path is a bug**: it converts missing context into confident fabrication.
- **Post-generation guardrail (no extra generative call):** programmatic check — every `[Sn]` cited must exist; strip any sentence whose numbers/dates do not appear in the cited chunk (regex/NLI check with a small entailment model such as DeBERTa-NLI is a non-generative tool and is allowed); if the check fails, either truncate to the supported claims or emit the refusal string. This is cheaper and more reliable than a second LLM self-check.
- **Numbers/dates:** instruct the model to copy figures verbatim from context, never compute or round unless the question asks for a calculation — and if it does ask, do the arithmetic in Python (non-generative tool) and feed the result back, or verify the model's arithmetic in code.
- **Multilingual docs:** retrieve in the document's language, prompt in English, answer in English (unless the submission format requires the source language). Do not translate documents with a generative model — that would be extra generative calls; if translation is needed, use the retriever's cross-lingual matching (Qwen3 embeddings are cross-lingual) and let the generator read the original text.

## 5. Local evaluation without test ground truth

1. **Build a golden set from train (if train has QA pairs with reference answers):** 20–50 questions is sufficient for rapid iteration (MLflow/RAGAS guidance). Score retrieval (recall@5, recall@10, MRR) against the known source passages, and generation (exact-match / token-F1 for short answers; RAGAS answer-correctness for long ones).
2. **Reference-free (works on test too):** RAGAS **faithfulness** (claims supported / total claims) and **answer relevancy** — both need only {question, contexts, answer}. Practitioner thresholds quoted in 2026 guides: faithfulness ≥0.75–0.80, answer relevancy ≥0.80–0.85, context precision ≥0.70, context recall ≥0.80. **Treat these as calibration starting points, not pass/fail gates** — they come from blog consensus, and the judge model's biases shift them. RAGAS context precision/recall and answer correctness **need** a reference answer, so they only work on train.
3. **Judge-model caution:** RAGAS/DeepEval use an LLM as judge. Using the *same* ≤3B model as judge is circular and weak; using a larger model as judge for *local evaluation only* (not for generating submission answers) is standard practice, but **check it does not breach the "one generative model" rule** — the conservative reading is: judge locally for development, never ship judge outputs, and document it. If in doubt, use the NLI entailment check + human spot-checks instead of an LLM judge.
4. **A/B protocol:** freeze the retriever; run Qwen3-1.7B-Instruct vs OCC-RAG-1.7B vs (if cleared) Qwen2.5-3B on the *same* retrieved contexts and the *same* golden questions; pick the winner on faithfulness + answer correctness + abstention rate on unanswerable probes. Then freeze the generator and A/B the retriever (embedding model, chunk size, k). Never tune both at once.
5. **Deliberately include unanswerable probes** in the golden set (questions whose answer is not in the corpus) — a model that hallucinates an answer there will do the same on withheld test questions.

## 6. Recommended plan (for the parent agent)

1. Confirm test size, scoring metric, answer format and the param-count rule from the Zindi pages/discussion (pending browser task).
2. Build the retrieval index on the VM (Docling → 512–1,024-token chunks → Qwen3-Embedding-0.6B → BM25 hybrid → Qwen3-Reranker-0.6B, top 5–8).
3. Timing run: 5 questions on Colab T4 with Qwen3-1.7B-Instruct Q4_K_M, and the same 5 with OCC-RAG-1.7B. Pick the winner on the train golden set.
4. Full run on Colab, checkpoint per question to Drive + VM. One generation call per question, temp 0, ≤500 output tokens.
5. Validate format against sample submission; submit (1/day — keep later submissions for improved retrieval/model swaps).

## 7. Open uncertainties

- **Test size, metric and answer format** — not yet known (parent's Zindi browser task pending). Everything about time estimates depends on test size.
- **Param-count rule** — literal 3.00B vs marketed "3B class". Decides whether Qwen2.5-3B/SmolLM3/Llama-3.2-3B are even eligible.
- **OCC-RAG-1.7B replication** — one paper (May–Jun 2026), no independent verification found; its reported "best faithfulness up to 32B" needs a local A/B before trust.
- **Question language and document languages** — assumed English questions over English + Albanian/Ukrainian/Macedonian sources; if questions are in local languages, Qwen3-1.7B's 119-language edge grows and SmolLM3's case collapses further.
- **Whether citations are wanted in the submitted answer** — if the scorer does exact-match/embedding-similarity against a short reference, citations in the answer text could *hurt*; keep citations in a sidecar and submit clean answers unless the format asks for them.
- **Colab GPU availability at run time** — not guaranteed; Kaggle is the backup, CPU + per-question checkpointing the last resort.
- **Qwen3.5 / Gemma 4 (Sep 2026 round-ups)** — newer small models exist but were not verified for param count, licence or benchmarks in this research; do not switch without the same checks applied above.

### Key sources
- Qwen3 tech report, arXiv:2505.09388 (May 2025) — Qwen3-1.7B/0.6B/4B tables.
- HF SmolLM3 blog, Jul 2025 + lighteval tables — SmolLM3 vs Qwen2.5-3B vs Llama-3.2-3B vs Qwen3-1.7B/4B.
- Meta Llama-3.2 model card (Sep 2024) — 3.21B, 128k, 8 langs, MMLU 63.4 / IFEval 77.4.
- Microsoft Phi-4-mini-instruct HF card (Feb 2025) — 3.8B, 128k, MMLU 67.3.
- HF Gemma-3 blog (Mar 2025) — 1B English-only / 4B 140 langs, 128k.
- Qwen/Qwen2.5-3B-Instruct HF card — 3.09B, 32,768 ctx, 29 langs.
- Qwen/Qwen3-4B-Instruct-2507 HF card — 4.02B total, 3.6B non-embedding, 262k ctx.
- OCC-RAG, arXiv:2606.00683 (30 May 2026) + `occ-ai/OCC-RAG-1.7B-GGUF` card.
- Qwen3-Embedding-0.6B model card (MMTEB 64.33/64.64 vs bge-m3 59.56/54.60); LobbyMap/EMNLP 2025, arXiv:2509.08907 (Docling/PyMuPDF, bge-m3 vs Nomic vs Qwen3-Embedding, bge-reranker-v2-m3).
- llama.cpp CPU: 3B Q4_K_M 7.76 tok/s (Aug 2026 study); T4: Llama-3.2-3B Q4 81.5 tok/s (tinycomputers.io), 84.6–103.7 tok/s decode (brax.me, Sep 2026).
- Colab free limits 2026: 12 h session, 90-min idle, T4 typical, dynamic quota (ThunderCompute/Spheron, Oct–Nov 2026).
- HF Inference Providers free tier 2026: no free credits on free accounts, PRO $9/mo incl. $2 (aifree.dev, Oct 2026; klymentiev.com, Sep 2026).
- RAGAS: faithfulness + answer relevancy are reference-free; context precision/recall need references (RAGAS docs via dev.to/MLflow/Particula, Oct 2026; thresholds are practitioner consensus, not official).
