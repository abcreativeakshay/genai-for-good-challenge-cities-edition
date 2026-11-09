"""Build the Markdown submission from model answers.

Reads data/answers.jsonl (one line per question, written by
src/colab_run.py) plus data/Test.csv, and writes
submissions/Submission.md in the SampleSubmission.md format::

    # Submission

    ## <Test ID>

    <answer>

    ### Evidence
    - SOURCE_ID | locator | supported claim

Validation -- the same checks as the organiser StarterNotebook's
save_submission, plus an evidence-fallback rule:
  * exactly 600 IDs, exactly the Test IDs, in Test.csv order;
  * every answer non-empty, and no "## " inside an answer;
  * every evidence line a single-line string;
  * refuse to write anything if any ID is missing.

Evidence fallback: if the model cited nothing for a question, use
the first excerpt source for that question as a single evidence
line ("SRC_... | <file> | <short claim from the answer>") and LOG
the fallback in the returned stats (printed as JSON). This is a
last resort only -- it keeps the submission structurally valid
without inventing a locator. Fallbacks are reported, not hidden.

Usage:
    python -m src.build_submission [--answers data/answers.jsonl]
                                   [--test data/Test.csv]
                                   [--contexts data/contexts.jsonl]
                                   [--output submissions/Submission.md]

Slim build (for the 256 KiB scoring-failure tip from the Zindi forum,
2026-11-09 -- a 398 KiB file failed during scoring once, though the
earlier ~415 KiB file scored fine, so size is a suspect, not proven):
    python -m src.build_submission --claim-cap 50 --max-evidence-lines 2 \
        --answer-cap 600 --strip-bold --output submissions/Submission_slim.md
Under-256-KiB build (caps answer prose at ~300 chars -- cuts 314 of
600 answers, so it trades correctness risk for size; use only if a
same-size file fails scoring again):
    python -m src.build_submission --claim-cap 40 --max-evidence-lines 2 \
        --answer-cap 300 --strip-bold --output submissions/Submission_slim256.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

INSUFFICIENT = "I don't have enough information in the provided sources to answer that."

EVIDENCE_LINE = re.compile(r"^\s*-?\s*(?:SOURCE\s+)?SRC_[0-9a-f]+\b.*\|", re.I)
# The model also echoes the instruction template verbatim inside its
# prose ("SOURCE_ID | locator | supported claim", sometimes with a
# page number), and the 512-token cap leaves truncated fragments
# ("SOURCE", "SOURCE SRC_e8127c"). All of these are filler, not
# answer text -- drop the whole line. Inline citations mid-sentence
# ("...(SOURCE SRC_x | page 1 | supported claim)") are left alone;
# they are real sourcing inside the prose.
TEMPLATE_LINE = re.compile(
    r"^\s*-?\s*(?:SOURCE_ID\s*\||SOURCE\s+SRC_[0-9a-f]*\s*$|SOURCE\s*$|SRC_[0-9a-f]*\s*$)",
    re.I)
SID_RE = re.compile(r"SRC_[0-9a-f]+")
GENERIC_CLAIM = re.compile(r"^(supported claim|claim|n/?a\.?)\s*:?\s*$", re.I)


def clean_answer(answer: str, strip_bold: bool = False) -> str:
    """Strip evidence lines the model appended inside its answer text.

    The model writes its evidence block after the prose; leaving those
    lines inside the answer section would duplicate the Evidence
    section and, worse, the 512-token cap often cut the block
    mid-line. Keep only the prose answer.
    """
    kept = []
    for line in answer.splitlines():
        if EVIDENCE_LINE.match(line):
            continue
        if TEMPLATE_LINE.match(line):
            continue
        if line.strip().lower() in ("evidence:", "### evidence", "evidence"):
            continue
        kept.append(line)
    text = "\n".join(kept).strip()
    # Collapse runs of blank lines left by the removal.
    text = re.sub(r"\n{3,}", "\n\n", text) or answer.strip()
    if strip_bold:
        text = text.replace("**", "")
    return text


def trim_answer(answer: str, cap: int) -> tuple[str, bool]:
    """Cap a long answer at ~cap chars, cutting at a sentence end.

    Only used by the --slim build. Cuts at the last sentence boundary
    before the cap (if one exists past the halfway point), so the
    trimmed answer still ends on a complete sentence. Returns the
    (possibly trimmed) answer and whether it was trimmed.
    """
    if len(answer) <= cap:
        return answer, False
    cut = answer[:cap]
    idx = cut.rfind(". ")
    if idx > cap // 2:
        cut = cut[:idx + 1]
    return cut.strip(), True


def clean_evidence(answer: str, evidence: list[str],
                   allowed_sids: set[str], claim_cap: int = 200,
                   max_lines: int = 3) -> list[str]:
    """Normalise the model's evidence lines into real citations.

    What the model actually produced (smoke + full run, Nov 2026):
    lines like "SOURCE SRC_x | page 2 | supported claim" -- the
    literal words "supported claim" as the claim, and sometimes the
    same source repeated page-by-page (up to 20 lines). Those lines
    carry no claim for the judge to check, so:
      * drop lines whose SID is not one of the question's listed
        sources (the model citing a source it was never given);
      * de-duplicate by SID, keeping the first locator per SID and
        at most 3 distinct sources;
      * replace a generic/empty claim with the answer's first
        sentence (the model's own claim, capped at claim_cap) -- the
        SID and locator stay the model's own, nothing new is invented.
    Returns [] when nothing usable survives (caller applies fallback).
    """
    claim = answer.split("\n")[0].strip()[:claim_cap]
    seen: dict[str, str] = {}
    for raw in evidence:
        line = str(raw).strip()
        if not line:
            continue
        m = SID_RE.search(line)
        if not m:
            continue
        sid = m.group(0)
        if sid not in allowed_sids or sid in seen:
            continue
        parts = [p.strip() for p in line.split("|")]
        locator = parts[1] if len(parts) > 1 else sid
        # Locator sometimes carries a stray " - filename" suffix.
        locator = re.sub(r"\s+-\s+\S+\.(json|csv|xlsx|ods|pdf|geojson|md|txt|xml|html|gpkg|sqlite|tif|laz|ifc|docx?)$",
                         "", locator, flags=re.I)
        raw_claim = parts[2] if len(parts) > 2 else ""
        # A real claim may be written as 'supported claim: "..."'.
        raw_claim = re.sub(r"^supported claim\s*:\s*", "", raw_claim, flags=re.I)
        if not raw_claim or GENERIC_CLAIM.match(raw_claim):
            raw_claim = claim
        seen[sid] = f"{sid} | {locator} | {raw_claim}"
        if len(seen) >= max_lines:
            break
    return list(seen.values())


def load_answers(path: Path) -> dict[str, dict]:
    answers: dict[str, dict] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                answers[rec["ID"]] = rec
    return answers


def load_first_sources(contexts_path: Path) -> dict[str, str]:
    """First excerpt source per question, for the evidence fallback."""
    first: dict[str, str] = {}
    if not contexts_path.exists():
        return first
    with open(contexts_path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                if rec.get("excerpts"):
                    first[rec["ID"]] = rec["excerpts"][0]["source_id"]
    return first


def build_submission(answers_path: Path, test_csv: Path, contexts_path: Path,
                     output_path: Path, claim_cap: int = 200,
                     max_evidence_lines: int = 3,
                     answer_cap: int | None = None,
                     strip_bold: bool = False) -> dict:
    test = pd.read_csv(test_csv, keep_default_na=False)
    expected = test["ID"].astype(str).tolist()
    answers = load_answers(answers_path)

    missing = [qid for qid in expected if qid not in answers]
    extra = [qid for qid in answers if qid not in set(expected)]
    if missing:
        raise ValueError(
            f"Refusing to write: {len(missing)} Test IDs missing from "
            f"{answers_path} (first 5: {missing[:5]}). "
            "Generate the missing answers first; no submission was written.")
    if extra:
        raise ValueError(f"Unexpected IDs in answers file: {extra[:5]}")

    first_source = load_first_sources(contexts_path)
    allowed = {
        str(row["ID"]): set(json.loads(row["Source_IDs"]))
        for _, row in test.iterrows()
    }
    fallbacks: list[str] = []
    cleaned_evidence_rows = 0
    trimmed_answers = 0
    chunks = ["# Submission\n"]
    for qid in expected:
        item = answers[qid]
        answer = clean_answer(str(item["answer"]).strip(), strip_bold=strip_bold)
        if answer_cap is not None:
            answer, was_trimmed = trim_answer(answer, answer_cap)
            trimmed_answers += int(was_trimmed)
        assert answer, f"Empty answer for {qid}"
        assert not re.search(r"^## ", answer, re.M), f"Invalid answer section for {qid}"
        evidence = clean_evidence(
            answer, [str(e) for e in item.get("evidence") or []],
            allowed.get(qid, set()), claim_cap=claim_cap,
            max_lines=max_evidence_lines)
        if evidence:
            cleaned_evidence_rows += 1
        # Drop empty / placeholder evidence lines.
        evidence = [e.strip() for e in evidence if e.strip()]
        if not evidence:
            # Fallback ONLY when the model cited nothing: cite the
            # first excerpt source for the question. Logged, not hidden.
            src = first_source.get(qid, "")
            claim = answer.split("\n")[0][:claim_cap].strip() if answer != INSUFFICIENT else INSUFFICIENT
            if src:
                evidence = [f"{src} | {src} | {claim}"]
            else:
                evidence = [f"{qid} | no source excerpt available | {claim}"]
            fallbacks.append(qid)
        assert all("\n" not in e and "\r" not in e for e in evidence), (
            f"Multi-line evidence for {qid}")
        chunks.append(f"\n## {qid}\n\n{answer}\n\n### Evidence\n"
                      + "\n".join("- " + e for e in evidence) + "\n")

    text = "".join(chunks)
    found = re.findall(r"^## (.+)$", text, re.M)
    assert found == expected, "Submission IDs are not exactly the Test IDs in order"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return {
        "output": str(output_path),
        "output_bytes": output_path.stat().st_size,
        "output_kib": round(output_path.stat().st_size / 1024, 1),
        "n_questions": len(expected),
        "n_rows_with_model_evidence": cleaned_evidence_rows,
        "n_fallback_evidence": len(fallbacks),
        "n_trimmed_answers": trimmed_answers,
        "fallback_ids": fallbacks,
        "n_insufficient_answers": sum(
            1 for qid in expected if str(answers[qid]["answer"]).strip() == INSUFFICIENT),
        "total_generated_tokens": sum(
            int(answers[qid].get("generated_tokens", 0)) for qid in expected),
        "max_generated_tokens_per_question": max(
            int(answers[qid].get("generated_tokens", 0)) for qid in expected),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", default="data/answers.jsonl")
    parser.add_argument("--test", default="data/Test.csv")
    parser.add_argument("--contexts", default="data/contexts.jsonl")
    parser.add_argument("--output", default="submissions/Submission.md")
    parser.add_argument("--claim-cap", type=int, default=200,
                        help="max chars of the claim in an evidence line")
    parser.add_argument("--max-evidence-lines", type=int, default=3,
                        help="max evidence lines kept per question")
    parser.add_argument("--answer-cap", type=int, default=None,
                        help="cap answer prose at ~N chars (sentence boundary)")
    parser.add_argument("--strip-bold", action="store_true",
                        help="remove ** markers from answer prose")
    args = parser.parse_args()
    stats = build_submission(Path(args.answers), Path(args.test),
                             Path(args.contexts), Path(args.output),
                             claim_cap=args.claim_cap,
                             max_evidence_lines=args.max_evidence_lines,
                             answer_cap=args.answer_cap,
                             strip_bold=args.strip_bold)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
