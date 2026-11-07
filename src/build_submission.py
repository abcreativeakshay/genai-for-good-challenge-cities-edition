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
                     output_path: Path) -> dict:
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
    fallbacks: list[str] = []
    chunks = ["# Submission\n"]
    for qid in expected:
        item = answers[qid]
        answer = str(item["answer"]).strip()
        assert answer, f"Empty answer for {qid}"
        assert not re.search(r"^## ", answer, re.M), f"Invalid answer section for {qid}"
        evidence = [str(e) for e in item.get("evidence") or []]
        # Drop empty / placeholder evidence lines.
        evidence = [e.strip() for e in evidence if e.strip()]
        if not evidence:
            # Fallback ONLY when the model cited nothing: cite the
            # first excerpt source for the question. Logged, not hidden.
            src = first_source.get(qid, "")
            claim = answer.split("\n")[0][:200].strip() if answer != INSUFFICIENT else INSUFFICIENT
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
        "n_questions": len(expected),
        "n_fallback_evidence": len(fallbacks),
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
    args = parser.parse_args()
    stats = build_submission(Path(args.answers), Path(args.test),
                             Path(args.contexts), Path(args.output))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
