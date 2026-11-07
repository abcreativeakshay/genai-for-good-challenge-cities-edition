"""Build per-question contexts for the GenAI for Good Challenge.

For each Test (or Train) row, read ONLY its listed Source_IDs,
extract them with src.extract, and write one JSONL line per question::

    {"ID": ..., "Domain": ..., "Question": ..., "Context": ...,
     "excerpts": [{"source_id": ..., "file": ..., "locator_hint": ...,
                   "text": ...}, ...]}

Budget: total excerpt text per question stays under ~12,000
characters (MAX_CONTEXT_CHARS). Sources are taken in priority order:

  0. Analysis / SourceGuide / ReadingGuide files (they summarise a
     case and point at the exact locations to use)
  1. small .txt / .md
  2. .json / .csv
  3. .geojson
  4. .xlsx / .ods
  5. .pdf / .html / .xml / .xsd / .gml
  6. metadata-only sources (a one-line note; no values)

Within a priority group, smaller files go first so more sources fit.
A source whose text does not fit in the remaining budget is skipped
(and logged) rather than squeezed in as a useless fragment, unless
at least 500 characters remain, in which case it is truncated with
a "[truncated]" marker.

Deterministic and resumable: sources are extracted once and cached
in memory; output is written incrementally and the run can be
re-started -- already-written question IDs are skipped. No
randomness anywhere (seed fixed, ordering fixed by the input file
and the priority rules).

Outputs (all under data/, gitignored -- never committed):
  data/contexts.jsonl      one line per question
  data/contexts_log.json   per-question char counts, skipped/truncated sources

Usage:
    python -m src.build_contexts [--input data/Test.csv] [--train]
    python -m src.build_contexts --input data/Train.csv --output data/train_contexts.jsonl --log data/train_contexts_log.json
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.extract import extract_source  # noqa: E402

MAX_CONTEXT_CHARS = 12_000
MIN_FRAGMENT_CHARS = 500
# Per-source extraction cap: a single source never contributes more
# than this many characters before the per-question budget is applied.
PER_SOURCE_MAX_CHARS = 20_000
# Per-excerpt cap inside a question: with ~5.5 sources listed per
# question and a 12,000-char budget, letting one source take the
# whole budget starves the rest (the first build averaged only 3.5
# of 5.5 sources present). The cap below is adaptive (see the loop):
# each source may take up to 8,000 chars, but 500 chars are always
# reserved for each source still to come, so every listed source
# gets at least a fragment.


PER_EXCERPT_MAX_CHARS = 8_000
RESERVE_PER_REMAINING_SOURCE = 500


def source_priority(original_file: str, suffix: str) -> int:
    name = original_file.lower()
    if "analysis" in name or "sourceguide" in name or "readingguide" in name:
        return 0
    if suffix in (".txt", ".md"):
        return 1
    if suffix in (".json", ".csv"):
        return 2
    if suffix == ".geojson":
        return 3
    if suffix in (".xlsx", ".ods"):
        return 4
    if suffix in (".pdf", ".html", ".xml", ".xsd", ".gml", ".docx", ".zip"):
        return 5
    return 6


def parse_source_ids(raw) -> list[str]:
    """Source_IDs column is a JSON-ish list string, e.g. '["SRC_.."]'."""
    if isinstance(raw, list):
        return [str(x) for x in raw]
    try:
        return [str(x) for x in json.loads(raw)]
    except Exception:
        return [str(x) for x in ast.literal_eval(raw)]


def build_contexts(input_csv: Path, manifest_csv: Path, sources_dir: Path,
                   output_path: Path, log_path: Path) -> dict:
    questions = pd.read_csv(input_csv, keep_default_na=False)
    manifest = pd.read_csv(manifest_csv, keep_default_na=False)
    by_id = {row["Source_ID"]: row.to_dict() for _, row in manifest.iterrows()}

    # Already-written IDs (resume): read the existing output, if any.
    done: dict[str, dict] = {}
    if output_path.exists():
        with open(output_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    done[rec["ID"]] = rec

    cache: dict[str, dict] = {}  # source_id -> extraction result
    log: dict[str, dict] = {}
    if log_path.exists():
        try:
            log = json.loads(log_path.read_text(encoding="utf-8"))
        except Exception:
            log = {}

    stats = {"questions_written": 0, "questions_skipped_done": 0,
             "total_chars": 0, "skipped_sources": 0, "truncated_sources": 0}

    with open(output_path, "a", encoding="utf-8") as out:
        for _, row in questions.iterrows():
            qid = str(row["ID"])
            if qid in done:
                stats["questions_skipped_done"] += 1
                continue
            source_ids = parse_source_ids(row["Source_IDs"])

            # Order sources by priority, then by file size (small first).
            def sort_key(sid: str):
                info = by_id.get(sid, {})
                original = str(info.get("Original_File", ""))
                suffix = Path(original).suffix.lower()
                size = int(info.get("Bytes") or 0)
                return (source_priority(original, suffix), size)

            ordered = sorted(source_ids, key=sort_key)

            excerpts: list[dict] = []
            used_chars = 0
            skipped: list[dict] = []
            truncated: list[str] = []
            for j, sid in enumerate(ordered):
                if sid not in cache:
                    info = by_id.get(sid)
                    if info is None:
                        skipped.append({"source_id": sid,
                                        "reason": "not in SourceManifest.csv"})
                        continue
                    cache[sid] = extract_source(sid, info, sources_dir,
                                                max_chars=PER_SOURCE_MAX_CHARS)
                ext = cache[sid]
                remaining = MAX_CONTEXT_CHARS - used_chars
                if remaining < MIN_FRAGMENT_CHARS:
                    skipped.append({"source_id": sid, "file": ext["file"],
                                    "reason": f"budget exhausted ({max(remaining, 0)} chars left)"})
                    continue
                text = ext["text"]
                # Adaptive cap: up to 8,000 chars, but always reserve
                # 500 chars for each source still to come, so every
                # listed source gets at least a fragment. Reserve room
                # for the truncation marker so the total stays
                # <= MAX_CONTEXT_CHARS.
                sources_left = len(ordered) - j - 1
                allowed = min(len(text), remaining, PER_EXCERPT_MAX_CHARS,
                              max(MIN_FRAGMENT_CHARS,
                                  remaining - RESERVE_PER_REMAINING_SOURCE * sources_left))
                if len(text) > allowed:
                    marker = (f"\n[truncated: showing {allowed} of {len(text)} "
                              "characters for this source]")
                    allowed = max(0, min(allowed, remaining - len(marker)))
                    text = text[:allowed] + marker
                    truncated.append(sid)
                excerpts.append({
                    "source_id": sid,
                    "file": ext["file"],
                    "locator_hint": ext["file"],
                    "text": text,
                })
                used_chars += len(text)

            record = {
                "ID": qid,
                "Domain": str(row.get("Domain", "")),
                "Question": str(row.get("Question", "")),
                "Context": str(row.get("Context", "")),
                "excerpts": excerpts,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            log[qid] = {
                "domain": record["Domain"],
                "n_sources_listed": len(source_ids),
                "n_excerpts": len(excerpts),
                "total_chars": used_chars,
                "skipped": skipped,
                "truncated": truncated,
            }
            stats["questions_written"] += 1
            stats["total_chars"] += used_chars
            stats["skipped_sources"] += len(skipped)
            stats["truncated_sources"] += len(truncated)

    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    stats["total_questions_in_log"] = len(log)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/Test.csv",
                        help="questions CSV (default: data/Test.csv)")
    parser.add_argument("--manifest", default="data/SourceManifest.csv")
    parser.add_argument("--sources-dir", default="data",
                        help="challenge data folder (manifest File paths are relative to it)")
    parser.add_argument("--output", default="data/contexts.jsonl")
    parser.add_argument("--log", default="data/contexts_log.json")
    args = parser.parse_args()

    stats = build_contexts(Path(args.input), Path(args.manifest),
                           Path(args.sources_dir), Path(args.output),
                           Path(args.log))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
