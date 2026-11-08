"""Source extractors for the GenAI for Good Challenge: Cities Edition.

Inference-only extraction: every function here READS supplied source
files and returns plain text with locators. Nothing here trains or
fine-tunes any model, and nothing writes back into a source file.

Each extractor returns a dict::

    {
        "source_id": "SRC_...",
        "file": "<original filename from the manifest>",
        "suffix": ".csv",
        "size_bytes": 12345,
        "text": "<plain text with locators, possibly truncated>",
        "truncated": True/False,
        "note": "<short note, e.g. why a format is metadata-only>",
    }

``text`` is capped by ``max_chars`` (default 8000) so a single huge
source (69 MB GTFS shapes.txt, 136 MB current.csv, ...) cannot blow
up a question's context budget. Truncation is always marked with a
``[truncated ...]`` marker and the total row/line/feature counts are
reported where they can be counted cheaply.

Formats NOT yet implemented natively (metadata only, never faked):
.gpkg / .sqlite (table names only for SQLite), .tif, .laz, .ifc, .jpg.
See src/README.md.
"""

from __future__ import annotations

import collections
import csv
import html
import json
import re
import sqlite3
import zipfile
from pathlib import Path

DEFAULT_MAX_CHARS = 8000

# Formats whose values we do not (yet) read natively. We report size
# and a note only -- never invented values.
METADATA_ONLY = {".gpkg", ".tif", ".laz", ".ifc", ".jpg", ".jpeg", ".png"}

# --- Deterministic computation helpers (submission #2) -----------------
# The model cannot count/sum/measure reliably from truncated head
# rows, so extractors now prepend a short COMPUTED summary block
# (full-file counts, blanks, numeric stats, areas) that is verified
# against the source file itself. Nothing here is guessed: every
# number comes from reading the whole file.
#
# Focus tokens: build_contexts passes alphanumeric tokens from the
# question + case context (e.g. "1912/A1", "TPO453"). Rows / items /
# features containing a focus token are shown first, so a question
# about row 900 of a CSV or feature 500 of a GeoJSON is not answered
# from an unrelated head sample.


# Words that appear in question phrasing / units / generic field
# names rather than in data values. Without them, tokens like "the",
# "for" or "zone" match nearly every row (e.g. every Liverpool zones
# feature has a "tree-preservation-zone-type" field), so focus
# ranking degrades to file order. Identifiers (1912/A1, TPO453),
# names, streets and species survive the filter.
FOCUS_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "by",
    "for", "from", "with", "per", "give", "given", "what", "which",
    "how", "many", "much", "when", "where", "that", "this", "there",
    "their", "about", "into", "over", "under", "between", "after",
    "before", "during", "should", "would", "could", "answer",
    "question", "following", "using", "use", "used", "list", "name",
    "number", "total", "value", "values", "data", "file", "source",
    "sources", "table", "calculate", "computed", "average", "mean",
    "sum", "count", "rounded", "decimal", "places", "early",
    "estimate", "budget", "based", "area", "zone", "hectares",
    "hectare", "metres", "meters", "metre", "meter", "square",
    "nearest", "closest", "largest", "smallest", "highest", "lowest",
}


def focus_tokens(*texts: str) -> set[str]:
    """Alphanumeric tokens (len >= 3) from question/context text,
    minus question-phrasing stopwords."""
    toks: set[str] = set()
    for text in texts:
        if not text:
            continue
        for m in re.finditer(r"[A-Za-z0-9][A-Za-z0-9_\-/\.]{2,}", text):
            tok = m.group(0).lower().rstrip(".-")
            if tok and tok not in FOCUS_STOPWORDS:
                toks.add(tok)
    return toks


def _focus_score(text: str, focus: set[str]) -> int:
    """How many DISTINCT focus tokens appear in text (substring).

    Count-based, not boolean: a token like "zone" that matches
    every row adds the same +1 everywhere and cannot swamp the
    ranking, while a rare identifier (1912/A1) lifts exactly its
    row to the top.
    """
    if not focus:
        return 0
    low = text.lower()
    return sum(1 for tok in focus if tok in low)


def _matches_focus(text: str, focus: set[str]) -> bool:
    return _focus_score(text, focus) > 0


def _idf_rank(texts: list[str], focus: set[str]) -> list[int]:
    """Indices of ``texts`` ranked by focus relevance.

    Each token is weighted by 1 / (number of texts it matches), so a
    rare identifier (1912/A1, matching 1 of 727 features) outweighs
    a common word (Liverpool, matching 38). Ties keep file order.
    """
    if not focus:
        return list(range(len(texts)))
    lows = [t.lower() for t in texts]
    df = {}
    for tok in focus:
        df[tok] = sum(1 for low in lows if tok in low)
    scores = []
    for low in lows:
        s = sum(1.0 / df[tok] for tok in focus if tok in low and df[tok] > 0)
        scores.append(s)
    return sorted(range(len(texts)), key=lambda i: (-scores[i], i))


def _fmt_num(value: float) -> str:
    return format(value, ".6g")


def _column_stats_from_acc(acc: dict) -> str:
    """One-line stats from a streaming column accumulator."""
    parts = [f"n={acc['n']} blank={acc['blank']} distinct={acc['distinct']}"]
    if acc["numeric"] == acc["n"] - acc["blank"] and acc["numeric"] > 0:
        parts.append(
            f"numeric min={_fmt_num(acc['min'])} max={_fmt_num(acc['max'])} "
            f"sum={_fmt_num(acc['sum'])} mean={_fmt_num(acc['sum'] / acc['numeric'])}")
    elif acc["top"] and acc["distinct"] <= 50:
        parts.append("top=" + ", ".join(
            f"{str(k)[:80]}({c})" for k, c in acc["top"].most_common(5)))
    return "; ".join(parts)


def _new_col_acc() -> dict:
    return {"n": 0, "blank": 0, "distinct": 0, "seen": set(), "numeric": 0,
            "min": None, "max": None, "sum": 0.0, "top": collections.Counter()}


def _acc_value(acc: dict, value: str) -> None:
    acc["n"] += 1
    if value == "" or value is None:
        acc["blank"] += 1
        return
    if len(acc["seen"]) < 2000:
        if value not in acc["seen"]:
            acc["seen"].add(value)
            acc["distinct"] += 1
    elif value not in acc["seen"]:
        acc["distinct"] = 2001  # capped: "2000+ distinct"
    acc["top"][value] += 1
    if len(acc["top"]) > 100:
        acc["top"] = collections.Counter(dict(acc["top"].most_common(20)))
    try:
        num = float(value)
    except (TypeError, ValueError):
        return
    acc["numeric"] += 1
    acc["sum"] += num
    acc["min"] = num if acc["min"] is None else min(acc["min"], num)
    acc["max"] = num if acc["max"] is None else max(acc["max"], num)


def _projector_for_coords(xs: list[float], ys: list[float]):
    """Return a lon/lat -> metres projector, or None for planar data.

    Lon/lat data (all |x|<=180, |y|<=90) is projected to the UTM zone
    of the mean longitude (pyproj, EPSG:326xx/327xx). Already-projected
    data (e.g. British National Grid metres) is used as-is. Falls back
    to a local equirectangular approximation if pyproj is unavailable.
    """
    if not xs or not ys:
        return None
    if max(abs(x) for x in xs) <= 180 and max(abs(y) for y in ys) <= 90:
        lon0 = sum(xs) / len(xs)
        lat0 = sum(ys) / len(ys)
        zone = int((lon0 + 180) // 6) + 1
        epsg = (32600 if lat0 >= 0 else 32700) + zone
        try:
            from pyproj import Transformer
            tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
            return lambda x, y: tr.transform(x, y)
        except Exception:
            import math
            kx = 111412.84 * math.cos(math.radians(lat0)) - 93.5 * math.cos(3 * math.radians(lat0))
            ky = 111132.92 - 559.82 * math.cos(2 * math.radians(lat0)) + 1.175 * math.cos(4 * math.radians(lat0))
            return lambda x, y: (x * kx, y * ky)
    return None


def _ring_area_m2(ring, project) -> float:
    """Shoelace area of one ring, in m^2 (projector maps to metres).

    Points may be 3D ([x, y, z]) -- only x/y are used.
    """
    import numpy as np
    pts = [project(pt[0], pt[1]) if project else (pt[0], pt[1]) for pt in ring]
    a = np.array(pts, dtype=float)
    if len(a) < 3:
        return 0.0
    return 0.5 * abs(float(np.dot(a[:, 0], np.roll(a[:, 1], -1))
                           - np.dot(a[:, 1], np.roll(a[:, 0], -1))))


def _geom_area_m2(geom, project) -> float:
    t = geom.get("type") if isinstance(geom, dict) else None
    if t == "Polygon":
        coords = geom.get("coordinates") or []
        if not coords:
            return 0.0
        area = _ring_area_m2(coords[0], project)
        for hole in coords[1:]:
            area -= _ring_area_m2(hole, project)
        return max(area, 0.0)
    if t == "MultiPolygon":
        return sum(_geom_area_m2({"type": "Polygon", "coordinates": p}, project)
                   for p in (geom.get("coordinates") or []))
    return 0.0


def _parse_wkb_geom(blob: bytes):
    """Minimal WKB reader (Point / LineString / Polygon / Multi*).

    Returns (geom_type, coords_or_rings) or None if unparseable.
    Only the geometry types in these GeoPackages are handled.
    """
    import struct
    if not blob or len(blob) < 9:
        return None
    # GeoPackage blobs start with the "GP" magic + a header: byte 2
    # version, byte 3 flags (bits 0-2 = envelope indicator), 4-byte
    # SRS id, then the envelope (0/32/48/48/64 bytes). Standard WKB
    # follows the header. Parse it properly instead of guessing.
    start = 0
    if blob[:2] == b"GP" and len(blob) >= 8:
        # Flags byte: bit 0 = binary type, bits 1-3 = envelope
        # indicator (0 = none, 1 = xy, 2 = xyz, 3 = xym, 4 = xyzm).
        indicator = (blob[3] >> 1) & 0x07
        env_bytes = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}.get(indicator)
        if env_bytes is None:
            return None
        start = 8 + env_bytes
    else:
        # Plain WKB (no GPKG header): scan for a plausible endian
        # byte (0/1) followed by a known type (1..7).
        for i in range(min(len(blob) - 5, 64)):
            if blob[i] in (0, 1) and 1 <= int.from_bytes(blob[i + 1:i + 5], "little" if blob[i] == 1 else "big") <= 7:
                start = i
                break
        else:
            return None
    if start + 5 > len(blob):
        return None
    bo = "<" if blob[start] == 1 else ">"
    gtype = struct.unpack_from(bo + "I", blob, start + 1)[0]
    off = start + 5

    def rd_point():
        nonlocal off
        x, y = struct.unpack_from(bo + "dd", blob, off)
        off += 16
        return (x, y)

    def rd_ring():
        nonlocal off
        n = struct.unpack_from(bo + "I", blob, off)[0]
        off += 4
        return [rd_point() for _ in range(n)]

    if gtype == 1:
        return ("Point", rd_point())
    if gtype == 2:
        n = struct.unpack_from(bo + "I", blob, off)[0]
        off += 4
        return ("LineString", [rd_point() for _ in range(n)])
    if gtype == 3:
        n = struct.unpack_from(bo + "I", blob, off)[0]
        off += 4
        return ("Polygon", [rd_ring() for _ in range(n)])
    if gtype in (4, 5, 6):
        n = struct.unpack_from(bo + "I", blob, off)[0]
        off += 4
        parts = []
        for _ in range(n):
            sub = _parse_wkb_geom(blob[off:])
            if sub is None:
                break
            parts.append(sub)
            break  # nested offsets are unreliable; report type only
        return (f"Multi{gtype}", parts)
    return None



def col_letter(idx: int) -> str:
    """0-based column index -> spreadsheet column letters (0->A, 26->AA)."""
    letters = ""
    n = idx + 1
    while n:
        n, rem = divmod(n - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def cell_ref(col_idx: int, row_idx0: int) -> str:
    """Spreadsheet cell reference, e.g. col 26, row 8 -> 'AA9'."""
    return f"{col_letter(col_idx)}{row_idx0 + 1}"


def _truncate(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    cut = text[:max_chars]
    return cut + f"\n[truncated: showing {max_chars} of {len(text)} characters]", True


def _result(source_id: str, original_file: str, path: Path,
            text: str, truncated: bool, note: str = "") -> dict:
    return {
        "source_id": source_id,
        "file": original_file,
        "suffix": path.suffix.lower(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "text": text,
        "truncated": truncated,
        "note": note,
    }


def _read_text_head(path: Path, max_chars: int) -> tuple[str, bool, int]:
    """Read the head of a text file; return (text, truncated, total_lines).

    Total lines are counted by streaming the whole file once. For the
    very large text sources this is a single sequential pass and is
    only done once per source (build_contexts caches extractions).
    """
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        head = fh.read(max_chars + 1)
    truncated = len(head) > max_chars
    if truncated:
        head = head[:max_chars]
    total_lines = 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for _ in fh:
            total_lines += 1
    return head, truncated, total_lines


def _extract_txt_md(path: Path, max_chars: int) -> tuple[str, bool]:
    text, truncated, total_lines = _read_text_head(path, max_chars)
    header = f"[text file: {total_lines} lines total]"
    out, trunc2 = _truncate(header + "\n" + text, max_chars)
    return out, truncated or trunc2


def _extract_csv(path: Path, max_chars: int, head_rows: int = 50,
                 focus: set[str] | None = None) -> tuple[str, bool]:
    """Head rows + header names + total row count + COMPUTED summary.

    Rows are read with the csv module (not pandas) so malformed /
    ragged rows do not abort the read, and so huge files are only
    streamed once. The summary block is computed over the WHOLE file
    (per-column blanks / distinct / numeric min-max-sum-mean), so
    count/sum/blank questions are answerable even when the needed
    rows are far past the head. Rows containing a focus token are
    shown first, then head rows fill the remainder.
    """
    focus = focus or set()
    header: list[str] = []
    rows: list[list[str]] = []
    focus_rows: list[tuple[int, list[str], set[str]]] = []
    total_rows = 0
    columns: list[dict] = []
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return "[empty CSV file]", False
        columns = [_new_col_acc() for _ in header]
        for row in reader:
            total_rows += 1
            for j, v in enumerate(row[:len(header)]):
                _acc_value(columns[j], v)
            if len(rows) < head_rows:
                rows.append(row)
            if focus:
                joined = " | ".join(row)
                matched = {t for t in focus if t in joined.lower()}
                if matched:
                    # CSVs are streamed, so no IDF pass is possible;
                    # identifier-like tokens (containing a digit,
                    # e.g. 1912/A1, TPO453) rank above plain words.
                    focus_rows.append((total_rows + 1, row, matched))
    lines = [f"[CSV: {total_rows} data rows, {len(header)} columns]",
             "header: " + " | ".join(header)]
    lines.append("[COMPUTED over all rows — do not recompute from the head rows below:]")
    for j, col in enumerate(header):
        lines.append(f"  col {col_letter(j)} ({col}): {_column_stats_from_acc(columns[j])}")
    # Best focus matches first (identifier-like tokens, then
    # token count), then head rows.
    def _csv_rank(item):
        _ln, _row, matched = item
        digit_toks = sum(1 for t in matched if any(c.isdigit() for c in t))
        return (-digit_toks, -len(matched), _ln)
    focus_rows.sort(key=_csv_rank)
    shown: list[tuple[int, list[str]]] = []
    seen_lines = set()
    for ln, row, _matched in focus_rows[:head_rows]:
        shown.append((ln, row))
        seen_lines.add(ln)
    for i, row in enumerate(rows, start=2):  # data row 1 is file line 2
        if i not in seen_lines:
            shown.append((i, row))
    for i, row in shown:
        cells = " | ".join(
            f"{cell_ref(j, i - 1)}={v}" if v not in ("", None) else ""
            for j, v in enumerate(row)
        )
        cells = " ; ".join(c for c in cells.split(" ; ") if c.strip())
        if cells:
            lines.append(f"row {i}: {cells}")
    if total_rows > head_rows:
        lines.append(f"[truncated: showing {len(shown)} of {total_rows} data rows "
                     f"({len(focus_rows)} matched the question)]")
    text = "\n".join(lines)
    return _truncate(text, max_chars)


def _extract_json(path: Path, max_chars: int, focus: set[str] | None = None) -> tuple[str, bool]:
    focus = focus or set()
    # Load the whole file when it is small enough to parse (the
    # summary needs ALL items, e.g. DCC.json is 6.3 MB / 15,962
    # records); fall back to a raw head for very large files.
    data = None
    truncated = False
    if path.stat().st_size < 100_000_000:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                data = json.load(fh)
        except Exception:
            data = None
    if data is None:
        text, truncated, _ = _read_text_head(path, max_chars)
        try:
            data = json.loads(text) if not truncated else None
        except Exception:
            data = None
    if isinstance(data, dict):
        keys = ", ".join(str(k) for k in list(data.keys())[:50])
        head = f"[JSON object, top-level keys: {keys}]\n" + json.dumps(
            data, indent=2, ensure_ascii=False)[:max_chars]
        return _truncate(head, max_chars)
    if isinstance(data, list):
        lines = [f"[JSON array, {len(data)} items]"]
        # COMPUTED summary over ALL items (field-wise), so counting
        # questions are answerable without reading every item.
        if data and all(isinstance(it, dict) for it in data):
            fields: dict[str, dict] = {}
            for it in data:
                for k, v in it.items():
                    acc = fields.setdefault(str(k), _new_col_acc())
                    _acc_value(acc, "" if v is None else str(v))
            lines.append("[COMPUTED over all items:]")
            for k, acc in fields.items():
                lines.append(f"  field {k}: {_column_stats_from_acc(acc)}")
            shown: list[dict] = []
            if focus:
                order = _idf_rank(
                    [json.dumps(it, ensure_ascii=False) for it in data], focus)
                shown = [data[i] for i in order
                         if _focus_score(json.dumps(data[i], ensure_ascii=False),
                                         focus) > 0][:5]
            for it in data[:5]:
                if it not in shown:
                    shown.append(it)
            lines.append("[items: focus matches first, then head items]")
            for it in shown[:10]:
                lines.append(json.dumps(it, ensure_ascii=False)[:1000])
            if len(data) > len(shown):
                lines.append(f"[truncated: showing {len(shown)} of {len(data)} items]")
        else:
            lines.append(json.dumps(data[:5], indent=2, ensure_ascii=False))
        return _truncate("\n".join(lines), max_chars)
    # Unparseable / truncated JSON: return the raw head.
    return _truncate("[JSON: could not parse as a whole document; raw head follows]\n" + text,
                     max_chars)


def _extract_geojson(path: Path, max_chars: int, head_features: int = 20,
                     focus: set[str] | None = None) -> tuple[str, bool]:
    focus = focus or set()
    size = path.stat().st_size
    if size < 20_000_000:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            data = json.load(fh)
        features = data.get("features", []) if isinstance(data, dict) else []
        lines = [f"[GeoJSON FeatureCollection: {len(features)} features]"]
        if isinstance(data, dict) and "bbox" in data:
            lines.append(f"bbox: {data['bbox']}")
        if isinstance(data, dict) and "crs" in data:
            lines.append(f"crs: {data['crs']}")
        # COMPUTED areas (projected metres; UTM zone of the data).
        # Areas are the numbers area questions ask for -- the model
        # must NOT recompute them from coordinates.
        all_x, all_y = [], []
        for feat in features:
            geom = feat.get("geometry", {}) if isinstance(feat, dict) else {}
            coords = geom.get("coordinates") if isinstance(geom, dict) else None
            if coords:
                flat = list(_flatten_coords(coords))
                all_x += [c[0] for c in flat]
                all_y += [c[1] for c in flat]
        project = _projector_for_coords(all_x, all_y)
        areas = [_geom_area_m2(f.get("geometry", {}), project)
                 if isinstance(f, dict) else 0.0 for f in features]
        total_ha = sum(areas) / 10000
        lines.append(f"[COMPUTED areas: {len(features)} features, "
                     f"total area {_fmt_num(total_ha)} hectares "
                     f"(projected metres; use these areas, do not recompute)]")
        # Focus-matching features first, then file order.
        order = list(range(len(features)))
        if focus:
            order = _idf_rank(
                [json.dumps(f.get("properties", {}), ensure_ascii=False)
                 for f in features], focus)
        for idx in order[:head_features]:
            feat = features[idx]
            props = feat.get("properties", {}) if isinstance(feat, dict) else {}
            geom = feat.get("geometry", {}) if isinstance(feat, dict) else {}
            gtype = geom.get("type") if isinstance(geom, dict) else None
            coords = geom.get("coordinates") if isinstance(geom, dict) else None
            coord_txt = json.dumps(coords, ensure_ascii=False)[:200] if coords is not None else ""
            area_txt = ""
            if areas[idx] > 0:
                area_txt = f" area_ha={_fmt_num(areas[idx] / 10000)}"
            lines.append(
                f"feature id={feat.get('id') if isinstance(feat, dict) else ''} "
                f"geometry={gtype}{area_txt} coordinates={coord_txt} "
                f"properties={json.dumps(props, ensure_ascii=False)}")
        if len(features) > head_features:
            lines.append(f"[truncated: showing first {head_features} of {len(features)} features "
                         f"(focus matches first)]")
        return _truncate("\n".join(lines), max_chars)
    # Large GeoJSON: stream -- count features and pull the first few
    # flat "properties" objects with a regex over the file head.
    head = path.read_text(encoding="utf-8", errors="replace")[:500_000]
    props = re.findall(r'"properties"\s*:\s*\{[^{}]*\}', head)
    count = 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for chunk in iter(lambda: fh.read(1_000_000), ""):
            count += chunk.count('"type": "Feature"') + chunk.count('"type":"Feature"')
    lines = [f"[GeoJSON FeatureCollection: ~{count} features (large file, streamed)]"]
    for p in props[:head_features]:
        lines.append(p[:1000])
    if len(props) > head_features or count > head_features:
        lines.append(f"[truncated: showing first {min(len(props), head_features)} feature properties]")
    return _truncate("\n".join(lines), max_chars)


def _flatten_coords(coords):
    """Yield (x, y) pairs from arbitrarily nested coordinate lists."""
    if isinstance(coords, (list, tuple)) and coords and isinstance(coords[0], (int, float)):
        yield (float(coords[0]), float(coords[1]))
        return
    if isinstance(coords, (list, tuple)):
        for c in coords:
            yield from _flatten_coords(c)


def _fmt_value(value) -> str:
    """Compact cell value for excerpts.

    Floats stored by spreadsheets often carry binary noise
    (8738.999999999996 for a stored 8739); %.6g trims that noise
    while keeping the precision the questions need (values are
    reported to two decimals). Strings are passed through, capped
    at 200 characters.
    """
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return format(value, ".6g")
    text = str(value).strip()
    if len(text) > 200:
        text = text[:200] + " [...]"
    return text


def _format_sheet(name: str, frame, max_rows: int = 60,
                  focus: set[str] | None = None) -> str:
    """One spreadsheet sheet as cell-reference lines.

    Read with header=None (see extract_spreadsheet) so pandas row
    index maps directly onto Excel row numbers: pandas row 8 is
    Excel row 9, i.e. cell AA9 is column 26, row index 8.
    Adds a COMPUTED per-column summary over the WHOLE sheet
    (numeric columns: min/max/sum/mean; text columns: top values)
    so sum/count questions are answerable even when the data rows
    are far past the head, and shows focus-matching rows first.
    """
    focus = focus or set()
    nrows, ncols = frame.shape
    lines = [f"[sheet {name}] ({nrows} rows x {ncols} cols)"]
    # COMPUTED column stats over all rows. Row 0 is usually a header:
    # if it holds no numbers at all while other rows do, exclude it
    # from the stats (it is still shown as a row below) so a text
    # header cannot turn a numeric column into a text column.
    all_rows = list(frame.itertuples(index=False, name=None))
    def _is_num(v):
        return isinstance(v, (int, float)) and not (isinstance(v, float) and v != v)
    skip_header = (
        nrows > 1
        and not any(_is_num(v) for v in all_rows[0])
        and any(_is_num(v) for row in all_rows[1:] for v in row))
    stat_rows = all_rows[1:] if skip_header else all_rows
    accs = [_new_col_acc() for _ in range(ncols)]
    for row in stat_rows:
        for j, value in enumerate(row):
            if value is None:
                continue
            if isinstance(value, float) and value != value:
                continue
            _acc_value(accs[j], str(value))
    lines.append("[COMPUTED over all rows of this sheet:]")
    for j, acc in enumerate(accs):
        if acc["n"] == 0:
            continue
        lines.append(f"  col {col_letter(j)}: {_column_stats_from_acc(acc)}")
    # Rows: focus matches first, then head rows.
    order = list(range(len(all_rows)))
    if focus:
        order = _idf_rank(
            [" | ".join(_fmt_value(v) for v in row if v is not None)
             for row in all_rows], focus)
    shown = 0
    for i in order:
        if shown >= max_rows:
            break
        row = all_rows[i]
        parts = []
        for j, value in enumerate(row):
            if value is None:
                continue
            if isinstance(value, float) and value != value:  # NaN
                continue
            text = _fmt_value(value)
            if not text or text.lower() == "nan":
                continue
            parts.append(f"{cell_ref(j, i)}={text}")
        if parts:
            lines.append(" ; ".join(parts))
            shown += 1
    if nrows > max_rows:
        lines.append(f"[truncated: showing {shown} of {nrows} rows of sheet {name} "
                     f"(focus matches first)]")
    return "\n".join(lines)


def _extract_spreadsheet(path: Path, max_chars: int,
                         focus: set[str] | None = None) -> tuple[str, bool]:
    import pandas as pd

    suffix = path.suffix.lower()
    engine = "openpyxl" if suffix == ".xlsx" else "odf"
    try:
        sheets = pd.read_excel(path, sheet_name=None, header=None, engine=engine)
    except Exception as exc:  # corrupt / unreadable workbook
        return f"[spreadsheet could not be read with {engine}: {exc}]", False
    # Data sheets first: rank sheets by number of numeric cells, so
    # cover / contents / notes sheets (mostly text) do not eat the
    # excerpt budget before the data table (e.g. BUS01a) is reached.
    # Sheet names are kept in the output, so cell locators still work.
    def numeric_cells(frame) -> int:
        try:
            return int(frame.apply(pd.to_numeric, errors="coerce").notna().sum().sum())
        except Exception:
            return 0

    ranked = sorted(sheets.items(), key=lambda kv: numeric_cells(kv[1]), reverse=True)
    parts = []
    for name, frame in ranked:
        parts.append(_format_sheet(str(name), frame, focus=focus))
    text = "\n".join(parts)
    return _truncate(text, max_chars)


def _extract_pdf(path: Path, max_chars: int) -> tuple[str, bool]:
    try:
        from pypdf import PdfReader
    except ImportError:
        return "[PDF extraction unavailable: pypdf is not installed]", False
    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        return f"[PDF could not be read: {exc}]", False
    parts = [f"[PDF: {len(reader.pages)} pages]"]
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        parts.append(f"PAGE {i}\n{text}")
    return _truncate("\n".join(parts), max_chars)


def _extract_html(path: Path, max_chars: int) -> tuple[str, bool]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"<script.*?</script>", " ", raw, flags=re.S | re.I)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return _truncate(text.strip(), max_chars)


def _extract_xml_like(path: Path, max_chars: int) -> tuple[str, bool]:
    # .xml / .xsd / .gml: keep the raw text (tag names carry the field
    # names, e.g. GML feature properties), truncated with a marker.
    text, truncated, total_lines = _read_text_head(path, max_chars)
    header = f"[{path.suffix.lower()} text file: {total_lines} lines total]"
    out, trunc2 = _truncate(header + "\n" + text, max_chars)
    return out, truncated or trunc2


def _extract_zip(path: Path, max_chars: int) -> tuple[str, bool]:
    try:
        with zipfile.ZipFile(path) as zf:
            members = [(info.filename, info.file_size) for info in zf.infolist()]
    except Exception as exc:
        return f"[ZIP could not be listed: {exc}]", False
    lines = [f"[ZIP archive: {len(members)} members]"]
    for name, size in members:
        lines.append(f"{name} ({size} bytes)")
    return _truncate("\n".join(lines), max_chars)


def _extract_docx(path: Path, max_chars: int) -> tuple[str, bool]:
    # No python-docx dependency: a .docx is a ZIP; document.xml holds
    # the text. Paragraph breaks are approximated from </w:p>.
    try:
        with zipfile.ZipFile(path) as zf:
            xml = zf.read("word/document.xml").decode("utf-8", errors="replace")
    except Exception as exc:
        return f"[DOCX could not be read: {exc}]", False
    xml = re.sub(r"</w:p>", "\n", xml)
    text = re.sub(r"<[^>]+>", "", xml)
    return _truncate(html.unescape(text).strip(), max_chars)


def _wkb_geom_summary(blob: bytes):
    """(geom_type, area_m2, length_m, centroid) for a WKB geometry
    blob, or None if unparseable. _parse_wkb_geom returns
    (type, coords); areas/lengths are computed here in projected
    metres (identity projector when coordinates are already in
    metres, UTM when they are lon/lat). Any parse error returns
    None -- a corrupt blob must not abort the whole extraction."""
    try:
        parsed = _parse_wkb_geom(blob)
    except Exception:
        return None
    if parsed is None:
        return None
    gtype, coords = parsed
    if gtype == "Point":
        return ("Point", 0.0, 0.0, (round(coords[0], 6), round(coords[1], 6)))
    if gtype == "LineString":
        project = _projector_for_coords([c[0] for c in coords],
                                        [c[1] for c in coords])
        pts = [project(c[0], c[1]) for c in coords]
        length = sum(((pts[i + 1][0] - pts[i][0]) ** 2
                      + (pts[i + 1][1] - pts[i][1]) ** 2) ** 0.5
                     for i in range(len(pts) - 1))
        return ("LineString", 0.0, length, None)
    if gtype == "Polygon":
        project = _projector_for_coords(
            [c[0] for ring in coords for c in ring],
            [c[1] for ring in coords for c in ring])
        outer = [project(c[0], c[1]) for c in coords[0]] if coords else []
        area = abs(_ring_area_m2(outer)) if outer else 0.0
        cx = sum(p[0] for p in outer) / len(outer) if outer else 0.0
        cy = sum(p[1] for p in outer) / len(outer) if outer else 0.0
        return ("Polygon", area, 0.0, (round(cx, 2), round(cy, 2)))
    return (gtype, 0.0, 0.0, None)


def _extract_sqlite(path: Path, max_chars: int,
                    focus: set[str] | None = None) -> tuple[str, bool]:
    """SQLite / GeoPackage: schema + row counts + COMPUTED column
    stats (SQL aggregates over the whole table) + sample rows +
    focus-matching rows. Geometry BLOB columns (WKB) are decoded
    to area / length / centroid where possible.
    """
    focus = focus or set()
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except Exception as exc:
        return f"[SQLite database could not be opened: {exc}]", False
    out = [f"[SQLite database: {path.stat().st_size} bytes]"]
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        for table in tables:
            if table.startswith("sqlite_") or table.startswith("gpkg_"):
                continue
            try:
                nrows = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            except Exception:
                continue
            cols = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
            colnames = [c[1] for c in cols]
            out.append(f"[table {table}: {nrows} rows, columns: {', '.join(colnames)}]")
            # COMPUTED per-column stats via SQL aggregates.
            for col in colnames:
                try:
                    row = conn.execute(
                        f'SELECT COUNT("{col}"), COUNT(DISTINCT "{col}"), '
                        f'MIN("{col}"), MAX("{col}"), AVG("{col}"), SUM("{col}") '
                        f'FROM "{table}"').fetchone()
                except Exception:
                    continue
                nn, nd, mn, mx, avg, total = row
                blanks = nrows - (nn or 0)
                if isinstance(mn, (int, float)) and isinstance(mx, (int, float)):
                    out.append(f"  col {col}: n={nrows} blank={blanks} distinct={nd} "
                               f"numeric min={_fmt_num(mn)} max={_fmt_num(mx)} "
                               f"sum={_fmt_num(total) if total is not None else ''} "
                               f"mean={_fmt_num(avg) if avg is not None else ''}")
                else:
                    top = ""
                    try:
                        top_rows = conn.execute(
                            f'SELECT "{col}", COUNT(*) c FROM "{table}" '
                            f'WHERE "{col}" IS NOT NULL GROUP BY "{col}" '
                            f'ORDER BY c DESC LIMIT 5').fetchall()
                        # Binary (geometry blob) values are not
                        # printable -- omit them instead of dumping bytes.
                        printable = [(v, c) for v, c in top_rows
                                     if not isinstance(v, (bytes, bytearray))]
                        if printable:
                            top = " top=" + ", ".join(
                                f"{str(v)[:80]}({c})" for v, c in printable)
                        elif top_rows:
                            top = " top=<binary values>"
                    except Exception:
                        pass
                    out.append(f"  col {col}: n={nrows} blank={blanks} "
                               f"distinct={nd}{top}")
            # Sample rows: focus matches first, then head rows.
            shown: list[tuple] = []
            distinctive = sorted(t for t in focus if len(t) >= 4)
            if distinctive and colnames:
                text_cols = [c for c in colnames]
                cond = " OR ".join(
                    f'CAST("{c}" AS TEXT) LIKE ?' for c in text_cols)
                params = [f"%{tok}%" for tok in distinctive for _ in text_cols]
                try:
                    rows = conn.execute(
                        f'SELECT * FROM "{table}" WHERE {cond} LIMIT 5',
                        params).fetchall()
                    shown.extend(rows)
                except Exception:
                    pass
            try:
                head = conn.execute(f'SELECT * FROM "{table}" LIMIT 5').fetchall()
            except Exception:
                head = []
            for r in head:
                if r not in shown:
                    shown.append(r)
            for r in shown[:10]:
                cells = []
                for j, v in enumerate(r):
                    if v is None:
                        continue
                    if isinstance(v, (bytes, bytearray)):
                        geom = _wkb_geom_summary(bytes(v))
                        if geom is not None:
                            gtype, area, length, centroid = geom
                            cells.append(
                                f"{colnames[j]}=WKB {gtype} "
                                f"area_m2={_fmt_num(area)} length_m={_fmt_num(length)} "
                                f"centroid={centroid}")
                        else:
                            cells.append(f"{colnames[j]}=<{len(v)}-byte blob>")
                    else:
                        cells.append(f"{colnames[j]}={v}")
                out.append("  row: " + " ; ".join(str(c)[:300] for c in cells))
            if nrows > len(shown):
                out.append(f"  [truncated: showing {len(shown)} of {nrows} rows "
                           f"of table {table} (focus matches first)]")
    finally:
        conn.close()
    return _truncate("\n".join(out), max_chars)


def _extract_tiff(path: Path, max_chars: int) -> tuple[str, bool]:
    """GeoTIFF / TIFF: size, mode, frames + COMPUTED per-frame
    numeric stats (min/max/mean) over the whole raster, so
    value-at-cell / min / max / mean questions are answerable
    without reading every pixel. Cell values are NOT dumped
    (a 4000x4000 frame has 16M cells); stats + metadata only.
    """
    try:
        from PIL import Image
    except ImportError:
        return "[TIFF extraction unavailable: Pillow is not installed]", False
    try:
        import numpy as np
    except ImportError:
        np = None
    try:
        img = Image.open(path)
    except Exception as exc:
        return f"[TIFF could not be opened: {exc}]", False
    out = [f"[TIFF: {img.size[0]}x{img.size[1]} px, mode={img.mode}, "
           f"format={img.format}, frames={getattr(img, 'n_frames', 1)}]"]
    try:
        tags = {k: img.tag_v2[k] for k in img.tag_v2.keys()} if hasattr(img, "tag_v2") else {}
        for k, v in list(tags.items())[:12]:
            out.append(f"  tag {k}: {str(v)[:200]}")
    except Exception:
        pass
    n_frames = getattr(img, "n_frames", 1)
    for fi in range(min(n_frames, 10)):
        try:
            img.seek(fi)
        except Exception:
            break
        if np is not None:
            try:
                arr = np.asarray(img, dtype=np.float64)
                flat = arr.ravel()
                out.append(f"  [COMPUTED frame {fi}: min={_fmt_num(float(flat.min()))} "
                           f"max={_fmt_num(float(flat.max()))} "
                           f"mean={_fmt_num(float(flat.mean()))} "
                           f"nonzero={int((flat != 0).sum())} of {flat.size} cells]")
                continue
            except Exception:
                pass
        try:
            lo, hi = img.getextrema()
            out.append(f"  [frame {fi}: min={lo} max={hi} (PIL extrema)]")
        except Exception:
            out.append(f"  [frame {fi}: stats unavailable]")
    if n_frames > 10:
        out.append(f"[truncated: showing stats for first 10 of {n_frames} frames]")
    return _truncate("\n".join(out), max_chars)


def _extract_metadata_only(path: Path, max_chars: int, note: str) -> tuple[str, bool]:
    size = path.stat().st_size
    text = (f"[metadata only: {path.name} is {size} bytes. {note} "
            "No values were read from this file and none are claimed.]")
    return text, False


def extract_file(path: Path, source_id: str = "", original_file: str = "",
                 max_chars: int = DEFAULT_MAX_CHARS,
                 focus: set[str] | None = None) -> dict:
    """Extract one source file to plain text with locators.

    ``focus`` is the question's token set (see focus_tokens): rows /
    items / features matching it are shown first, and computed
    summaries cover the whole file so the model does not have to
    guess from the head rows.
    """
    path = Path(path)
    original_file = original_file or path.name
    suffix = path.suffix.lower()
    focus = focus or set()

    if not path.exists():
        return _result(source_id, original_file, path, "", False,
                       note=f"file not found at {path}")
    if suffix in (".txt", ".md"):
        text, truncated = _extract_txt_md(path, max_chars)
        note = ""
    elif suffix == ".csv":
        text, truncated = _extract_csv(path, max_chars, focus=focus)
        note = ""
    elif suffix == ".json":
        text, truncated = _extract_json(path, max_chars, focus=focus)
        note = ""
    elif suffix == ".geojson":
        text, truncated = _extract_geojson(path, max_chars, focus=focus)
        note = ""
    elif suffix in (".xlsx", ".ods"):
        text, truncated = _extract_spreadsheet(path, max_chars, focus=focus)
        note = ""
    elif suffix == ".pdf":
        text, truncated = _extract_pdf(path, max_chars)
        note = ""
    elif suffix == ".html":
        text, truncated = _extract_html(path, max_chars)
        note = ""
    elif suffix in (".xml", ".xsd", ".gml"):
        text, truncated = _extract_xml_like(path, max_chars)
        note = ""
    elif suffix == ".zip":
        text, truncated = _extract_zip(path, max_chars)
        note = ""
    elif suffix == ".docx":
        text, truncated = _extract_docx(path, max_chars)
        note = ""
    elif suffix in (".sqlite", ".gpkg"):
        text, truncated = _extract_sqlite(path, max_chars, focus=focus)
        note = ""
    elif suffix in (".tif", ".tiff"):
        text, truncated = _extract_tiff(path, max_chars)
        note = ""
    elif suffix in METADATA_ONLY:
        text, truncated = _extract_metadata_only(
            path, max_chars,
            "Native geometry/raster/3D processing is not yet implemented.")
        note = "native processing not yet implemented"
    else:
        # Unknown suffix: try plain text, fall back to metadata only.
        try:
            text, truncated = _extract_txt_md(path, max_chars)
            note = ""
        except Exception:
            text, truncated = _extract_metadata_only(
                path, max_chars, "Unrecognised format.")
            note = "unrecognised format"
    return _result(source_id, original_file, path, text, truncated, note=note)


def extract_source(source_id: str, manifest_row: dict, sources_dir: Path,
                   max_chars: int = DEFAULT_MAX_CHARS,
                   focus: set[str] | None = None) -> dict:
    """Resolve a manifest row to a file on disk and extract it.

    ``manifest_row`` is a row of SourceManifest.csv (keys Source_ID,
    File, Original_File). ``sources_dir`` is data/ ; the manifest
    ``File`` column is relative to the challenge data folder, e.g.
    ``sources/SRC_...__file.csv``.
    """
    data_dir = Path(sources_dir)
    rel = manifest_row.get("File", "")
    path = data_dir / rel if rel else data_dir / manifest_row.get("Original_File", "")
    return extract_file(path, source_id=source_id,
                        original_file=manifest_row.get("Original_File", path.name),
                        max_chars=max_chars, focus=focus)


if __name__ == "__main__":
    import sys
    for arg in sys.argv[1:]:
        result = extract_file(Path(arg))
        print(f"--- {result['file']} ({result['suffix']}, "
              f"{result['size_bytes']} bytes, truncated={result['truncated']}) ---")
        print(result["text"][:2000])
