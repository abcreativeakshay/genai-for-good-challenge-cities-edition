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


def _extract_csv(path: Path, max_chars: int, head_rows: int = 50) -> tuple[str, bool]:
    """Head rows + header names + total row count.

    Rows are read with the csv module (not pandas) so malformed /
    ragged rows do not abort the read, and so huge files are only
    streamed once.
    """
    header: list[str] = []
    rows: list[list[str]] = []
    total_rows = 0
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return "[empty CSV file]", False
        for row in reader:
            total_rows += 1
            if len(rows) < head_rows:
                rows.append(row)
    lines = [f"[CSV: {total_rows} data rows, {len(header)} columns]",
             "header: " + " | ".join(header)]
    for i, row in enumerate(rows, start=2):  # data row 1 is file line 2
        cells = " | ".join(
            f"{cell_ref(j, i - 1)}={v}" if v not in ("", None) else ""
            for j, v in enumerate(row)
        )
        cells = " ; ".join(c for c in cells.split(" ; ") if c.strip())
        if cells:
            lines.append(f"row {i}: {cells}")
    if total_rows > head_rows:
        lines.append(f"[truncated: showing first {head_rows} of {total_rows} data rows]")
    text = "\n".join(lines)
    return _truncate(text, max_chars)


def _extract_json(path: Path, max_chars: int) -> tuple[str, bool]:
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
        head = f"[JSON array, {len(data)} items]\n" + json.dumps(
            data[:5], indent=2, ensure_ascii=False)
        return _truncate(head, max_chars)
    # Unparseable / truncated JSON: return the raw head.
    return _truncate("[JSON: could not parse as a whole document; raw head follows]\n" + text,
                     max_chars)


def _extract_geojson(path: Path, max_chars: int, head_features: int = 20) -> tuple[str, bool]:
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
        for feat in features[:head_features]:
            props = feat.get("properties", {}) if isinstance(feat, dict) else {}
            geom = feat.get("geometry", {}) if isinstance(feat, dict) else {}
            gtype = geom.get("type") if isinstance(geom, dict) else None
            coords = geom.get("coordinates") if isinstance(geom, dict) else None
            coord_txt = json.dumps(coords, ensure_ascii=False)[:200] if coords is not None else ""
            lines.append(
                f"feature id={feat.get('id') if isinstance(feat, dict) else ''} "
                f"geometry={gtype} coordinates={coord_txt} "
                f"properties={json.dumps(props, ensure_ascii=False)}")
        if len(features) > head_features:
            lines.append(f"[truncated: showing first {head_features} of {len(features)} features]")
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


def _format_sheet(name: str, frame, max_rows: int = 60) -> str:
    """One spreadsheet sheet as cell-reference lines.

    Read with header=None (see extract_spreadsheet) so pandas row
    index maps directly onto Excel row numbers: pandas row 8 is
    Excel row 9, i.e. cell AA9 is column 26, row index 8.
    """
    nrows, ncols = frame.shape
    lines = [f"[sheet {name}] ({nrows} rows x {ncols} cols)"]
    for i, row in enumerate(frame.head(max_rows).itertuples(index=False, name=None)):
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
    if nrows > max_rows:
        lines.append(f"[truncated: showing first {max_rows} of {nrows} rows of sheet {name}]")
    return "\n".join(lines)


def _extract_spreadsheet(path: Path, max_chars: int) -> tuple[str, bool]:
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
        parts.append(_format_sheet(str(name), frame))
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


def _extract_metadata_only(path: Path, max_chars: int, note: str) -> tuple[str, bool]:
    size = path.stat().st_size
    text = (f"[metadata only: {path.name} is {size} bytes. {note} "
            "No values were read from this file and none are claimed.]")
    return text, False


def extract_file(path: Path, source_id: str = "", original_file: str = "",
                 max_chars: int = DEFAULT_MAX_CHARS) -> dict:
    """Extract one source file to plain text with locators."""
    path = Path(path)
    original_file = original_file or path.name
    suffix = path.suffix.lower()

    if not path.exists():
        return _result(source_id, original_file, path, "", False,
                       note=f"file not found at {path}")
    if suffix in (".txt", ".md"):
        text, truncated = _extract_txt_md(path, max_chars)
        note = ""
    elif suffix == ".csv":
        text, truncated = _extract_csv(path, max_chars)
        note = ""
    elif suffix == ".json":
        text, truncated = _extract_json(path, max_chars)
        note = ""
    elif suffix == ".geojson":
        text, truncated = _extract_geojson(path, max_chars)
        note = ""
    elif suffix in (".xlsx", ".ods"):
        text, truncated = _extract_spreadsheet(path, max_chars)
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
    elif suffix == ".sqlite":
        try:
            with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
                tables = [r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
            text = (f"[SQLite database: {path.stat().st_size} bytes. "
                    f"Tables: {', '.join(tables)}. "
                    "Native SQL/graph computation is not yet implemented; "
                    "no values were read from this file.]")
        except Exception as exc:
            text = f"[SQLite database could not be opened: {exc}]"
        truncated, note = False, "native SQLite computation not yet implemented"
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
                   max_chars: int = DEFAULT_MAX_CHARS) -> dict:
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
                        max_chars=max_chars)


if __name__ == "__main__":
    import sys
    for arg in sys.argv[1:]:
        result = extract_file(Path(arg))
        print(f"--- {result['file']} ({result['suffix']}, "
              f"{result['size_bytes']} bytes, truncated={result['truncated']}) ---")
        print(result["text"][:2000])
