"""
springer_to_json.py
Parse Springer JATS XML files and convert to the same JSON format
as the Elsevier processed files (data/processed/*.json).
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

INPUT_DIR  = Path("data/raw/springer")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_text(el):
    """Get all text content from an element, ignoring sub-element tags."""
    if el is None:
        return ""
    return " ".join(el.itertext()).strip()


def _jats_build_header_grid(thead):
    """Deterministically resolves the effective leaf (one-per-column)
    headers for a JATS/HTML <thead>, tracking colspan/rowspan occupancy
    across header rows (there is no CALS-style colspec/namest/nameend in
    JATS — column position must be reconstructed from row order + colspan/
    rowspan alone). Never guesses: a row only becomes the leaf-header row
    if, combined with any single-column cell carried down by rowspan from
    an earlier row, it covers every column exactly once with width 1.
    Rows/cells that don't qualify (colspan>1) are preserved as
    header_groups rather than discarded or flattened into `headers`."""
    rows = [r for r in thead.iter("tr")]
    if not rows:
        return [], [], "no_header"

    # carry[col] = (rows_remaining_after_this_row, label, span_start, span_end)
    # span_start/span_end identify the ORIGINAL source cell a carried column
    # belongs to, so a wide (colspan>1) rowspan'd cell is recognized as one
    # unit wherever it recurs, rather than as several independent columns.
    carry = {}
    per_row_info = []  # (new_spans, carried_in_snapshot) per row, in order
    n_cols = 0

    for row in rows:
        cells = [c for c in row if c.tag in ("th", "td")]
        carried_in = {c: (lbl, s, e) for c, (_, lbl, s, e) in carry.items()}
        occupied = set(carried_in.keys())
        col = 1
        new_spans = []
        new_carry_additions = {}
        for cell in cells:
            while col in occupied:
                col += 1
            try:
                colspan = int(cell.get("colspan") or 1)
            except (TypeError, ValueError):
                colspan = 1
            try:
                rowspan = int(cell.get("rowspan") or 1)
            except (TypeError, ValueError):
                rowspan = 1
            label = get_text(cell)
            start, end = col, col + colspan - 1
            new_spans.append((start, end, label))
            if rowspan > 1:
                # rows remaining AFTER the row this cell is placed in --
                # must not be decremented again in this same iteration, or
                # a rowspan="2" cell would incorrectly vanish before the
                # very next row (the one it's meant to carry into) sees it.
                for c in range(start, end + 1):
                    new_carry_additions[c] = (rowspan - 1, label, start, end)
            occupied |= set(range(start, end + 1))
            col = end + 1
        n_cols = max(n_cols, col - 1, max(occupied, default=0))
        per_row_info.append((new_spans, carried_in))
        carry = {c: (rem - 1, lbl, s, e) for c, (rem, lbl, s, e) in carry.items() if rem - 1 > 0}
        carry.update(new_carry_additions)

    leaf_headers = None
    header_groups = []
    added_group_spans = set()

    for new_spans, carried_in in per_row_info:
        covered = set(carried_in.keys())
        gap_free = True
        for start, end, _ in new_spans:
            col_range = set(range(start, end + 1))
            if covered & col_range:
                gap_free = False
            covered |= col_range
        is_complete = gap_free and covered == set(range(1, n_cols + 1))
        row_all_new_narrow = all(end == start for start, end, _ in new_spans)

        if is_complete and row_all_new_narrow:
            if leaf_headers is not None:
                # a shallower row was already a complete leaf row (genuine
                # multi-level leaf structure) -- preserve it as group
                # context rather than discarding it.
                for i, prev_text in enumerate(leaf_headers):
                    if prev_text:
                        header_groups.append({"label": prev_text, "start_col": i + 1, "end_col": i + 1})
            row_headers = [None] * n_cols
            seen_carry_spans = set()
            for c, (label, s, e) in sorted(carried_in.items()):
                if (s, e, label) not in seen_carry_spans:
                    row_headers[s - 1] = label
                    seen_carry_spans.add((s, e, label))
            for start, end, label in new_spans:
                row_headers[start - 1] = label
            leaf_headers = row_headers
        else:
            for start, end, label in new_spans:
                if end > start and (start, end, label) not in added_group_spans:
                    header_groups.append({"label": label, "start_col": start, "end_col": end})
                    added_group_spans.add((start, end, label))

    if leaf_headers is None:
        return [], header_groups, "unresolved"
    return leaf_headers, header_groups, "resolved"


def _jats_build_tbody_grid(tbody, n_cols):
    """Deterministically reconstructs JATS/HTML <tbody> rows into an
    n_cols-wide grid, tracking colspan/rowspan occupancy (there is no
    colname/namest/nameend in JATS, so placement relies purely on row
    order + colspan/rowspan, mirroring _jats_build_header_grid's carry
    mechanism). Per output cell, distinguishes:
      "explicit"      -- a real <td>/<th> placed in this row
      "carried"       -- structurally repeated from an earlier row's
                          rowspan (never claimed to be literally repeated
                          in the source XML)
      "spanned_empty" -- a column consumed by a wide (colspan>1) cell
                          whose value was recorded once, at its first
                          column, not fabricated N times
      "unresolved"    -- placement could not be determined; that row's
                          literal, unpadded cell text is kept as-is
    Returns (rows, row_provenance, tbody_status): "resolved" (every row
    placed), "partial" (some rows unresolved), or "no_body" when there is
    no tbody or no target width to reconstruct against."""
    if tbody is None or not n_cols:
        return None, None, "no_body"

    raw_rows = tbody.findall("tr")
    if not raw_rows:
        return None, None, "no_body"

    carry = {}  # col -> (value, source_row_index, source_col)
    carry_remaining = {}  # col -> rows_remaining
    out_rows, out_prov = [], []
    any_unresolved = False

    for r_idx, row in enumerate(raw_rows):
        cells = [c for c in row if c.tag in ("th", "td")]
        carried_in = dict(carry)
        occupied = set(carried_in.keys())
        col = 1
        placements = []
        for cell in cells:
            while col in occupied:
                col += 1
            try:
                colspan = int(cell.get("colspan") or 1)
            except (TypeError, ValueError):
                colspan = 1
            try:
                rowspan = int(cell.get("rowspan") or 1)
            except (TypeError, ValueError):
                rowspan = 1
            value = get_text(cell)
            start, end = col, col + colspan - 1
            placements.append((start, end, value, rowspan))
            occupied |= set(range(start, end + 1))
            col = end + 1

        complete = occupied == set(range(1, n_cols + 1))
        new_carry_additions, new_carry_remaining = {}, {}

        if not complete:
            literal = [get_text(c) for c in cells]
            out_rows.append(literal)
            out_prov.append([{"source": "unresolved", "source_row_index": r_idx, "source_col": None}
                              for _ in literal])
            any_unresolved = True
        else:
            row_values, row_prov = [None] * n_cols, [None] * n_cols
            for c, (value, source_row, source_col) in carried_in.items():
                row_values[c - 1] = value
                row_prov[c - 1] = {"source": "carried", "source_row_index": source_row, "source_col": source_col}
            for start, end, value, rowspan in placements:
                row_values[start - 1] = value
                row_prov[start - 1] = {"source": "explicit", "source_row_index": r_idx, "source_col": start}
                for c in range(start + 1, end + 1):
                    row_values[c - 1] = None
                    row_prov[c - 1] = {"source": "spanned_empty", "source_row_index": r_idx, "source_col": start}
                if rowspan > 1:
                    for c in range(start, end + 1):
                        new_carry_additions[c] = (value, r_idx, start)
                        new_carry_remaining[c] = rowspan - 1
            out_rows.append(row_values)
            out_prov.append(row_prov)

        carry_remaining = {c: rem - 1 for c, rem in carry_remaining.items() if rem - 1 > 0}
        carry = {c: v for c, v in carry.items() if c in carry_remaining}
        carry.update(new_carry_additions)
        carry_remaining.update(new_carry_remaining)

    if not out_rows:
        return None, None, "no_body"
    return out_rows, out_prov, ("partial" if any_unresolved else "resolved")


def parse_jats_xml(xml_text):
    """
    Parse a Springer JATS XML file and return a dict matching the Elsevier JSON format:
    {doi, title, abstract, journal, year, sections[], tables[]}
    """
    root = ET.fromstring(xml_text)

    # find the article element (may be nested inside <response>)
    article = root.find(".//article")
    if article is None:
        return None

    front = article.find("front")
    if front is None:
        return None

    article_meta = front.find("article-meta")
    journal_meta = front.find("journal-meta")

    # ── Metadata ─────────────────────────────────────────────────────────────
    doi = ""
    for aid in (article_meta.findall("article-id") if article_meta else []):
        if aid.get("pub-id-type") == "doi":
            doi = aid.text or ""
            break

    title = ""
    title_group = article_meta.find("title-group") if article_meta else None
    if title_group is not None:
        at = title_group.find("article-title")
        title = get_text(at)

    abstract = ""
    abs_el = article_meta.find("abstract") if article_meta else None
    if abs_el is not None:
        abstract = get_text(abs_el)

    journal = ""
    if journal_meta is not None:
        jt = journal_meta.find("journal-title-group/journal-title")
        if jt is None:
            jt = journal_meta.find(".//journal-title")
        journal = get_text(jt)

    year = ""
    if article_meta is not None:
        for pub_date in article_meta.findall("pub-date"):
            yr = pub_date.find("year")
            if yr is not None and yr.text:
                year = yr.text
                break

    # ── Sections (body text) ─────────────────────────────────────────────────
    body = article.find("body")
    sections = []
    if body is not None:
        for sec in body.iter("sec"):
            heading_el = sec.find("title")
            heading = get_text(heading_el) if heading_el is not None else ""

            # collect paragraph text directly in this section (not sub-sections)
            paragraphs = []
            for p in sec.findall("p"):
                text = get_text(p).strip()
                if text:
                    paragraphs.append(text)

            if heading or paragraphs:
                sections.append({"heading": heading, "paragraphs": paragraphs})

    # ── Tables ───────────────────────────────────────────────────────────────
    tables = []
    for table_wrap in article.iter("table-wrap"):
        # caption
        cap_el = table_wrap.find(".//caption/p")
        if cap_el is None:
            cap_el = table_wrap.find(".//caption/title")
        caption = get_text(cap_el)

        table_el = table_wrap.find(".//table")
        if table_el is None:
            continue

        # headers — from <thead>, effective leaf headers (never a positional
        # flatten of every <tr> in a multi-row thead)
        headers, header_groups, alignment_status = [], [], "no_header"
        thead = table_el.find("thead")
        if thead is not None:
            headers, header_groups, alignment_status = _jats_build_header_grid(thead)

        # rows — from <tbody>, reconstructed into an n_cols-wide grid (using
        # the resolved leaf-header count as the target width) whenever
        # possible, so rowspan/colspan-continuation rows are no longer
        # silently ragged/skipped downstream.
        tbody = table_el.find("tbody")
        n_cols = len(headers) if alignment_status == "resolved" else None
        grid_rows, row_provenance, tbody_status = _jats_build_tbody_grid(tbody, n_cols)

        if grid_rows is not None:
            rows = [r for r in grid_rows if any(c for c in r)]
            row_provenance = [p for r, p in zip(grid_rows, row_provenance) if any(c for c in r)]
        else:
            rows, row_provenance = [], []
            if tbody is not None:
                for tr in tbody.findall("tr"):
                    row_cells = []
                    for cell in tr:
                        if cell.tag in ("th", "td"):
                            row_cells.append(get_text(cell))
                    if row_cells:
                        rows.append(row_cells)
                        row_provenance.append([{"source": "explicit", "source_row_index": None, "source_col": None}
                                                for _ in row_cells])
            tbody_status = "no_headers_to_target" if tbody is not None else "no_body"

        if rows:
            tables.append({
                "caption": caption,
                "headers": headers,
                "rows":    rows,
                "header_groups": header_groups,
                "table_alignment_status": alignment_status,
                "row_provenance": row_provenance,
                "tbody_alignment_status": tbody_status,
            })

    return {
        "doi":      doi,
        "title":    title,
        "abstract": abstract,
        "journal":  journal,
        "year":     year,
        "sections": sections,
        "tables":   tables,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    xml_files = list(INPUT_DIR.glob("*.xml"))
    print(f"Found {len(xml_files)} Springer XML files\n")

    converted = 0
    failed    = 0

    for i, xml_path in enumerate(sorted(xml_files), 1):
        # skip if already converted
        safe = xml_path.stem  # e.g. "10-1007_s11661-021-06534-9"
        out_path_check = OUTPUT_DIR / f"springer_{safe}.json"
        if out_path_check.exists():
            continue

        print(f"[{i}/{len(xml_files)}] {xml_path.name}")

        try:
            xml_text = xml_path.read_text(encoding="utf-8")
            paper = parse_jats_xml(xml_text)
        except ET.ParseError as e:
            print(f"  XML parse error: {e}")
            failed += 1
            continue

        if paper is None:
            print(f"  No <article> element found — skip")
            failed += 1
            continue

        n_sections = len(paper["sections"])
        n_tables   = len(paper["tables"])
        print(f"  title   : {paper['title'][:65]}...")
        print(f"  doi     : {paper['doi']}")
        print(f"  year    : {paper['year']}")
        print(f"  sections: {n_sections}  tables: {n_tables}")

        # save — prefix with springer_ to distinguish from Elsevier files
        safe_doi = paper["doi"].replace("/", "_").replace(".", "-")
        out_name = f"springer_{safe_doi}.json"
        out_path = OUTPUT_DIR / out_name

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(paper, f, indent=2, ensure_ascii=False)

        print(f"  saved → {out_name}")
        converted += 1
        print()

    print("=" * 50)
    print(f"Converted : {converted}")
    print(f"Failed    : {failed}")
    print(f"Output    : {OUTPUT_DIR}/springer_*.json")
    print()
    print("Next step: run llm_extraction.py (it will pick up all processed/*.json)")


if __name__ == "__main__":
    main()
