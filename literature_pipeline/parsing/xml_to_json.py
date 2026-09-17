import json
import os
from pathlib import Path
from xml.etree import ElementTree as ET

INPUT_DIR  = "data/raw"       # XML files from Elsevier
OUTPUT_DIR = "data/processed" # one JSON per article goes here

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Elsevier XML uses many namespaces — we need these to find tags correctly
NS = {
    "dc":     "http://purl.org/dc/elements/1.1/",
    "prism":  "http://prismstandard.org/namespaces/basic/2.0/",
    "ce":     "http://www.elsevier.com/xml/common/dtd",
    "xocs":   "http://www.elsevier.com/xml/xocs/dtd",
    "cals":   "http://www.elsevier.com/xml/common/cals/dtd",
}


def clean_text(element):
    """Get all text inside an element, stripping XML tags like <ce:cross-ref>."""
    if element is None:
        return ""
    return " ".join(element.itertext()).strip()


def _parse_colspecs(table):
    """Returns (colname_to_colnum, n_cols) from this table's <colspec>
    elements, or (None, None) if no colspecs are declared (some CALS
    tables omit them for simple, ungrouped headers). Per the CALS/OASIS
    table model, a <colspec> that omits an explicit colnum is implicitly
    numbered by its own document-order position among the colspecs (1st
    colspec = column 1, 2nd = column 2, ...) -- this is a real, standard
    pattern, not an edge case, so it is resolved the same way rather than
    left unresolved."""
    tgroup = table.find(".//{%s}tgroup" % NS["cals"])
    if tgroup is None:
        return None, None
    colname_to_colnum = {}
    for position, cs in enumerate(tgroup.findall("{%s}colspec" % NS["cals"]), start=1):
        name, num = cs.get("colname"), cs.get("colnum")
        if not name:
            continue
        if num:
            try:
                colname_to_colnum[name] = int(num)
                continue
            except ValueError:
                pass
        colname_to_colnum[name] = position
    if not colname_to_colnum:
        return None, None
    return colname_to_colnum, max(colname_to_colnum.values())


def _entry_span(entry, colname_to_colnum, next_col):
    """Returns (start_col, end_col), 1-indexed inclusive, for one <entry>.
    Resolved via namest/nameend (group span) or colname (single column)
    when present. Per the CALS/OASIS table model, an entry that carries
    NONE of these is implicitly positioned at the next unoccupied column
    in document order (`next_col`) — this is a real, common, valid CALS
    pattern (colspec declared for the table, but individual <entry>
    elements left unlabeled), not an ambiguous case, so it is resolved
    positionally rather than marked unresolved."""
    namest, nameend = entry.get("namest"), entry.get("nameend")
    if namest and nameend:
        start, end = colname_to_colnum.get(namest), colname_to_colnum.get(nameend)
        return (start, end) if start is not None and end is not None else None
    colname = entry.get("colname")
    if colname:
        col = colname_to_colnum.get(colname)
        return (col, col) if col is not None else None
    return (next_col, next_col)


def _build_header_grid(table, thead):
    """Deterministically resolves the effective leaf (one-per-column)
    headers for a table's <thead>, using CALS colspec/colname/namest/
    nameend structure plus morerows (row-span) occupancy carried across
    header rows — some tables use morerows within <thead> itself (e.g. a
    row-group label spanning two header rows alongside colspan'd group
    headers), not only in <tbody>. Never positionally flattens rows
    together and never guesses: a row only becomes the leaf-header row if
    its own entries, combined with any single-column label carried down by
    an earlier row's morerows, resolve to a complete, non-overlapping cover
    of every column with width exactly 1. Rows that don't qualify
    (group-label rows, spanning entries) are preserved separately as
    header_groups rather than being discarded or concatenated into
    `headers`. If no row in a multi-row <thead> can be resolved this way
    (e.g. no colspec at all to disambiguate multiple rows), alignment is
    reported as unresolved rather than guessed."""
    colname_to_colnum, n_cols = _parse_colspecs(table)
    rows = thead.findall("{%s}row" % NS["cals"])

    leaf_headers = None
    header_groups = []

    if colname_to_colnum is not None:
        carry = {}  # col -> (label, source_start, source_end)
        carry_remaining = {}  # col -> rows_remaining

        for row in rows:
            entries = row.findall("{%s}entry" % NS["ce"])
            carried_in = dict(carry)
            occupied = set(carried_in.keys())
            spans = []
            resolvable = True
            next_col = 1
            new_carry_additions, new_carry_remaining = {}, {}
            for entry in entries:
                while next_col in occupied:
                    next_col += 1
                span = _entry_span(entry, colname_to_colnum, next_col)
                if span is None:
                    resolvable = False
                    break
                start, end = span
                text = clean_text(entry)
                spans.append((span, text))
                morerows_raw = entry.get("morerows")
                try:
                    extra_rows = int(morerows_raw) if morerows_raw else 0
                except ValueError:
                    extra_rows = 0
                if extra_rows > 0 and end == start:
                    # only single-column carries are propagated as reusable
                    # leaf labels; a wide (colspan>1) morerows'd cell is
                    # still recorded via header_groups below, not carried
                    # as if it were an independent leaf slot.
                    new_carry_additions[start] = (text, start, end)
                    new_carry_remaining[start] = extra_rows
                occupied |= set(range(start, end + 1))
                next_col = end + 1
            if not resolvable:
                carry_remaining = {c: rem - 1 for c, rem in carry_remaining.items() if rem - 1 > 0}
                carry = {c: v for c, v in carry.items() if c in carry_remaining}
                continue

            covered = set(carried_in.keys())
            is_leaf_complete = True
            for (start, end), _ in spans:
                if end - start != 0:
                    is_leaf_complete = False
                col_range = set(range(start, end + 1))
                if covered & col_range:
                    is_leaf_complete = False
                covered |= col_range
            if covered != set(range(1, n_cols + 1)):
                is_leaf_complete = False

            if is_leaf_complete:
                if leaf_headers is not None:
                    # A shallower row was already a complete one-per-column
                    # leaf row (a genuine multi-level leaf structure, e.g. a
                    # units sub-header) — preserve it as group context
                    # rather than silently discarding it.
                    for i, prev_text in enumerate(leaf_headers):
                        if prev_text:
                            header_groups.append({"label": prev_text, "start_col": i + 1, "end_col": i + 1})
                row_headers = [None] * n_cols
                seen_carry_spans = set()
                for c, (label, s, e) in sorted(carried_in.items()):
                    if (s, e, label) not in seen_carry_spans:
                        row_headers[s - 1] = label
                        seen_carry_spans.add((s, e, label))
                for (start, end), text in spans:
                    row_headers[start - 1] = text
                leaf_headers = row_headers
            else:
                for (start, end), text in spans:
                    if end > start:
                        header_groups.append({"label": text, "start_col": start, "end_col": end})

            carry_remaining = {c: rem - 1 for c, rem in carry_remaining.items() if rem - 1 > 0}
            carry = {c: v for c, v in carry.items() if c in carry_remaining}
            carry.update(new_carry_additions)
            carry_remaining.update(new_carry_remaining)
    elif len(rows) == 1:
        # No colspec declared, but an unambiguous single header row — the
        # common simple case, resolvable purely positionally.
        entries = rows[0].findall("{%s}entry" % NS["ce"])
        leaf_headers = [clean_text(e) for e in entries]

    if leaf_headers is None:
        return [], header_groups, "unresolved"
    return leaf_headers, header_groups, "resolved"


def _build_tbody_grid(tbody, colname_to_colnum, n_cols):
    """Deterministically reconstructs CALS <tbody> rows into an n_cols-wide
    grid, using colname/namest/nameend (as _entry_span does for thead) plus
    morerows (row-span) occupancy tracking. Distinguishes, per output cell,
    whether its value is:
      - "explicit"      -- a real <entry> placed in this row
      - "carried"       -- structurally repeated from an earlier row's
                            morerows span (never claimed to have been
                            literally repeated in the source XML)
      - "spanned_empty" -- a column consumed by a wide (namest/nameend)
                            data cell whose value was recorded once, at its
                            first column, not fabricated N times
      - "unresolved"    -- placement could not be determined; that row's
                            literal, unpadded cell text is kept as-is
                            (never guessed, never positionally shifted)
    Returns (rows, row_provenance, tbody_status). tbody_status is
    "resolved" (every row placed), "partial" (some rows unresolved), or
    "no_colspec"/"no_body" when reconstruction could not be attempted."""
    if tbody is None:
        return None, None, "no_body"
    if colname_to_colnum is None or not n_cols:
        return None, None, "no_colspec"

    raw_rows = tbody.findall("{%s}row" % NS["cals"])
    if not raw_rows:
        return None, None, "no_body"

    carry = {}  # col -> (value, source_row_index, source_col)  (pre-decrement snapshot handled via rows_remaining below)
    carry_remaining = {}  # col -> rows_remaining
    out_rows, out_prov = [], []
    any_unresolved = False

    for r_idx, row in enumerate(raw_rows):
        entries = row.findall("{%s}entry" % NS["ce"])
        carried_in = dict(carry)
        occupied = set(carried_in.keys())
        col = 1
        placements = []
        resolvable = True
        for entry in entries:
            while col in occupied:
                col += 1
            span = _entry_span(entry, colname_to_colnum, col)
            if span is None:
                resolvable = False
                break
            start, end = span
            value = clean_text(entry)
            morerows_raw = entry.get("morerows")
            try:
                extra_rows = int(morerows_raw) if morerows_raw else 0
            except ValueError:
                extra_rows = 0
            placements.append((start, end, value, extra_rows))
            occupied |= set(range(start, end + 1))
            col = end + 1

        complete = resolvable and occupied == set(range(1, n_cols + 1))
        new_carry_additions, new_carry_remaining = {}, {}

        if not complete:
            literal = [clean_text(e) for e in entries]
            out_rows.append(literal)
            out_prov.append([{"source": "unresolved", "source_row_index": r_idx, "source_col": None}
                              for _ in literal])
            any_unresolved = True
        else:
            row_values, row_prov = [None] * n_cols, [None] * n_cols
            for c, (value, source_row, source_col) in carried_in.items():
                row_values[c - 1] = value
                row_prov[c - 1] = {"source": "carried", "source_row_index": source_row, "source_col": source_col}
            for start, end, value, extra_rows in placements:
                row_values[start - 1] = value
                row_prov[start - 1] = {"source": "explicit", "source_row_index": r_idx, "source_col": start}
                for c in range(start + 1, end + 1):
                    row_values[c - 1] = None
                    row_prov[c - 1] = {"source": "spanned_empty", "source_row_index": r_idx, "source_col": start}
                if extra_rows > 0:
                    for c in range(start, end + 1):
                        new_carry_additions[c] = (value, r_idx, start)
                        new_carry_remaining[c] = extra_rows
            out_rows.append(row_values)
            out_prov.append(row_prov)

        carry_remaining = {c: rem - 1 for c, rem in carry_remaining.items() if rem - 1 > 0}
        carry = {c: v for c, v in carry.items() if c in carry_remaining}
        carry.update(new_carry_additions)
        carry_remaining.update(new_carry_remaining)

    if not out_rows:
        return None, None, "no_body"
    return out_rows, out_prov, ("partial" if any_unresolved else "resolved")


def parse_tables(root):
    """Extract all tables: caption + effective leaf headers + rows +
    group-header metadata + alignment status. `headers` is always the
    effective one-per-column leaf header list (never a positional flatten
    of every <thead> row) so it stays safe to align against row cells."""
    tables = []
    for table in root.iter("{%s}table" % NS["ce"]):
        caption_el = table.find("ce:caption", NS)
        caption = clean_text(caption_el) if caption_el is not None else ""

        headers, header_groups, alignment_status = [], [], "no_header"
        thead = table.find(".//{%s}thead" % NS["cals"])
        if thead is not None:
            headers, header_groups, alignment_status = _build_header_grid(table, thead)

        colname_to_colnum, n_cols = _parse_colspecs(table)
        tbody = table.find(".//{%s}tbody" % NS["cals"])

        grid_rows, row_provenance, tbody_status = _build_tbody_grid(tbody, colname_to_colnum, n_cols)

        if grid_rows is not None:
            rows = [r for r in grid_rows if any(c for c in r)]
            row_provenance = [p for r, p in zip(grid_rows, row_provenance) if any(c for c in r)]
        else:
            # fall back to the original literal extraction (no colspec /
            # no tbody at all -- nothing to reconstruct against)
            rows, row_provenance = [], []
            if tbody is not None:
                for row in tbody.findall("{%s}row" % NS["cals"]):
                    cells = [clean_text(e) for e in row.iter("{%s}entry" % NS["ce"])]
                    if any(cells):
                        rows.append(cells)
                        row_provenance.append([{"source": "explicit", "source_row_index": None, "source_col": None}
                                                for _ in cells])
            tbody_status = "no_colspec" if tbody is not None else "no_body"

        tables.append({
            "caption": caption,
            "headers": headers,
            "rows":    rows,
            "header_groups": header_groups,
            "table_alignment_status": alignment_status,
            "row_provenance": row_provenance,
            "tbody_alignment_status": tbody_status,
        })
    return tables


def parse_sections(root):
    """Extract sections with their heading and paragraphs."""
    sections = []
    for section in root.iter("{%s}section" % NS["ce"]):
        title_el = section.find("ce:section-title", NS)
        heading = clean_text(title_el) if title_el is not None else ""

        paragraphs = []
        for para in section.findall("ce:para", NS):
            text = clean_text(para)
            if text:
                paragraphs.append(text)

        if heading or paragraphs:
            sections.append({
                "heading":    heading,
                "paragraphs": paragraphs,
            })
    return sections


def parse_xml(xml_path):
    """Parse a single Elsevier XML file and return a dict."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # coredata is in svapi namespace so direct path fails — use recursive search instead
    doi      = root.findtext(".//{%s}doi"             % NS["prism"]) or ""
    title    = root.findtext(".//{%s}title"           % NS["dc"])    or ""
    abstract = root.findtext(".//{%s}description"     % NS["dc"])    or ""
    journal  = root.findtext(".//{%s}publicationName" % NS["prism"]) or ""
    year     = root.findtext(".//{%s}coverDate"       % NS["prism"]) or ""

    # sections and tables live inside <xocs:doc> which is nested — need .//' for recursive search
    original_text = root.find(".//{%s}doc" % NS["xocs"])

    sections = parse_sections(original_text) if original_text is not None else []
    tables   = parse_tables(original_text)   if original_text is not None else []

    return {
        "doi":      doi.strip(),
        "title":    title.strip(),
        "abstract": abstract.strip(),
        "journal":  journal.strip(),
        "year":     year[:4] if year else "",
        "sections": sections,
        "tables":   tables,
    }


def main():
    # process all XML files in data/raw/
    xml_files = list(Path(INPUT_DIR).glob("*.xml"))
    print(f"Found {len(xml_files)} XML files\n")

    for xml_path in xml_files:
        out_path = Path(OUTPUT_DIR) / xml_path.with_suffix(".json").name
        if out_path.exists():
            continue  # already parsed
        print(f"Parsing: {xml_path.name}")
        try:
            data = parse_xml(xml_path)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"  sections: {len(data['sections'])}  tables: {len(data['tables'])}  → {out_path.name}")
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nDone — JSON files saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
