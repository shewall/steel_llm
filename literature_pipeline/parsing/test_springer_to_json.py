"""
test_springer_to_json.py — plain-assert style tests for JATS/HTML table
header parsing (colspan/rowspan grid reconstruction). Synthetic XML
fixtures + one real Springer XML file known to previously fail.
No LLM calls, no network.
"""
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from literature_pipeline.parsing.springer_to_json import (  # noqa: E402
    parse_jats_xml, _jats_build_header_grid, _jats_build_tbody_grid, get_text,
)

FAILURES = []


def check(label, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {label}" + (f" — {detail}" if detail and status == "FAIL" else ""))
    if not cond:
        FAILURES.append(label)


def _old_buggy_flatten(thead_xml: str) -> list:
    """The ORIGINAL implementation being replaced, kept here only to
    demonstrate the pre-fix failure mode (this is not used anywhere in
    production code)."""
    thead = ET.fromstring(f"<thead>{thead_xml}</thead>")
    headers = []
    for row in thead.iter("tr"):
        for cell in row:
            if cell.tag in ("th", "td"):
                headers.append(get_text(cell))
    return headers


def _thead(inner: str):
    return ET.fromstring(f"<thead>{inner}</thead>")


# ─────────────────────────────────────────────────────────────────────────
# TEST A — two-row header with colspan
# ─────────────────────────────────────────────────────────────────────────
test_a = """
<tr><th></th><th colspan="2">GA</th><th colspan="2">WA</th></tr>
<tr><th>Grade</th><th>AB</th><th>A</th><th>AB</th><th>A</th></tr>
"""
old_a = _old_buggy_flatten(test_a)
check("TEST A pre-fix: old flatten produces wrong header count (8 for 5 leaf columns)",
      len(old_a) == 8, detail=str(old_a))

headers_a, groups_a, status_a = _jats_build_header_grid(_thead(test_a))
check("TEST A: exactly 5 effective leaf headers", len(headers_a) == 5, detail=str(headers_a))
check("TEST A: leaf headers are the true leaf row", headers_a == ["Grade", "AB", "A", "AB", "A"], detail=str(headers_a))
check("TEST A: alignment resolved", status_a == "resolved")
check("TEST A: group metadata recoverable", any(g["label"] == "GA" for g in groups_a) and any(g["label"] == "WA" for g in groups_a),
      detail=str(groups_a))


# ─────────────────────────────────────────────────────────────────────────
# TEST B — rowspan=2 stub/header cell (mirrors the real s00170 case)
# ─────────────────────────────────────────────────────────────────────────
test_b = """
<tr><th rowspan="2">Mechanical property</th><th colspan="2">GA</th><th colspan="2">WA</th></tr>
<tr><th>AB</th><th>A</th><th>AB</th><th>A</th></tr>
"""
old_b = _old_buggy_flatten(test_b)
check("TEST B pre-fix: old flatten produces wrong header count (7 for 5 leaf columns)",
      len(old_b) == 7, detail=str(old_b))

headers_b, groups_b, status_b = _jats_build_header_grid(_thead(test_b))
check("TEST B: exactly 5 effective leaf headers", len(headers_b) == 5, detail=str(headers_b))
check("TEST B: rowspan stub cell correctly fills column 1 only once",
      headers_b == ["Mechanical property", "AB", "A", "AB", "A"], detail=str(headers_b))
check("TEST B: alignment resolved", status_b == "resolved")


# ─────────────────────────────────────────────────────────────────────────
# TEST C — combined rowspan + colspan
# ─────────────────────────────────────────────────────────────────────────
test_c = """
<tr><th rowspan="2" colspan="2">Specimen ID</th><th colspan="4">Tensile properties</th></tr>
<tr><th>YS</th><th>UTS</th><th>UE</th><th>TE</th></tr>
"""
headers_c, groups_c, status_c = _jats_build_header_grid(_thead(test_c))
check("TEST C: exactly 6 effective leaf headers (2-wide rowspan+colspan stub + 4 leaf)",
      len(headers_c) == 6, detail=str(headers_c))
check("TEST C: leaf headers correct", headers_c == ["Specimen ID", None, "YS", "UTS", "UE", "TE"], detail=str(headers_c))
check("TEST C: alignment resolved", status_c == "resolved")
check("TEST C: 2-column group stub preserved in header_groups",
      any(g["label"] == "Specimen ID" and g["end_col"] - g["start_col"] == 1 for g in groups_c), detail=str(groups_c))


# ─────────────────────────────────────────────────────────────────────────
# TEST D — three-level grouped header
# ─────────────────────────────────────────────────────────────────────────
test_d = """
<tr><th></th><th colspan="4">Mechanical properties</th></tr>
<tr><th></th><th colspan="2">Macroscale</th><th colspan="2">Microscale</th></tr>
<tr><th>Grade</th><th>YS</th><th>UTS</th><th>Hv</th><th>CRSS</th></tr>
"""
headers_d, groups_d, status_d = _jats_build_header_grid(_thead(test_d))
check("TEST D: exactly 5 effective leaf headers (deepest leaf-complete row)",
      len(headers_d) == 5, detail=str(headers_d))
check("TEST D: leaf headers are the deepest row, not the intermediate ones",
      headers_d == ["Grade", "YS", "UTS", "Hv", "CRSS"], detail=str(headers_d))
check("TEST D: both intermediate hierarchy levels preserved as group metadata",
      any(g["label"] == "Mechanical properties" for g in groups_d) and
      any(g["label"] == "Macroscale" for g in groups_d) and
      any(g["label"] == "Microscale" for g in groups_d), detail=str(groups_d))


# ─────────────────────────────────────────────────────────────────────────
# TEST E — incomplete/ambiguous header grid -> unresolved, never guessed
# ─────────────────────────────────────────────────────────────────────────
test_e = """
<tr><th colspan="3">Group A</th></tr>
<tr><th>Grade</th><th>YS</th></tr>
"""
headers_e, groups_e, status_e = _jats_build_header_grid(_thead(test_e))
check("TEST E: alignment explicitly unresolved (declared colspan=3 group but only 2 leaf cells follow)",
      status_e == "unresolved", detail=f"status={status_e} headers={headers_e}")
check("TEST E: parser does not guess/invent a 3rd leaf header", headers_e == [], detail=str(headers_e))


# ─────────────────────────────────────────────────────────────────────────
# TEST F — header/row mismatch must never be silently positional-shifted
# (this is the same downstream safeguard already added for Elsevier —
# verifying it also protects Springer-sourced tables, since both feed the
# same evidence_prep_v2._build_table_candidates)
# ─────────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v2" / "evidence"))
from literature_pipeline.v2.evidence import evidence_prep_v2 as e2v2  # noqa: E402

SPRINGER_MISMATCH_PAPER = {
    "doi": "10.9999/springer-mismatch",
    "tables": [
        {"caption": "Table 1", "headers": ["Grade", "YS [MPa]", "UTS [MPa]"],
         "rows": [["X1", "530"]]},  # 3 headers, only 2 cells -- genuinely ragged tbody row
    ],
    "sections": [],
}
mismatch_inv = e2v2.build_candidate_inventory(SPRINGER_MISMATCH_PAPER)
check("TEST F: mismatched Springer-shaped row flagged, not positionally shifted",
      mismatch_inv["id_index"]["T0:R0"]["row_header_alignment"] == "table_header_row_mismatch")
check("TEST F: mismatched row produces zero cell candidates",
      not any(rec["kind"] == "table_cell" and rec["row_id"] == "T0:R0" for rec in mismatch_inv["id_index"].values()))


# ─────────────────────────────────────────────────────────────────────────
# TBODY GRID RECONSTRUCTION (JATS: rowspan / colspan)
# ─────────────────────────────────────────────────────────────────────────

REAL_XML = Path(__file__).resolve().parents[2] / "data" / "raw" / "springer" / "10-1007_s00170-026-17487-x.xml"


def _tbody_el(inner: str):
    return ET.fromstring(f"<tbody>{inner}</tbody>")


# B1 — rowspan-only
b1 = '<tr><td rowspan="2">Ferrite A</td><td>0.50</td><td>277</td></tr><tr><td>0.48</td><td></td></tr>'
rows_b1, prov_b1, status_b1 = _jats_build_tbody_grid(_tbody_el(b1), 3)
check("B1: tbody status resolved", status_b1 == "resolved", detail=str(status_b1))
check("B1: carried label fills correct column in row 1", rows_b1 is not None and rows_b1[1][0] == "Ferrite A", detail=str(rows_b1))
check("B1: carried cell marked 'carried'", prov_b1 is not None and prov_b1[1][0]["source"] == "carried")

# B2 — colspan-only (mirrors the real "Wrought" 30±10 spanning both grain-size columns)
b2 = '<tr><td>Wrought</td><td>459 ± 8</td><td colspan="2">30 ± 10</td></tr>'
rows_b2, prov_b2, status_b2 = _jats_build_tbody_grid(_tbody_el(b2), 4)
check("B2: tbody status resolved", status_b2 == "resolved", detail=str(status_b2))
check("B2: spanning value recorded once at its first column",
      rows_b2 is not None and rows_b2[0][2] == "30 ± 10", detail=str(rows_b2))
check("B2: second spanned column is honestly empty, not a fabricated repeat",
      rows_b2 is not None and rows_b2[0][3] is None, detail=str(rows_b2))
check("B2: spanned-empty column carries its own provenance marker",
      prov_b2 is not None and prov_b2[0][3]["source"] == "spanned_empty")

# B3 — combined rowspan + colspan
b3 = '<tr><td rowspan="2">Group</td><td colspan="2">wide note</td></tr><tr><td>100</td><td>200</td></tr>'
rows_b3, prov_b3, status_b3 = _jats_build_tbody_grid(_tbody_el(b3), 3)
check("B3: tbody status resolved (combined rowspan + colspan)", status_b3 == "resolved", detail=str(status_b3))
check("B3: carried label present in row 1", rows_b3 is not None and rows_b3[1][0] == "Group", detail=str(rows_b3))
check("B3: row1's own explicit values correct", rows_b3 is not None and rows_b3[1][1:] == ["100", "200"])

# B4 — continuation row under a rowspan (mirrors the real Ni/Co/.../N pattern)
b4 = '<tr><td rowspan="3">Chemical composition</td><td>Ni</td><td>18.82</td></tr><tr><td>Co</td><td>8.80</td></tr><tr><td>Mo</td><td>4.46</td></tr>'
rows_b4, prov_b4, status_b4 = _jats_build_tbody_grid(_tbody_el(b4), 3)
check("B4: all 3 rows resolve to full width", rows_b4 is not None and all(len(r) == 3 for r in rows_b4), detail=str(rows_b4))
check("B4: continuation rows correctly carry the group label",
      rows_b4 is not None and rows_b4[1][0] == "Chemical composition" and rows_b4[2][0] == "Chemical composition")
check("B4: continuation rows' own element/value cells stay correctly aligned",
      rows_b4 is not None and rows_b4[1][1:] == ["Co", "8.80"] and rows_b4[2][1:] == ["Mo", "4.46"])

# B5 — multiple simultaneous active rowspans (two independent columns, each
# with their own still-active rowspan at the same time)
b5 = ('<tr><td rowspan="2">GroupA</td><td rowspan="2">GroupB</td><td>x1</td></tr>'
      '<tr><td>x2</td></tr>')
rows_b5, prov_b5, status_b5 = _jats_build_tbody_grid(_tbody_el(b5), 3)
check("B5: tbody status resolved with two simultaneous active rowspans", status_b5 == "resolved", detail=str(status_b5))
check("B5: both carried columns correct in row 1",
      rows_b5 is not None and rows_b5[1][0] == "GroupA" and rows_b5[1][1] == "GroupB", detail=str(rows_b5))
check("B5: row1's own explicit value correctly placed in the remaining column",
      rows_b5 is not None and rows_b5[1][2] == "x2")

# B6 — malformed/overlapping span -> unresolved, never guessed
# (colspan=2 declared but only enough real cells to reach column 2 of 4;
# deterministic placement of the remainder is impossible)
b6 = '<tr><td colspan="2">wide</td><td>only one more cell</td></tr>'
rows_b6, prov_b6, status_b6 = _jats_build_tbody_grid(_tbody_el(b6), 4)
check("B6: incomplete row flagged unresolved/partial, not guessed",
      status_b6 == "partial" and prov_b6[0][0]["source"] == "unresolved", detail=f"status={status_b6}")
check("B6: unresolved row's literal cell text preserved, not shifted",
      rows_b6 == [["wide", "only one more cell"]], detail=str(rows_b6))

# REAL SPRINGER RESIDUAL TABLE — s00170-026-17487-x, Table 1 (rowspan groups
# + an explicit blank stub-closing cell, from the fully classified set)
REAL_TBODY_XML = REAL_XML  # same file, different table within it
if REAL_TBODY_XML.is_file():
    xml_text_tb = REAL_TBODY_XML.read_text(encoding="utf-8")
    paper_tb = parse_jats_xml(xml_text_tb)
    composition_table = next((t for t in paper_tb["tables"]
                               if "preliminary characterization of gas-atomized" in (t.get("caption") or "").lower()), None)
    check("REAL SPRINGER TBODY TABLE: composition table found", composition_table is not None)
    if composition_table:
        check("REAL SPRINGER TBODY TABLE: tbody status resolved",
              composition_table.get("tbody_alignment_status") == "resolved",
              detail=str(composition_table.get("tbody_alignment_status")))
        rows_tb = composition_table["rows"]
        check("REAL SPRINGER TBODY TABLE: all rows resolve to width 4",
              rows_tb and all(len(r) == 4 for r in rows_tb), detail=str([len(r) for r in rows_tb]))
        check("REAL SPRINGER TBODY TABLE: 'Chemical composition' label carried into Co/Mo/Ti/C/Fe/N rows",
              all(rows_tb[i][0] == rows_tb[0][0] for i in range(1, 6)), detail=str([r[0] for r in rows_tb[:7]]))
        check("REAL SPRINGER TBODY TABLE: new group label starts correctly at the Powder-morphometry row",
              "Powder morphometric" in str(rows_tb[7][0]) or "Powder morphometric" in str(rows_tb[8][0]),
              detail=str([r[0] for r in rows_tb[7:9]]))


# ─────────────────────────────────────────────────────────────────────────
# TEST G — real Springer paper known to fail: s00170-026-17487-x, Table 3
# ─────────────────────────────────────────────────────────────────────────

if REAL_XML.is_file():
    xml_text = REAL_XML.read_text(encoding="utf-8")
    paper = parse_jats_xml(xml_text)
    mech_table = None
    for t in paper["tables"]:
        if "tensile tests conducted in triplicate" in (t.get("caption") or "").lower():
            mech_table = t
            break

    check("REAL SPRINGER PAPER: mechanical-properties table (Table 3) found", mech_table is not None)
    if mech_table:
        check("REAL SPRINGER PAPER: exactly 7 effective leaf headers",
              len(mech_table["headers"]) == 7, detail=f"got {len(mech_table['headers'])}: {mech_table['headers']}")
        check("REAL SPRINGER PAPER: alignment resolved",
              mech_table.get("table_alignment_status") == "resolved", detail=str(mech_table.get("table_alignment_status")))

        rows = mech_table["rows"]
        check("REAL SPRINGER PAPER: 4 property rows (UTS, YS, elongation, HV)", len(rows) == 4, detail=str(len(rows)))

        if len(rows) == 4 and len(mech_table["headers"]) == 7:
            # headers has repeated leaf names across groups (GA/WA/VIM-VAR
            # each contribute their own "AB"/"A" sub-columns), so verify by
            # position rather than by a name->value dict, which would
            # silently collapse the duplicates.
            check("REAL SPRINGER PAPER: UTS row's GA/AB column (position 1) -> 1110 ± 14",
                  "1110" in str(rows[0][1]), detail=str(rows[0]))
            check("REAL SPRINGER PAPER: YS row's GA/AB column (position 1) -> 987 ± 13",
                  "987" in str(rows[1][1]), detail=str(rows[1]))
else:
    print(f"[SKIP] real Springer XML not found at {REAL_XML}")


print()
print("=" * 70)
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
else:
    print("ALL CHECKS PASSED")
