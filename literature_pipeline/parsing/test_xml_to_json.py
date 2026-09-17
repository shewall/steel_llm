"""
test_xml_to_json.py — plain-assert style tests for CALS table header
parsing. Synthetic XML fixtures + the real Acta Materialia XML file.
No LLM calls, no network.
"""
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from literature_pipeline.parsing.xml_to_json import parse_tables, NS, _parse_colspecs, _build_tbody_grid  # noqa: E402

FAILURES = []


def check(label, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {label}" + (f" — {detail}" if detail and status == "FAIL" else ""))
    if not cond:
        FAILURES.append(label)


def _wrap(tgroup_xml: str) -> ET.Element:
    """Wraps a <tgroup> fragment in the minimal ce:table/xocs:doc structure
    parse_tables() expects, with the real Elsevier namespaces."""
    xml = f"""<xocs:doc xmlns:xocs="{NS['xocs']}" xmlns:ce="{NS['ce']}">
      <ce:table xmlns="{NS['cals']}" id="tbl_test">
        <ce:caption><ce:simple-para xmlns="{NS['ce']}">Test table</ce:simple-para></ce:caption>
        {tgroup_xml}
      </ce:table>
    </xocs:doc>"""
    return ET.fromstring(xml)


CE = NS["ce"]
CALS = NS["cals"]


def entry(text="", **attrs):
    attr_str = " ".join(f'{k}="{v}"' for k, v in attrs.items())
    return f'<entry xmlns="{CE}" {attr_str}>{text}</entry>'


# ─────────────────────────────────────────────────────────────────────────
# TEST A — grouped 2-row CALS header, 10 leaf columns
# ─────────────────────────────────────────────────────────────────────────
colspecs_10 = "\n".join(f'<colspec colname="col{i}" colnum="{i}"/>' for i in range(1, 11))
leaf_labels = ["", "Ferrite [MPa]", "Mart. [MPa]", "YS [MPa]", "UTS [MPa]",
               "UE [%]", "TE [%]", "Vol. M [%]", "Ferrite G.S. [µm]", "Mart. G.S. [µm]"]

test_a_xml = f"""
<tgroup cols="10">
  {colspecs_10}
  <thead>
    <row>
      {entry("", colname="col1")}
      {entry("Microscale", namest="col2", nameend="col3")}
      {entry("Macroscale", namest="col4", nameend="col7")}
      {entry("Microstructure", namest="col8", nameend="col10")}
    </row>
    <row>
      {"".join(entry(t, colname=f"col{i+1}") for i, t in enumerate(leaf_labels))}
    </row>
  </thead>
  <tbody>
    <row>
      {entry("DP1", colname="col1")}
      {entry("205", colname="col2")}
      {entry("2716", colname="col3")}
      {entry("530", colname="col4")}
      {entry("830", colname="col5")}
      {entry("8.8", colname="col6")}
      {entry("14.4", colname="col7")}
      {entry("42", colname="col8")}
      {entry("5.9", colname="col9")}
      {entry("1.1", colname="col10")}
    </row>
  </tbody>
</tgroup>
"""

root_a = _wrap(test_a_xml)
tables_a = parse_tables(root_a)
check("TEST A: exactly one table parsed", len(tables_a) == 1)
if tables_a:
    ta = tables_a[0]
    check("TEST A: exactly 10 effective leaf headers", len(ta["headers"]) == 10,
          f"got {len(ta['headers'])}: {ta['headers']}")
    check("TEST A: leaf headers match the leaf row, not the group row",
          ta["headers"] == leaf_labels, f"got {ta['headers']}")
    check("TEST A: alignment status is resolved", ta.get("table_alignment_status") == "resolved")
    check("TEST A: group metadata recoverable (Microscale/Macroscale/Microstructure present)",
          {g["label"] for g in ta.get("header_groups", [])} >= {"Microscale", "Macroscale", "Microstructure"},
          f"got {ta.get('header_groups')}")


# ─────────────────────────────────────────────────────────────────────────
# TEST B — more than one hierarchical header level
# ─────────────────────────────────────────────────────────────────────────
colspecs_3 = "\n".join(f'<colspec colname="col{i}" colnum="{i}"/>' for i in range(1, 4))
test_b_xml = f"""
<tgroup cols="3">
  {colspecs_3}
  <thead>
    <row>
      {entry("", colname="col1")}
      {entry("Mechanical properties", namest="col2", nameend="col3")}
    </row>
    <row>
      {entry("Grade", colname="col1")}
      {entry("Macroscale", namest="col2", nameend="col3")}
    </row>
    <row>
      {entry("", colname="col1")}
      {entry("YS [MPa]", colname="col2")}
      {entry("UTS [MPa]", colname="col3")}
    </row>
  </thead>
  <tbody>
    <row>
      {entry("DP1", colname="col1")}
      {entry("530", colname="col2")}
      {entry("830", colname="col3")}
    </row>
  </tbody>
</tgroup>
"""
root_b = _wrap(test_b_xml)
tables_b = parse_tables(root_b)
if tables_b:
    tb = tables_b[0]
    check("TEST B: exactly 3 effective leaf headers (deepest leaf-complete row), not naive last-row text",
          tb["headers"] == ["", "YS [MPa]", "UTS [MPa]"], f"got {tb['headers']}")
    check("TEST B: alignment status is resolved", tb.get("table_alignment_status") == "resolved")
    check("TEST B: intermediate hierarchical levels preserved as group metadata (not discarded)",
          any(g["label"] == "Mechanical properties" for g in tb.get("header_groups", [])) and
          any(g["label"] == "Macroscale" for g in tb.get("header_groups", [])),
          f"got {tb.get('header_groups')}")


# ─────────────────────────────────────────────────────────────────────────
# TEST C — uncertain alignment: no colspec, more than one header row,
# no namest/nameend to disambiguate. Must NOT guess.
# ─────────────────────────────────────────────────────────────────────────
test_c_xml = """
<tgroup cols="4">
  <thead>
    <row>
      <entry xmlns="%(ce)s"></entry>
      <entry xmlns="%(ce)s">Group A</entry>
    </row>
    <row>
      <entry xmlns="%(ce)s">Grade</entry>
      <entry xmlns="%(ce)s">YS</entry>
      <entry xmlns="%(ce)s">UTS</entry>
      <entry xmlns="%(ce)s">TE</entry>
    </row>
  </thead>
  <tbody>
    <row>
      <entry xmlns="%(ce)s">DP1</entry>
      <entry xmlns="%(ce)s">530</entry>
      <entry xmlns="%(ce)s">830</entry>
      <entry xmlns="%(ce)s">14.4</entry>
    </row>
  </tbody>
</tgroup>
""" % {"ce": CE}
root_c = _wrap(test_c_xml)
tables_c = parse_tables(root_c)
if tables_c:
    tc = tables_c[0]
    check("TEST C: alignment explicitly flagged unresolved (no colspec, ambiguous multi-row header)",
          tc.get("table_alignment_status") == "unresolved", f"got {tc.get('table_alignment_status')}")
    check("TEST C: parser does NOT guess/invent leaf headers when unresolved",
          tc["headers"] == [], f"got {tc['headers']}")


# ─────────────────────────────────────────────────────────────────────────
# TEST D — the actual bug pattern: flat concatenation must be gone.
# The old buggy parser produced 14 headers for 10 row cells on this exact
# shape (TEST A's fixture). This test locks in that the count is now equal.
# ─────────────────────────────────────────────────────────────────────────
if tables_a:
    ta = tables_a[0]
    row0 = ta["rows"][0]
    check("TEST D: header/row-width now match on the grouped-header table (was 14 vs 10)",
          len(ta["headers"]) == len(row0), f"headers={len(ta['headers'])} row={len(row0)}")


# ─────────────────────────────────────────────────────────────────────────
# TEST E — colspec declared for the table, but individual <entry> elements
# carry no colname/namest/nameend at all (a real, common CALS pattern found
# in the corpus, e.g. simple composition tables). Per CALS/OASIS semantics
# these are positioned sequentially, NOT treated as unresolvable.
# ─────────────────────────────────────────────────────────────────────────
colspecs_5 = "\n".join(f'<colspec colname="col{i}" colnum="{i}"/>' for i in range(1, 6))
test_e_xml = f"""
<tgroup cols="5">
  {colspecs_5}
  <thead>
    <row>
      {entry("C")}
      {entry("Si")}
      {entry("Mn")}
      {entry("P")}
      {entry("S")}
    </row>
  </thead>
  <tbody>
    <row>
      {entry("0.13")}
      {entry("0.20")}
      {entry("1.51")}
      {entry("0.008")}
      {entry("0.002")}
    </row>
  </tbody>
</tgroup>
"""
root_e = _wrap(test_e_xml)
tables_e = parse_tables(root_e)
if tables_e:
    te = tables_e[0]
    check("TEST E: colspec present but entries unlabeled -> resolved via sequential position, not unresolved",
          te.get("table_alignment_status") == "resolved", f"got {te.get('table_alignment_status')}")
    check("TEST E: leaf headers resolved in document order",
          te["headers"] == ["C", "Si", "Mn", "P", "S"], f"got {te['headers']}")


# ─────────────────────────────────────────────────────────────────────────
# TBODY GRID RECONSTRUCTION (CALS: morerows / namest+nameend)
# ─────────────────────────────────────────────────────────────────────────

REAL_XML = Path(__file__).resolve().parents[2] / "data" / "raw" / "10-1016_j-actamat-2019-11-002.xml"


def _table_el(tgroup_xml: str) -> ET.Element:
    xml = f'<ce:table xmlns:ce="{CE}" xmlns="{CALS}" id="tbl_test">{tgroup_xml}</ce:table>'
    return ET.fromstring(xml)


def _tbody_grid_from(tgroup_xml: str):
    table_el = _table_el(tgroup_xml)
    colname_to_colnum, n_cols = _parse_colspecs(table_el)
    tbody = table_el.find(".//{%s}tbody" % CALS)
    return _build_tbody_grid(tbody, colname_to_colnum, n_cols)


# A1 — morerows-only: a leading label cell spans 2 rows
a1_xml = f"""
<tgroup cols="3">
  <colspec colname="col1" colnum="1"/><colspec colname="col2" colnum="2"/><colspec colname="col3" colnum="3"/>
  <tbody>
    <row>{entry("Ferrite A", colname="col1", morerows="1")}{entry("0.50", colname="col2")}{entry("277", colname="col3")}</row>
    <row>{entry("0.48", colname="col2")}{entry("", colname="col3")}</row>
  </tbody>
</tgroup>
"""
rows_a1, prov_a1, status_a1 = _tbody_grid_from(a1_xml)
check("A1: tbody status resolved", status_a1 == "resolved", detail=str(status_a1))
check("A1: row1 correct final width (3 cols)", rows_a1 is not None and len(rows_a1[1]) == 3, detail=str(rows_a1))
check("A1: carried cell occupies correct column (col1='Ferrite A')",
      rows_a1 is not None and rows_a1[1][0] == "Ferrite A", detail=str(rows_a1))
check("A1: carried cell is marked 'carried', not 'explicit'",
      prov_a1 is not None and prov_a1[1][0]["source"] == "carried", detail=str(prov_a1))
check("A1: row1's own real cells are marked 'explicit'",
      prov_a1 is not None and prov_a1[1][1]["source"] == "explicit")

# A2 — namest/nameend colspan-only: one body cell spans multiple columns
a2_xml = """
<tgroup cols="5">
  <colspec colname="col1" colnum="1"/><colspec colname="col2" colnum="2"/><colspec colname="col3" colnum="3"/>
  <colspec colname="col4" colnum="4"/><colspec colname="col5" colnum="5"/>
  <tbody>
    <row>
      <entry xmlns="%(ce)s" colname="col1">S690QL</entry>
      <entry xmlns="%(ce)s" namest="col2" nameend="col5">high-strain-rate test plan</entry>
    </row>
    <row>
      <entry xmlns="%(ce)s" colname="col1">Core</entry>
      <entry xmlns="%(ce)s" colname="col2">A</entry>
      <entry xmlns="%(ce)s" colname="col3">B</entry>
      <entry xmlns="%(ce)s" colname="col4">C</entry>
      <entry xmlns="%(ce)s" colname="col5">D</entry>
    </row>
  </tbody>
</tgroup>
""" % {"ce": CE}
rows_a2, prov_a2, status_a2 = _tbody_grid_from(a2_xml)
check("A2: tbody status resolved", status_a2 == "resolved", detail=str(status_a2))
check("A2: spanning cell value recorded once at its first column",
      rows_a2 is not None and rows_a2[0][1] == "high-strain-rate test plan", detail=str(rows_a2))
check("A2: spanned-but-not-first columns are honestly empty, not fabricated repeats",
      rows_a2 is not None and rows_a2[0][2:] == [None, None, None], detail=str(rows_a2))
check("A2: spanned-empty columns carry their own provenance marker",
      prov_a2 is not None and prov_a2[0][2]["source"] == "spanned_empty", detail=str(prov_a2))
check("A2: downstream row (its own real data) still aligns correctly",
      rows_a2 is not None and rows_a2[1] == ["Core", "A", "B", "C", "D"], detail=str(rows_a2))

# A3 — combined morerows + namest/nameend
a3_xml = f"""
<tgroup cols="4">
  <colspec colname="col1" colnum="1"/><colspec colname="col2" colnum="2"/>
  <colspec colname="col3" colnum="3"/><colspec colname="col4" colnum="4"/>
  <tbody>
    <row>{entry("Group", colname="col1", morerows="1")}{entry("wide note", namest="col2", nameend="col4")}</row>
    <row>{entry("100", colname="col2")}{entry("200", colname="col3")}{entry("300", colname="col4")}</row>
  </tbody>
</tgroup>
"""
rows_a3, prov_a3, status_a3 = _tbody_grid_from(a3_xml)
check("A3: tbody status resolved (combined morerows + namest/nameend)", status_a3 == "resolved", detail=str(status_a3))
check("A3: morerows-carried label present in row 1", rows_a3 is not None and rows_a3[1][0] == "Group", detail=str(rows_a3))
check("A3: row1's own explicit values correct", rows_a3 is not None and rows_a3[1][1:] == ["100", "200", "300"])

# A4 — continuation-row case: mirrors the real pillar-table pattern
# (label omitted because the previous row's cell spans downward; the
# remaining cells must still align to their correct columns)
a4_xml = f"""
<tgroup cols="5">
  <colspec colname="col1" colnum="1"/><colspec colname="col2" colnum="2"/><colspec colname="col3" colnum="3"/>
  <colspec colname="col4" colnum="4"/><colspec colname="col5" colnum="5"/>
  <tbody>
    <row>{entry("Ferrite C", colname="col1", morerows="1")}{entry("sys1", colname="col2")}{entry("0.49", colname="col3")}{entry("", colname="col4")}{entry("", colname="col5")}</row>
    <row>{entry("sys2", colname="col2")}{entry("0.49", colname="col3")}{entry("304", colname="col4")}{entry("149", colname="col5")}</row>
  </tbody>
</tgroup>
"""
rows_a4, prov_a4, status_a4 = _tbody_grid_from(a4_xml)
check("A4: continuation row resolves to correct width", rows_a4 is not None and len(rows_a4[1]) == 5, detail=str(rows_a4))
check("A4: continuation row's compressive-YS/CRSS columns align correctly (304, 149 -- not swapped)",
      rows_a4 is not None and rows_a4[1][3] == "304" and rows_a4[1][4] == "149", detail=str(rows_a4))

# A5 — ambiguous/malformed body span: deterministic placement impossible
# (a cell references a namest column that isn't declared in colspec at all)
a5_xml = """
<tgroup cols="3">
  <colspec colname="col1" colnum="1"/><colspec colname="col2" colnum="2"/><colspec colname="col3" colnum="3"/>
  <tbody>
    <row>
      <entry xmlns="%(ce)s" colname="col1">X</entry>
      <entry xmlns="%(ce)s" namest="colZ" nameend="colZ">bad ref</entry>
    </row>
  </tbody>
</tgroup>
""" % {"ce": CE}
rows_a5, prov_a5, status_a5 = _tbody_grid_from(a5_xml)
check("A5: unresolvable span -> row flagged unresolved, not guessed",
      status_a5 == "partial" and prov_a5[0][0]["source"] == "unresolved", detail=f"status={status_a5} prov={prov_a5}")
check("A5: unresolved row's literal cell text is preserved, not shifted",
      rows_a5 == [["X", "bad ref"]], detail=str(rows_a5))

# REAL ELSEVIER RESIDUAL TABLE — from the classified 27-table set
# (10-1016_j-actamat-2019-11-002.xml, table index 1: pillar analysis,
# 3 groups of morerows="2" continuation rows)
if REAL_XML.is_file():
    tree_real = ET.parse(REAL_XML)
    doc_real = tree_real.getroot().find(".//{%s}doc" % NS["xocs"])
    real_pillar_tables = parse_tables(doc_real) if doc_real is not None else []
    pillar_table = next((t for t in real_pillar_tables if "activated slip systems" in (t.get("caption") or "").lower()), None)
    check("REAL ELSEVIER TBODY TABLE: pillar table found", pillar_table is not None)
    if pillar_table:
        check("REAL ELSEVIER TBODY TABLE: tbody status resolved", pillar_table.get("tbody_alignment_status") == "resolved",
              detail=str(pillar_table.get("tbody_alignment_status")))
        check("REAL ELSEVIER TBODY TABLE: all 9 rows now width 5 (was ragged 4/5 before this fix)",
              pillar_table["rows"] and all(len(r) == 5 for r in pillar_table["rows"]),
              detail=str([len(r) for r in pillar_table["rows"]]))
        check("REAL ELSEVIER TBODY TABLE: row index 7's compressive-YS/CRSS correctly 304/149 (not the previously-mixed-up 149)",
              pillar_table["rows"][7][3] == "304" and pillar_table["rows"][7][4] == "149",
              detail=str(pillar_table["rows"][7]))
        check("REAL ELSEVIER TBODY TABLE: continuation rows' label column marked carried, not explicit",
              pillar_table["row_provenance"][1][0]["source"] == "carried",
              detail=str(pillar_table["row_provenance"][1][0]))


# ─────────────────────────────────────────────────────────────────────────
# REAL PAPER TEST — the actual Acta Materialia XML file
# ─────────────────────────────────────────────────────────────────────────
if REAL_XML.is_file():
    tree = ET.parse(REAL_XML)
    root = tree.getroot()
    doc = root.find(".//{%s}doc" % NS["xocs"])
    real_tables = parse_tables(doc) if doc is not None else []

    mech_table = None
    for t in real_tables:
        if "macro and micro scale" in (t.get("caption") or "").lower():
            mech_table = t
            break

    check("REAL PAPER: mechanical-behavior table (Table 4) found", mech_table is not None)

    if mech_table:
        check("REAL PAPER: exactly 10 effective leaf headers",
              len(mech_table["headers"]) == 10, f"got {len(mech_table['headers'])}: {mech_table['headers']}")
        check("REAL PAPER: alignment status resolved",
              mech_table.get("table_alignment_status") == "resolved",
              f"got {mech_table.get('table_alignment_status')}")

        rows = mech_table["rows"]
        check("REAL PAPER: exactly 2 data rows (DP1, DP2)", len(rows) == 2, f"got {len(rows)}")

        if len(rows) == 2 and len(mech_table["headers"]) == 10:
            headers = mech_table["headers"]
            dp1 = dict(zip(headers, rows[0]))
            dp2 = dict(zip(headers, rows[1]))

            check("REAL PAPER: DP1 YS -> 530", dp1.get("YS [MPa]") == "530", f"got {dp1.get('YS [MPa]')}")
            check("REAL PAPER: DP1 UTS -> 830", dp1.get("UTS [MPa]") == "830", f"got {dp1.get('UTS [MPa]')}")
            check("REAL PAPER: DP1 UE -> 8.8", dp1.get("UE [%]") == "8.8", f"got {dp1.get('UE [%]')}")
            check("REAL PAPER: DP1 TE -> 14.4", dp1.get("TE [%]") == "14.4", f"got {dp1.get('TE [%]')}")

            check("REAL PAPER: DP2 YS -> 494", dp2.get("YS [MPa]") == "494", f"got {dp2.get('YS [MPa]')}")
            check("REAL PAPER: DP2 UTS -> 830", dp2.get("UTS [MPa]") == "830", f"got {dp2.get('UTS [MPa]')}")
            check("REAL PAPER: DP2 UE -> 11.5", dp2.get("UE [%]") == "11.5", f"got {dp2.get('UE [%]')}")
            check("REAL PAPER: DP2 TE -> 18.6", dp2.get("TE [%]") == "18.6", f"got {dp2.get('TE [%]')}")

            check("REAL PAPER: 42±1 / 38±1 must NOT be mapped to YS",
                  "42" not in str(dp1.get("YS [MPa]")) and "38" not in str(dp2.get("YS [MPa]")),
                  f"DP1 YS={dp1.get('YS [MPa]')!r} DP2 YS={dp2.get('YS [MPa]')!r}")
            check("REAL PAPER: 5.9±0.1 / 7.0±0.1 must NOT be mapped to UTS",
                  "5.9" not in str(dp1.get("UTS [MPa]")) and "7.0" not in str(dp2.get("UTS [MPa]")),
                  f"DP1 UTS={dp1.get('UTS [MPa]')!r} DP2 UTS={dp2.get('UTS [MPa]')!r}")

            check("REAL PAPER: Vol. M -> 42 / 38",
                  dp1.get("Vol. M [%]", "").startswith("42") and dp2.get("Vol. M [%]", "").startswith("38"))
            check("REAL PAPER: Ferrite G.S. -> 5.9 / 7.0",
                  dp1.get("Ferrite G.S. [µm]", "").startswith("5.9") and dp2.get("Ferrite G.S. [µm]", "").startswith("7.0"))
            check("REAL PAPER: Mart. G.S. -> 1.1 / 1.0",
                  dp1.get("Mart. G.S. [µm]", "").startswith("1.1") and dp2.get("Mart. G.S. [µm]", "").startswith("1.0"))
else:
    print(f"[SKIP] real paper XML not found at {REAL_XML}")


print()
print("=" * 70)
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
else:
    print("ALL CHECKS PASSED")
