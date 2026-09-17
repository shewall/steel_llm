"""
regenerate_structural_v2.py — one-shot, offline, CPU-only regeneration of
the processed corpus using the corrected (thead+tbody grid reconstruction,
index-based candidate-safe) Elsevier CALS and Springer JATS parsers.

Writes ONLY to data/processed_structural_v2/ and a manifest file there.
NEVER touches data/processed/. No network calls. No LLM calls.
"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from literature_pipeline.parsing.xml_to_json import parse_xml  # noqa: E402
from literature_pipeline.parsing.springer_to_json import parse_jats_xml  # noqa: E402

OUT_DIR = REPO_ROOT / "data" / "processed_structural_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ELSEVIER_RAW = REPO_ROOT / "data" / "raw"
SPRINGER_RAW = REPO_ROOT / "data" / "raw" / "springer"


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def table_stats(tables):
    resolved = sum(1 for t in tables if t.get("table_alignment_status") == "resolved"
                   and t.get("tbody_alignment_status") in ("resolved", "no_colspec", "no_headers_to_target", "no_body", None))
    # a stricter "fully resolved" count matching the structural-layer report's definition
    fully_resolved = 0
    unresolved = 0
    for t in tables:
        head_ok = t.get("table_alignment_status") == "resolved"
        body_status = t.get("tbody_alignment_status")
        body_ok = body_status in ("resolved",) or (
            body_status in ("no_colspec", "no_headers_to_target", "no_body")
            and (not t.get("rows") or len(set(len(r) for r in t["rows"])) <= 1
                 and (not t["rows"] or len(t["headers"]) == len(t["rows"][0])))
        )
        if head_ok and body_ok:
            fully_resolved += 1
        else:
            unresolved += 1
    return fully_resolved, unresolved


manifest = {
    "regenerated_at": datetime.now(timezone.utc).isoformat(),
    "parser_files": {
        "elsevier": {
            "path": "literature_pipeline/parsing/xml_to_json.py",
            "sha256_16": file_hash(REPO_ROOT / "literature_pipeline" / "parsing" / "xml_to_json.py"),
        },
        "springer": {
            "path": "literature_pipeline/parsing/springer_to_json.py",
            "sha256_16": file_hash(REPO_ROOT / "literature_pipeline" / "parsing" / "springer_to_json.py"),
        },
    },
    "input_counts": {},
    "output_counts": {},
    "records": [],  # one entry per input document (success or failure)
}

# ── Elsevier ─────────────────────────────────────────────────────────────
elsevier_files = sorted(ELSEVIER_RAW.glob("*.xml"))
manifest["input_counts"]["elsevier"] = len(elsevier_files)
elsevier_ok = 0

for xml_path in elsevier_files:
    out_name = xml_path.with_suffix(".json").name
    out_path = OUT_DIR / out_name
    record = {
        "publisher": "elsevier",
        "source_raw_path": str(xml_path.relative_to(REPO_ROOT)),
        "output_processed_path": None,
        "doi": None,
        "status": None,
        "failure_reason": None,
        "table_count": None,
        "resolved_tables": None,
        "unresolved_tables": None,
        "structural_warning": False,
    }
    try:
        data = parse_xml(xml_path)
    except Exception as e:
        record["status"] = "failed"
        record["failure_reason"] = f"{type(e).__name__}: {e}"
        manifest["records"].append(record)
        continue

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    fully_resolved, unresolved = table_stats(data.get("tables", []))
    record.update({
        "status": "success",
        "output_processed_path": str(out_path.relative_to(REPO_ROOT)),
        "doi": data.get("doi") or None,
        "table_count": len(data.get("tables", [])),
        "resolved_tables": fully_resolved,
        "unresolved_tables": unresolved,
        "structural_warning": unresolved > 0,
    })
    manifest["records"].append(record)
    elsevier_ok += 1

manifest["output_counts"]["elsevier_success"] = elsevier_ok
manifest["output_counts"]["elsevier_failed"] = len(elsevier_files) - elsevier_ok

# ── Springer ─────────────────────────────────────────────────────────────
springer_files = sorted(SPRINGER_RAW.glob("*.xml"))
manifest["input_counts"]["springer"] = len(springer_files)
springer_ok = 0
springer_no_article = []

for xml_path in springer_files:
    record = {
        "publisher": "springer",
        "source_raw_path": str(xml_path.relative_to(REPO_ROOT)),
        "output_processed_path": None,
        "doi": None,
        "status": None,
        "failure_reason": None,
        "table_count": None,
        "resolved_tables": None,
        "unresolved_tables": None,
        "structural_warning": False,
    }
    try:
        xml_text = xml_path.read_text(encoding="utf-8")
        data = parse_jats_xml(xml_text)
    except Exception as e:
        record["status"] = "failed"
        record["failure_reason"] = f"{type(e).__name__}: {e}"
        manifest["records"].append(record)
        continue

    if data is None:
        record["status"] = "failed"
        record["failure_reason"] = "no_article_element"
        manifest["records"].append(record)
        springer_no_article.append(str(xml_path.relative_to(REPO_ROOT)))
        continue

    safe_doi = data["doi"].replace("/", "_").replace(".", "-")
    out_name = f"springer_{safe_doi}.json"
    out_path = OUT_DIR / out_name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    fully_resolved, unresolved = table_stats(data.get("tables", []))
    record.update({
        "status": "success",
        "output_processed_path": str(out_path.relative_to(REPO_ROOT)),
        "doi": data.get("doi") or None,
        "table_count": len(data.get("tables", [])),
        "resolved_tables": fully_resolved,
        "unresolved_tables": unresolved,
        "structural_warning": unresolved > 0,
    })
    manifest["records"].append(record)
    springer_ok += 1

manifest["output_counts"]["springer_success"] = springer_ok
manifest["output_counts"]["springer_failed"] = len(springer_files) - springer_ok
manifest["output_counts"]["springer_no_article_element"] = springer_no_article
manifest["output_counts"]["total_success"] = elsevier_ok + springer_ok
manifest["output_counts"]["total_input"] = len(elsevier_files) + len(springer_files)

manifest_path = OUT_DIR / "regeneration_manifest.json"
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

print(f"Elsevier: {elsevier_ok}/{len(elsevier_files)} succeeded")
print(f"Springer: {springer_ok}/{len(springer_files)} succeeded ({len(springer_no_article)} no_article_element)")
print(f"Total: {elsevier_ok + springer_ok}/{len(elsevier_files) + len(springer_files)}")
print(f"Manifest: {manifest_path.relative_to(REPO_ROOT)}")
