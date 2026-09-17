"""
run_one_paper.py — minimal, controlled end-to-end orchestrator for exactly
ONE processed paper through Stage 1 -> Stage 2 v2 -> Stage 3 -> dataset
builder.

This module CALLS the existing, already-tested stage functions. It does not
reimplement, alter, or duplicate any scientific extraction, grouping,
linkage, validation, or dataset-schema logic — it only sequences existing
calls, writes intermediate outputs to a dedicated run directory, and records
run-level provenance (model/runtime settings, timestamps, per-stage status).

Explicitly out of scope for this script (by design, not an oversight):
  - batching/looping over multiple papers
  - any concurrency
  - calling any modelling-library code (literature_pipeline/modelling/)
  - changing prompts, candidate caps, linkage rules, validation gates,
    dataset schema, evidence accounting, or provenance rules
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from literature_pipeline.v2.screening.cpp_prescreen import (  # noqa: E402
    screen_paper, combined_status, MODEL as STAGE1_MODEL,
)
from literature_pipeline.v2.evidence.benchmark_stage2_v2 import run_benchmark_for_paper  # noqa: E402
from literature_pipeline.v2.evidence import evidence_prep_v2 as e2v2  # noqa: E402
from literature_pipeline.v2.extraction.extract_observations import (  # noqa: E402
    assemble_paper_record, MODEL as STAGE3_MODEL,
)
from literature_pipeline.v2.dataset.build_observation_dataset import build_dataset, write_dataset  # noqa: E402

RUNS_ROOT = REPO_ROOT / "data" / "cloud_runs"

# Only this Stage 1 outcome means the paper is retained for extraction.
STAGE1_RETAIN_STATUS = "cpp_extractable"

# PATCH (ambiguous-routing gap fix): Stage 1's own design (see
# cpp_prescreen.py's module docstring and its "..._defer_stage2"/
# "..._needs_stage2_verification" reason codes) deliberately defers
# genuinely uncertain-but-potentially-useful papers to
# stage1_relevance == "ambiguous_relevance" rather than excluding them —
# Stage 2 is explicitly expected to be able to revise that call after
# seeing fuller evidence. Requiring combined_status == "cpp_extractable"
# alone discarded every such paper identically to a hard non_cpp exclusion,
# regardless of whether Stage 2 could actually have processed it. This set
# names the extractability values for which Stage 2 has a real extraction
# path (table/text/mixed evidence) — matches COMBINED_STATUS_MAP_EXTRACTABLE
# in cpp_prescreen.py exactly, kept as a separate constant here (not
# imported) since this is a caller-level routing POLICY, not a change to
# combined_status()'s own taxonomy or semantics.
AMBIGUOUS_RETAIN_EXTRACTABILITY = {"table_extractable", "text_extractable", "mixed_modality"}


def stage1_routing_decision(relevance: str, extractability: str, status: str) -> str:
    """Caller-level routing policy only. Does not alter combined_status()
    or the relevance/extractability taxonomies. Returns one of:
      - "retain_cpp_extractable"     — the clean, unambiguous Stage 1 pass
      - "retain_ambiguous_extractable" — Stage 1 deferred as ambiguous, but
        its own extractability signal indicates Stage 2 has a real
        extraction path (table/text/mixed evidence) — retained so Stage 2
        can do the evidence-level verification Stage 1 was designed to
        defer to it, rather than silently discarding it.
      - "exclude_stage1"             — a genuine Stage 1 exclusion (non_cpp,
        not_assessed, cpp_relevant_but_not_extractable), OR an ambiguous
        case with no real extraction path (figure_only, not_extractable_
        other, not_assessed, no_content) — Stage 2 has nothing to verify
        regardless of relevance in that case.
    ambiguous_relevance is never converted into cpp_relevant anywhere by
    this function — the saved Stage 1 result keeps its original
    classification untouched; this value is additive routing metadata only."""
    if status == STAGE1_RETAIN_STATUS:
        return "retain_cpp_extractable"
    if relevance == "ambiguous_relevance" and extractability in AMBIGUOUS_RETAIN_EXTRACTABILITY:
        return "retain_ambiguous_extractable"
    return "exclude_stage1"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ollama_version() -> str:
    try:
        out = subprocess.run(["ollama", "--version"], capture_output=True, text=True, timeout=10)
        return (out.stdout or out.stderr).strip()
    except Exception as e:
        return f"unavailable ({type(e).__name__}: {e})"


def _quantization_from_tag(model_tag: str) -> str:
    """Best-effort quantization label parsed from the model tag string
    itself (e.g. 'qwen3-next:80b-a3b-instruct-q4_K_M' -> 'q4_K_M'). Purely
    string-derived, no network/API call, so this stays available even
    during an import-only dry run."""
    if ":" not in model_tag:
        return "unknown"
    tail = model_tag.split(":", 1)[1]
    parts = tail.split("-")
    return parts[-1] if parts else "unknown"


def build_run_manifest(run_id: str, paper_path: Path) -> dict:
    return {
        "run_id": run_id,
        "input_paper_path": str(paper_path),
        "doi": None,  # filled in once Stage 1 loads the paper
        "start_time": _now_iso(),
        "end_time": None,
        "status": "started",
        "stage1_routing_decision": None,  # filled in once Stage 1 completes; see stage1_routing_decision()
        "stage_status": {"stage1": None, "stage2": None, "stage3": None, "dataset": None},
        "stage_runtime_s": {"stage1": None, "stage2": None, "stage3": None, "dataset": None},
        "model_config": {
            "stage1_model": STAGE1_MODEL,
            "stage2_model": e2v2.MODEL,
            "stage3_model": STAGE3_MODEL,
            "quantization_tag": _quantization_from_tag(STAGE3_MODEL),
            "temperature": 0.0,
            "seed": 42,
            "stage1_num_ctx": 4096,
            "stage2_num_ctx": 32768,
            "stage3_num_ctx": 32768,
        },
        "environment": {
            "python_version": platform.python_version(),
            "ollama_version": _ollama_version(),
        },
    }


def run_one_paper(paper_path: Path, dry_run: bool = False) -> dict:
    """dry_run=True builds the manifest and directory structure and returns
    immediately before Stage 1 — used for the import/wiring check only,
    never calls Ollama."""
    paper_path = Path(paper_path)
    if not paper_path.is_file():
        raise FileNotFoundError(f"processed-paper JSON not found: {paper_path}")

    run_id = f"{paper_path.stem}_{datetime.now():%Y%m%dT%H%M%SZ}"
    run_dir = RUNS_ROOT / run_id
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run directory: {run_dir}")

    stage1_dir = run_dir / "stage1"
    stage2_dir = run_dir / "stage2"
    stage2_diag_dir = run_dir / "stage2" / "_diagnostics"
    stage3_dir = run_dir / "stage3"
    dataset_dir = run_dir / "dataset"
    logs_dir = run_dir / "logs"
    for d in (stage1_dir, stage2_dir, stage2_diag_dir, stage3_dir, dataset_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    manifest = build_run_manifest(run_id, paper_path)
    manifest_path = logs_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    if dry_run:
        manifest["status"] = "dry_run_ok"
        manifest["end_time"] = _now_iso()
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        return manifest

    try:
        # ── Stage 1 ──────────────────────────────────────────────────────
        t0 = time.time()
        stage1_result = screen_paper(paper_path)
        manifest["stage_runtime_s"]["stage1"] = round(time.time() - t0, 3)
        (stage1_dir / paper_path.name).write_text(json.dumps(stage1_result, indent=2, ensure_ascii=False))

        doi = stage1_result.get("doi") or paper_path.stem
        manifest["doi"] = doi

        status = combined_status(stage1_result["stage1_relevance"], stage1_result["stage1_extractability"])
        manifest["stage_status"]["stage1"] = status

        routing_decision = stage1_routing_decision(
            stage1_result["stage1_relevance"], stage1_result["stage1_extractability"], status)
        manifest["stage1_routing_decision"] = routing_decision

        if routing_decision == "exclude_stage1":
            manifest["status"] = "excluded_at_stage1"
            manifest["end_time"] = _now_iso()
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
            return manifest

        # ── Stage 2 v2 ───────────────────────────────────────────────────
        t0 = time.time()
        benchmark = run_benchmark_for_paper(paper_path, out_dir=stage2_dir, diag_dir=stage2_diag_dir)
        manifest["stage_runtime_s"]["stage2"] = round(time.time() - t0, 3)

        stage2_output_path = stage2_dir / paper_path.name
        if benchmark.get("run_error") or not stage2_output_path.is_file():
            manifest["stage_status"]["stage2"] = "failed"
            manifest["status"] = "failed_at_stage2"
            manifest["end_time"] = _now_iso()
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
            return manifest
        manifest["stage_status"]["stage2"] = "ok"
        stage2_result = json.loads(stage2_output_path.read_text())

        # ── Stage 3 ──────────────────────────────────────────────────────
        t0 = time.time()
        paper = json.loads(paper_path.read_text())
        try:
            stage3_result = assemble_paper_record(doi, stage2_result, paper, llm_enabled=True)
        except Exception as exc:
            manifest["stage_runtime_s"]["stage3"] = round(time.time() - t0, 3)
            manifest["stage_status"]["stage3"] = f"failed: {type(exc).__name__}: {exc}"
            manifest["status"] = "failed_at_stage3"
            manifest["end_time"] = _now_iso()
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
            return manifest
        manifest["stage_runtime_s"]["stage3"] = round(time.time() - t0, 3)

        stage3_output_path = stage3_dir / paper_path.name
        stage3_output_path.write_text(json.dumps(stage3_result, indent=2, ensure_ascii=False))
        manifest["stage_status"]["stage3"] = "ok"

        # ── Dataset builder ──────────────────────────────────────────────
        t0 = time.time()
        try:
            build_result = build_dataset([stage3_output_path])
            write_dataset(build_result, out_dir=dataset_dir)
        except Exception as exc:
            manifest["stage_runtime_s"]["dataset"] = round(time.time() - t0, 3)
            manifest["stage_status"]["dataset"] = f"failed: {type(exc).__name__}: {exc}"
            manifest["status"] = "failed_at_dataset_builder"
            manifest["end_time"] = _now_iso()
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
            return manifest
        manifest["stage_runtime_s"]["dataset"] = round(time.time() - t0, 3)
        manifest["stage_status"]["dataset"] = "ok"

        manifest["status"] = "success"
        manifest["end_time"] = _now_iso()
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        return manifest

    except Exception as exc:
        manifest["status"] = f"unhandled_error: {type(exc).__name__}: {exc}"
        manifest["end_time"] = _now_iso()
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Controlled end-to-end run of exactly ONE processed paper "
                    "through Stage 1 -> Stage 2 v2 -> Stage 3 -> dataset builder.")
    parser.add_argument("paper_path", type=Path, help="Path to exactly one data/processed/<doi>.json file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build the run directory and manifest only; never calls Ollama or any stage function.")
    args = parser.parse_args()

    manifest = run_one_paper(args.paper_path, dry_run=args.dry_run)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
