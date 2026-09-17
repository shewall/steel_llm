"""
run_production_corpus.py — minimal orchestration layer that repeatedly
invokes the already-accepted, unmodified per-paper pipeline
(literature_pipeline.v2.run_one_paper) across the full processed-paper
corpus.

Explicitly out of scope (by design): any scientific logic change, any
prompt/gate/heuristic change, any retry-beyond-what-run_one_paper-already-
does for a single paper. This file only sequences existing per-paper runs,
classifies outcomes, checkpoints progress, and rebuilds the combined dataset
from preserved per-paper Stage 3 outputs.

Each paper is run in its own subprocess (python3 -m literature_pipeline.v2.
run_one_paper <path>) so that a crash, hang, or runaway memory/GPU state in
one paper can never take down the corpus loop or corrupt another paper's
run directory. A generous per-paper wall-clock timeout classifies the paper
as "timeout" and moves on.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
CLOUD_RUNS_DIR = REPO_ROOT / "data" / "cloud_runs"
PRODUCTION_MANIFEST_PATH = CLOUD_RUNS_DIR / "production_manifest.jsonl"
PRODUCTION_SUMMARY_PATH = CLOUD_RUNS_DIR / "production_summary.json"
PRODUCTION_DATASET_OUT_DIR = REPO_ROOT / "data" / "datasets_v2" / "production_run"

PER_PAPER_TIMEOUT_S = 1200  # generous ceiling; see module docstring
# Outcomes that are considered a final, non-retryable disposition for a
# paper on resume. Anything NOT in this set (infrastructure_failure,
# timeout) is re-attempted on resume, since those indicate an environment
# glitch rather than a genuine scientific/pipeline outcome for that paper.
RESUME_SKIP_OUTCOMES = {
    "completed_with_dataset_rows",
    "completed_no_usable_target",
    "excluded_stage1",
    "stage2_failure",
    "stage3_failure",
    "dataset_failure",
}

OUTCOME_CATEGORIES = [
    "completed_with_dataset_rows",
    "completed_no_usable_target",
    "excluded_stage1",
    "stage2_failure",
    "stage3_failure",
    "dataset_failure",
    "timeout",
    "infrastructure_failure",
]

SYSTEMIC_FAILURE_WINDOW = 8
SYSTEMIC_FAILURE_CATEGORIES = {"infrastructure_failure", "timeout"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def discover_papers() -> list[Path]:
    return sorted(PROCESSED_DIR.glob("*.json"))


def load_existing_manifest_entries() -> dict:
    """Returns {paper_stem: last_entry_dict} from any prior production run."""
    entries: dict[str, dict] = {}
    if not PRODUCTION_MANIFEST_PATH.is_file():
        return entries
    with PRODUCTION_MANIFEST_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            entries[rec["paper_stem"]] = rec
    return entries


def _dataset_row_count(manifest: dict) -> int:
    run_dir = CLOUD_RUNS_DIR / manifest["run_id"]
    dataset_csv = run_dir / "dataset" / "cpp_dataset_v2.csv"
    if not dataset_csv.is_file():
        return 0
    try:
        import pandas as pd
        return len(pd.read_csv(dataset_csv))
    except Exception:
        # Fallback: count non-header lines without pandas.
        try:
            lines = dataset_csv.read_text(encoding="utf-8").splitlines()
            return max(0, len(lines) - 1)
        except Exception:
            return 0


def classify_outcome(manifest: dict) -> tuple[str, str | None]:
    """Returns (outcome_category, failure_reason_or_None)."""
    status = manifest.get("status", "")
    if status == "excluded_at_stage1":
        return "excluded_stage1", None
    if status == "failed_at_stage2":
        return "stage2_failure", manifest.get("stage_status", {}).get("stage2")
    if status == "failed_at_stage3":
        return "stage3_failure", manifest.get("stage_status", {}).get("stage3")
    if status == "failed_at_dataset_builder":
        return "dataset_failure", manifest.get("stage_status", {}).get("dataset")
    if status == "success":
        rows = _dataset_row_count(manifest)
        if rows > 0:
            return "completed_with_dataset_rows", None
        return "completed_no_usable_target", None
    if status.startswith("unhandled_error"):
        lowered = status.lower()
        infra_markers = ("connection", "ollama", "timeout", "timed out", "oserror", "memory")
        if any(m in lowered for m in infra_markers):
            return "infrastructure_failure", status
        # An unhandled exception past a completed stage 1/2/3 is treated
        # per the last stage that had started, otherwise infra by default.
        if manifest.get("stage_status", {}).get("stage3") not in (None,):
            return "stage3_failure", status
        if manifest.get("stage_status", {}).get("stage2") not in (None,):
            return "stage2_failure", status
        return "infrastructure_failure", status
    return "infrastructure_failure", status or "unknown_status"


def run_single_paper_subprocess(paper_path: Path, python_bin: str) -> tuple[dict | None, str | None, float]:
    """Invokes the existing, unmodified run_one_paper CLI in a subprocess.
    Returns (manifest_or_None, error_category_or_None, runtime_s)."""
    t0 = time.time()
    try:
        proc = subprocess.run(
            [python_bin, "-m", "literature_pipeline.v2.run_one_paper", str(paper_path)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=PER_PAPER_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return None, "timeout", time.time() - t0
    runtime_s = time.time() - t0

    # The manifest run_one_paper.py writes to disk is the single authoritative
    # source regardless of subprocess returncode/stdout — it is written
    # incrementally as each stage completes (and on any handled failure),
    # so it reflects the real outcome even when something about the
    # subprocess's stdout/exit code is noisy or unreliable. Only fall back
    # to stdout parsing if no on-disk manifest can be found at all.
    candidate_dirs = sorted(
        CLOUD_RUNS_DIR.glob(f"{paper_path.stem}_*"),
        key=lambda p: p.stat().st_mtime,
    )
    if candidate_dirs:
        manifest_path = candidate_dirs[-1] / "logs" / "run_manifest.json"
        if manifest_path.is_file():
            try:
                return json.loads(manifest_path.read_text()), None, runtime_s
            except Exception:
                pass

    if proc.returncode != 0:
        return None, "infrastructure_failure", runtime_s

    try:
        manifest = json.loads(proc.stdout)
        return manifest, None, runtime_s
    except Exception:
        return None, "infrastructure_failure", runtime_s


def append_manifest_entry(entry: dict) -> None:
    CLOUD_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with PRODUCTION_MANIFEST_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def write_summary(summary: dict) -> None:
    PRODUCTION_SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False))


def build_summary(all_entries: dict, corpus_size: int) -> dict:
    counts = {cat: 0 for cat in OUTCOME_CATEGORIES}
    for e in all_entries.values():
        cat = e.get("outcome_category")
        if cat in counts:
            counts[cat] += 1
    processed = len(all_entries)
    return {
        "updated_at": _now_iso(),
        "corpus_size": corpus_size,
        "processed": processed,
        "remaining": max(0, corpus_size - processed),
        "outcome_counts": counts,
        "stage1_excluded": counts["excluded_stage1"],
        "stage1_retained": processed - counts["excluded_stage1"],
        "papers_with_usable_rows": counts["completed_with_dataset_rows"],
        "papers_zero_rows": counts["completed_no_usable_target"],
        "total_failures": (
            counts["stage2_failure"] + counts["stage3_failure"] + counts["dataset_failure"]
            + counts["timeout"] + counts["infrastructure_failure"]
        ),
    }


def rebuild_combined_dataset(all_entries: dict) -> dict | None:
    """Deterministically rebuilds the combined dataset from every preserved
    per-paper Stage 3 output recorded in the production manifest — never
    relies on any single monolithic file as the only copy of the data."""
    from literature_pipeline.v2.dataset.build_observation_dataset import build_dataset, write_dataset

    stage3_paths = []
    for e in all_entries.values():
        if e.get("outcome_category") not in ("completed_with_dataset_rows", "completed_no_usable_target"):
            continue
        run_dir = CLOUD_RUNS_DIR / e["run_id"]
        stage3_files = list((run_dir / "stage3").glob("*.json"))
        stage3_paths.extend(stage3_files)

    if not stage3_paths:
        return None

    build_result = build_dataset(stage3_paths)
    return write_dataset(build_result, out_dir=PRODUCTION_DATASET_OUT_DIR)


def main():
    parser = argparse.ArgumentParser(description="Run the full processed-paper corpus through the accepted pipeline.")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N un-skipped papers (for smoke testing).")
    parser.add_argument("--rebuild-dataset-every", type=int, default=25, help="Rebuild the combined dataset every N newly processed papers.")
    args = parser.parse_args()

    papers = discover_papers()
    corpus_size = len(papers)
    all_entries = load_existing_manifest_entries()

    print(f"[production] corpus_size={corpus_size} already_recorded={len(all_entries)}", flush=True)

    processed_this_run = 0
    recent_outcomes: list[str] = []
    systemic_stop = False

    for paper_path in papers:
        stem = paper_path.stem
        existing = all_entries.get(stem)
        if existing and existing.get("outcome_category") in RESUME_SKIP_OUTCOMES:
            continue

        if args.limit is not None and processed_this_run >= args.limit:
            break

        start_time = _now_iso()
        manifest, error_category, runtime_s = run_single_paper_subprocess(paper_path, args.python_bin)

        if manifest is None:
            outcome_category = error_category or "infrastructure_failure"
            failure_reason = f"subprocess_{outcome_category}"
            run_id = None
        else:
            outcome_category, failure_reason = classify_outcome(manifest)
            run_id = manifest.get("run_id")

        entry = {
            "paper_stem": stem,
            "paper_path": str(paper_path),
            "run_id": run_id,
            "outcome_category": outcome_category,
            "failure_reason": failure_reason,
            "stage1_routing_decision": (manifest or {}).get("stage1_routing_decision"),
            "stage_status": (manifest or {}).get("stage_status"),
            "start_time": start_time,
            "end_time": _now_iso(),
            "runtime_s": round(runtime_s, 3),
        }
        if outcome_category == "completed_with_dataset_rows":
            entry["dataset_row_count"] = _dataset_row_count(manifest)

        append_manifest_entry(entry)
        all_entries[stem] = entry
        processed_this_run += 1

        print(f"[production] {processed_this_run} {stem} -> {outcome_category} ({runtime_s:.1f}s)", flush=True)

        recent_outcomes.append(outcome_category)
        if len(recent_outcomes) > SYSTEMIC_FAILURE_WINDOW:
            recent_outcomes.pop(0)
        if (
            len(recent_outcomes) == SYSTEMIC_FAILURE_WINDOW
            and all(o in SYSTEMIC_FAILURE_CATEGORIES for o in recent_outcomes)
        ):
            systemic_stop = True
            print(f"[production] SYSTEMIC FAILURE DETECTED: last {SYSTEMIC_FAILURE_WINDOW} papers all "
                  f"infrastructure_failure/timeout. Stopping corpus run for investigation.", flush=True)

        summary = build_summary(all_entries, corpus_size)
        write_summary(summary)

        if processed_this_run % args.rebuild_dataset_every == 0:
            rebuild_combined_dataset(all_entries)

        if systemic_stop:
            break

    final_paths = rebuild_combined_dataset(all_entries)
    summary = build_summary(all_entries, corpus_size)
    summary["systemic_stop"] = systemic_stop
    summary["combined_dataset_paths"] = {k: str(v) for k, v in (final_paths or {}).items()}
    write_summary(summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
