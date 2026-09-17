"""
benchmark_stage2_v2.py — external, single-paper benchmark runner for Stage 2
v2 (literature_pipeline/v2/evidence/evidence_prep_v2.py).

Purpose: pure experimental instrumentation around the EXISTING, unmodified
Stage 2 v2 pipeline, to produce timing/attempt-level diagnostics once real
GPU/HPC access is available. This file does not implement, redesign, or
change any part of Stage 2 v2 — it only observes it.

── What this file does NOT do ────────────────────────────────────────────────
  - No change to evidence_prep_v2.py, Stage 1, Stage 2 v1, or Stage 3.
  - No prompt changes, no cap changes, no retry-policy changes, no timeout
    changes.
  - No reconstruction of Stage 2 v2's own accounting/material statistics —
    `stage2_v2_diagnostics` is copied verbatim from the real output into the
    benchmark JSON, never recomputed here.
  - No batching, no parallel/multi-paper execution, no material caps, no
    circuit breakers, no GPU-specific code, no production-corpus tooling.
    main() processes exactly one paper per invocation.

── Design: how the call_llm monkeypatch observes real retry attempts ────────
call_llm_with_retry() (in evidence_prep_v2.py) calls the module-level name
`call_llm` by NAME inside its retry loop — Python resolves that name from
the module's globals at each call, not from a reference captured when
call_llm_with_retry was defined. So reassigning
`evidence_prep_v2.call_llm = <wrapper>` here is observed on every actual
attempt the retry loop makes (up to LLM_MAX_ATTEMPTS), not just the first.
The wrapper is a pure pass-through: it calls the ORIGINAL call_llm, times
it, classifies the outcome for logging, and returns/re-raises EXACTLY what
the original did — nothing about the returned value or raised exception is
altered, and no additional retry is introduced here (call_llm_with_retry's
own, unmodified retry loop is what produces up to two attempts; this file
only observes them).

The one read-only exception: on a successful call, the wrapper ALSO peeks
at the returned text with evidence_prep_v2.parse_llm_response() purely to
log whether that specific attempt's text was valid JSON. This peek's
result is used ONLY for the log entry — it is never returned, never
substituted, never used to decide whether to retry, reject, or otherwise
touch what flows back into the real pipeline. This matters because
"malformed JSON" is actually decided one layer ABOVE call_llm/call_llm_
with_retry, inside process_paper_v2 itself (after a successful call_llm_
with_retry return) — and does NOT trigger a retry there. Without this
peek, a malformed-JSON response would be indistinguishable from a clean
success at the per-attempt level.

── logical_calls vs. llm_attempts ────────────────────────────────────────────
A "logical call" is one of the three call sites process_paper_v2() invokes
(material discovery once, one material_grouping call per material, at most
one recovery call). Each logical call may contain up to LLM_MAX_ATTEMPTS
(currently 2) real "llm_attempts" because of call_llm_with_retry's own
retry loop. Both are logged, cross-referenced by logical_call_seq, so it is
always possible to see "logical call #3 (material_grouping, M2) took 2
attempts: attempt 1 timed out, attempt 2 succeeded."

── NOT SAFE FOR PARALLEL EXECUTION ───────────────────────────────────────────
The instrumentation below uses a mutable "current logical call" context
(logical_call_seq / call_type / material_id / attempt-in-call counter) that
is set immediately before each call_llm_with_retry invocation and read
inside the monkeypatched call_llm. This is only correct because
process_paper_v2() calls things strictly sequentially, and this runner
processes one paper at a time in one thread. It would race under any
concurrent/parallel/multi-threaded use — not addressed here, out of scope
for this preparation step.
"""

import argparse
import json
import time
import traceback
from pathlib import Path

from literature_pipeline.v2.evidence import evidence_prep_v2 as e2v2

OUT_DIR = Path("data/evidence_v2/stage2_v2_runs/benchmark")
DIAG_DIR = Path("data/evidence_v2/stage2_v2_runs/benchmark/diagnostics")


def _classify_call_llm_exception(e: Exception) -> str:
    """Diagnostic-only classification of a genuine call_llm-level failure,
    from the exact (unmodified) LLMCallError message text call_llm raises.
    Order matters: the more specific "response shape" messages are checked
    before the generic HTTP-status check, since the former also happen to
    contain the substring "HTTP"."""
    msg = str(e)
    if "hard wall-clock timeout" in msg:
        return "timeout"
    if "non-JSON HTTP response" in msg or "no 'response' field" in msg or "'response' field was not a string" in msg:
        return "bad_http_response"
    if "request failed" in msg:
        return "request_exception"
    if "HTTP " in msg and ": " in msg:
        return "http_error_status"
    return "other_error"


class Stage2V2BenchmarkInstrumentation:
    """One instance per paper (created fresh in run_benchmark_for_paper, not
    reused across papers) — pure logging state, never consulted by any
    Stage 2 v2 decision logic. install()/restore() are the only points that
    touch evidence_prep_v2's module state, and restore() is always called
    from a finally block by the caller."""

    def __init__(self):
        self.attempt_log: list[dict] = []
        self.logical_call_log: list[dict] = []
        self._global_attempt_seq = 0
        self._global_logical_seq = 0
        self._current_logical_call_seq = None
        self._current_call_type = None
        self._current_material_id = None
        self._attempt_in_call = 0
        self._original_call_llm = None

    def install(self):
        self._original_call_llm = e2v2.call_llm
        e2v2.call_llm = self._instrumented_call_llm

    def restore(self):
        if self._original_call_llm is not None:
            e2v2.call_llm = self._original_call_llm
            self._original_call_llm = None

    def _instrumented_call_llm(self, prompt: str) -> str:
        """Pure pass-through wrapper: calls the ORIGINAL call_llm, times it,
        logs the outcome, and returns/re-raises exactly what the original
        did. No retry, no timeout change, no prompt change happens here."""
        self._global_attempt_seq += 1
        self._attempt_in_call += 1
        global_attempt_seq = self._global_attempt_seq
        attempt_number_in_call = self._attempt_in_call
        t0 = time.time()

        try:
            text = self._original_call_llm(prompt)
        except e2v2.LLMCallError as e:
            duration_s = time.time() - t0
            self.attempt_log.append({
                "global_attempt_seq": global_attempt_seq,
                "logical_call_seq": self._current_logical_call_seq,
                "call_type": self._current_call_type,
                "material_id": self._current_material_id,
                "attempt_number_in_call": attempt_number_in_call,
                "duration_s": round(duration_s, 3),
                "outcome": _classify_call_llm_exception(e),
                "error_message": str(e),
                "prompt_chars": len(prompt),
                "response_chars": None,
            })
            raise  # exact original exception, unmodified

        duration_s = time.time() - t0
        # DIAGNOSTIC-ONLY peek: parse_llm_response() here never influences
        # what is returned below — `text` is returned completely unchanged
        # regardless of what this peek finds.
        json_parseable = e2v2.parse_llm_response(text) is not None
        self.attempt_log.append({
            "global_attempt_seq": global_attempt_seq,
            "logical_call_seq": self._current_logical_call_seq,
            "call_type": self._current_call_type,
            "material_id": self._current_material_id,
            "attempt_number_in_call": attempt_number_in_call,
            "duration_s": round(duration_s, 3),
            "outcome": "success_parseable" if json_parseable else "success_but_malformed_json",
            "error_message": None,
            "prompt_chars": len(prompt),
            "response_chars": len(text),
        })
        return text  # exact original text, unmodified

    def begin_logical_call(self, call_type: str, material_id: str | None = None) -> dict:
        self._global_logical_seq += 1
        self._current_logical_call_seq = self._global_logical_seq
        self._current_call_type = call_type
        self._current_material_id = material_id
        self._attempt_in_call = 0
        return {"logical_call_seq": self._current_logical_call_seq, "t0": time.time()}

    def end_logical_call(self, handle: dict, outcome: str, error_message: str | None = None):
        self.logical_call_log.append({
            "logical_call_seq": handle["logical_call_seq"],
            "call_type": self._current_call_type,
            "material_id": self._current_material_id,
            "n_attempts": self._attempt_in_call,
            "duration_s": round(time.time() - handle["t0"], 3),
            "outcome": outcome,  # "ok" | "all_attempts_failed"
            "error_message": error_message,
        })
        self._current_logical_call_seq = None
        self._current_call_type = None
        self._current_material_id = None


def make_call_functions(instrumentation: Stage2V2BenchmarkInstrumentation):
    """Builds the three injected call functions process_paper_v2() expects.
    Each one calls the SAME e2v2.build_*_prompt(...) + e2v2.call_llm_with_
    retry(...) a real (future) caller would use — the only addition is
    timing the logical call and recording whether it ultimately succeeded
    or exhausted its retries, via the unmodified exception that call_llm_
    with_retry already raises."""

    def material_discovery_call_fn(rendered_text: str) -> str:
        handle = instrumentation.begin_logical_call("material_discovery")
        prompt = e2v2.build_material_discovery_prompt(rendered_text)
        try:
            text = e2v2.call_llm_with_retry(prompt)
        except e2v2.LLMCallError as e:
            instrumentation.end_logical_call(handle, "all_attempts_failed", str(e))
            raise
        instrumentation.end_logical_call(handle, "ok")
        return text

    def grouping_call_fn(material_id: str, rendered_text: str) -> str:
        handle = instrumentation.begin_logical_call("material_grouping", material_id)
        prompt = e2v2.build_grouping_prompt(material_id, rendered_text)
        try:
            text = e2v2.call_llm_with_retry(prompt)
        except e2v2.LLMCallError as e:
            instrumentation.end_logical_call(handle, "all_attempts_failed", str(e))
            raise
        instrumentation.end_logical_call(handle, "ok")
        return text

    def recovery_call_fn(rendered_text: str, materials_summary: str = "") -> str:
        # process_paper_v2 currently calls this as recovery_call_fn(rendered_text)
        # only (no materials_summary argument) — the default here matches
        # build_recovery_prompt's own default, so behavior is identical
        # either way this is invoked.
        handle = instrumentation.begin_logical_call("recovery")
        prompt = e2v2.build_recovery_prompt(rendered_text, materials_summary)
        try:
            text = e2v2.call_llm_with_retry(prompt)
        except e2v2.LLMCallError as e:
            instrumentation.end_logical_call(handle, "all_attempts_failed", str(e))
            raise
        instrumentation.end_logical_call(handle, "ok")
        return text

    return material_discovery_call_fn, grouping_call_fn, recovery_call_fn


def run_benchmark_for_paper(paper_path: Path, out_dir: Path = OUT_DIR, diag_dir: Path = DIAG_DIR) -> dict:
    """Runs Stage 2 v2 on exactly ONE paper, with instrumentation installed
    only for the duration of this call. Always restores call_llm (finally),
    always writes both output files before returning (or raising), so a
    crash on this paper never loses anything from a paper processed earlier
    in a sequential loop a caller might build around this function later."""
    paper = json.loads(paper_path.read_text())
    doi = paper.get("doi", paper_path.stem)
    filename = paper_path.name

    instrumentation = Stage2V2BenchmarkInstrumentation()
    material_discovery_call_fn, grouping_call_fn, recovery_call_fn = make_call_functions(instrumentation)

    result = None
    run_error = None
    instrumentation.install()
    paper_t0 = time.time()
    try:
        result = e2v2.process_paper_v2(paper, doi, material_discovery_call_fn, grouping_call_fn, recovery_call_fn)
    except Exception as exc:
        run_error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    finally:
        paper_total_s = time.time() - paper_t0
        instrumentation.restore()

    total_logical_call_s = sum(e["duration_s"] for e in instrumentation.logical_call_log)
    total_attempt_s = sum(e["duration_s"] for e in instrumentation.attempt_log)

    benchmark = {
        "paper_identifier": {"doi": doi, "filename": filename},
        "total_paper_wallclock_s": round(paper_total_s, 3),
        "total_logical_calls": len(instrumentation.logical_call_log),
        "total_llm_attempts": len(instrumentation.attempt_log),
        "total_logical_call_wallclock_s": round(total_logical_call_s, 3),
        "total_llm_attempt_wallclock_s": round(total_attempt_s, 3),
        "note_on_totals": ("these three totals are NOT expected to be equal — rendering, prompt "
                            "assembly, and other runner/Python overhead outside the timed regions "
                            "all contribute to the gap between them"),
        "logical_calls": instrumentation.logical_call_log,
        "llm_attempts": instrumentation.attempt_log,
        "run_error": run_error,
        "stage2_v2_diagnostics": result.get("stage2_v2_diagnostics") if result is not None else None,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)
    if result is not None:
        (out_dir / filename).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    (diag_dir / f"{paper_path.stem}_benchmark.json").write_text(json.dumps(benchmark, indent=2, ensure_ascii=False))

    return benchmark


def print_console_summary(benchmark: dict) -> None:
    pid = benchmark["paper_identifier"]
    print(f"=== {pid['doi']} ({pid['filename']}) ===")
    print(f"  total paper wall-clock:        {benchmark['total_paper_wallclock_s']}s")
    print(f"  logical calls:                 {benchmark['total_logical_calls']}  "
          f"(wall-clock {benchmark['total_logical_call_wallclock_s']}s)")
    print(f"  actual LLM attempts:           {benchmark['total_llm_attempts']}  "
          f"(wall-clock {benchmark['total_llm_attempt_wallclock_s']}s)")
    if benchmark["run_error"]:
        print(f"  RUN ERROR: {benchmark['run_error'].splitlines()[0]}")
    diag = benchmark.get("stage2_v2_diagnostics")
    if diag:
        print(f"  n_materials_from_call1={diag['n_materials_from_call1']}  n_materials_final={diag['n_materials_final']}")
        print(f"  recovery: triggered={diag['recovery']['triggered']}  recovered_count={diag['recovery']['recovered_count']}")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2 v2 single-paper benchmark runner (sequential, single-threaded only — see module docstring).")
    parser.add_argument("paper_path", type=Path, help="Path to exactly one data/processed/<doi>.json file")
    args = parser.parse_args()

    benchmark = run_benchmark_for_paper(args.paper_path)
    print_console_summary(benchmark)


if __name__ == "__main__":
    main()
