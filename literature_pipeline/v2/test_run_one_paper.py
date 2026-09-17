"""
test_run_one_paper.py — offline, dependency-free sanity tests for
run_one_paper.py's Stage 1 -> Stage 2 routing policy. No LLM calls, no
network access, no RunPod — screen_paper is monkeypatched with a canned
Stage 1 result for the end-to-end manifest tests (R10), and
stage1_routing_decision (a pure function) is tested directly for R1-R9.

Run: python literature_pipeline/v2/test_run_one_paper.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from literature_pipeline.v2 import run_one_paper as rop
from literature_pipeline.v2.screening.cpp_prescreen import combined_status

FAILURES = []
_ORIGINAL_SCREEN_PAPER = rop.screen_paper
_ORIGINAL_RUNS_ROOT = rop.RUNS_ROOT
_ORIGINAL_RUN_BENCHMARK = rop.run_benchmark_for_paper


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" — {detail}" if detail and status == "FAIL" else ""))
    if not condition:
        FAILURES.append(label)


def _restore():
    rop.screen_paper = _ORIGINAL_SCREEN_PAPER
    rop.RUNS_ROOT = _ORIGINAL_RUNS_ROOT
    rop.run_benchmark_for_paper = _ORIGINAL_RUN_BENCHMARK


# ─────────────────────────────────────────────────────────────────────────
# R1-R9 — stage1_routing_decision as a pure function, using the real,
# unmodified combined_status() from cpp_prescreen.py exactly as
# run_one_paper() computes it.
# ─────────────────────────────────────────────────────────────────────────

def _decision(relevance: str, extractability: str) -> str:
    status = combined_status(relevance, extractability)
    return rop.stage1_routing_decision(relevance, extractability, status)


check("R1: cpp_relevant + table_extractable -> retain_cpp_extractable",
      _decision("cpp_relevant", "table_extractable") == "retain_cpp_extractable")
check("R2: ambiguous_relevance + table_extractable -> retain_ambiguous_extractable",
      _decision("ambiguous_relevance", "table_extractable") == "retain_ambiguous_extractable")
check("R3: ambiguous_relevance + text_extractable -> retain_ambiguous_extractable",
      _decision("ambiguous_relevance", "text_extractable") == "retain_ambiguous_extractable")
check("R4: ambiguous_relevance + mixed_modality -> retain_ambiguous_extractable",
      _decision("ambiguous_relevance", "mixed_modality") == "retain_ambiguous_extractable")
check("R5: ambiguous_relevance + figure_only -> exclude_stage1",
      _decision("ambiguous_relevance", "figure_only") == "exclude_stage1")
check("R6: ambiguous_relevance + not_extractable_other -> exclude_stage1",
      _decision("ambiguous_relevance", "not_extractable_other") == "exclude_stage1")
check("R7: ambiguous_relevance + not_assessed -> exclude_stage1",
      _decision("ambiguous_relevance", "not_assessed") == "exclude_stage1")
check("R8: non_cpp + table_extractable -> exclude_stage1 (unaffected by the fix)",
      _decision("non_cpp", "table_extractable") == "exclude_stage1")
check("R9: not_assessed + not_assessed -> exclude_stage1",
      _decision("not_assessed", "not_assessed") == "exclude_stage1")
check("boundary: ambiguous_relevance + no_content -> exclude_stage1",
      _decision("ambiguous_relevance", "no_content") == "exclude_stage1")
check("sanity: cpp_relevant + figure_only -> exclude_stage1 (combined_status()'s own "
      "cpp_relevant_but_not_extractable outcome, unaffected)",
      _decision("cpp_relevant", "figure_only") == "exclude_stage1")
check("sanity: combined_status() itself never returns 'cpp_extractable' for "
      "ambiguous_relevance (combined_status() itself is unmodified)",
      combined_status("ambiguous_relevance", "table_extractable") == "ambiguous")


# ─────────────────────────────────────────────────────────────────────────
# R10 — manifest/routing metadata distinguishes clean-retained /
# ambiguous-retained / excluded, via the real run_one_paper() with
# screen_paper monkeypatched to a canned result (no Ollama, no network).
# RUNS_ROOT is redirected to a temp directory for every case below, never
# touching data/cloud_runs/.
# ─────────────────────────────────────────────────────────────────────────

def _make_paper_file(tmp_path: Path) -> Path:
    p = tmp_path / "10-9999_test-paper.json"
    p.write_text(json.dumps({"doi": "10.9999/test-paper", "title": "Test", "sections": [], "tables": []}))
    return p


with tempfile.TemporaryDirectory() as _tmp:
    _tmp_path = Path(_tmp)

    # excluded case (non_cpp): Stage 2 must never be reached. Proven by
    # making run_benchmark_for_paper raise if ever called.
    rop.screen_paper = lambda pf: {
        "doi": "10.9999/test-paper", "stage1_relevance": "non_cpp",
        "stage1_extractability": "not_assessed", "stage1_reason_code": "test_canned",
        "rule_fired": "test_canned", "decision_basis": "test_canned",
    }
    rop.RUNS_ROOT = _tmp_path / "cloud_runs_excluded"

    def _fail_if_stage2_called(*a, **k):
        raise AssertionError("Stage 2 (run_benchmark_for_paper) was called for an exclude_stage1 case")
    rop.run_benchmark_for_paper = _fail_if_stage2_called

    m_excluded = rop.run_one_paper(_make_paper_file(_tmp_path))
    check("R10: excluded case -> manifest['status'] == 'excluded_at_stage1'",
          m_excluded["status"] == "excluded_at_stage1", detail=str(m_excluded.get("status")))
    check("R10: excluded case -> manifest['stage1_routing_decision'] == 'exclude_stage1'",
          m_excluded.get("stage1_routing_decision") == "exclude_stage1",
          detail=str(m_excluded.get("stage1_routing_decision")))
    _restore()

    # ambiguous-but-extractable case: Stage 2 SHOULD be reached (this is the
    # entire point of the fix). Proven by making run_benchmark_for_paper
    # raise a distinctive error the instant it's called, and catching
    # exactly that -- any other outcome (no exception, or a different
    # exception) means routing did not correctly continue.
    rop.screen_paper = lambda pf: {
        "doi": "10.9999/test-paper", "stage1_relevance": "ambiguous_relevance",
        "stage1_extractability": "table_extractable", "stage1_reason_code": "test_canned",
        "rule_fired": "test_canned", "decision_basis": "test_canned",
    }
    rop.RUNS_ROOT = _tmp_path / "cloud_runs_ambiguous"

    def _stage2_reached_marker(*a, **k):
        raise RuntimeError("stage2_reached_as_expected")
    rop.run_benchmark_for_paper = _stage2_reached_marker

    _ambiguous_input_dir = _tmp_path / "ambiguous_input"
    _ambiguous_input_dir.mkdir()
    try:
        rop.run_one_paper(_make_paper_file(_ambiguous_input_dir))
        check("R10: ambiguous-retained case (table_extractable) reaches the Stage 2 call site", False,
              detail="no exception was raised at all")
    except RuntimeError as e:
        check("R10: ambiguous-retained case (table_extractable) DOES reach the Stage 2 call site",
              str(e) == "stage2_reached_as_expected", detail=str(e))
    except Exception as e:
        check("R10: ambiguous-retained case (table_extractable) DOES reach the Stage 2 call site", False,
              detail=f"unexpected {type(e).__name__}: {e}")
    _restore()

    check("R10: stage1_routing_decision distinguishes clean vs ambiguous retention (pure-function check)",
          rop.stage1_routing_decision("cpp_relevant", "table_extractable",
                                       combined_status("cpp_relevant", "table_extractable"))
          != rop.stage1_routing_decision("ambiguous_relevance", "table_extractable",
                                          combined_status("ambiguous_relevance", "table_extractable")))


# ─────────────────────────────────────────────────────────────────────────
# Accounting check (task section 5): the Stage 1 result is saved to disk
# exactly as returned by screen_paper -- unaltered by the routing fix.
# ─────────────────────────────────────────────────────────────────────────

with tempfile.TemporaryDirectory() as _tmp2:
    _tmp2_path = Path(_tmp2)
    _canned = {
        "doi": "10.9999/test-paper", "stage1_relevance": "ambiguous_relevance",
        "stage1_extractability": "figure_only", "stage1_reason_code": "test_canned_accounting",
        "rule_fired": "test_canned", "decision_basis": "test_canned",
    }
    rop.screen_paper = lambda pf: dict(_canned)
    rop.RUNS_ROOT = _tmp2_path / "cloud_runs_accounting"
    _accounting_runs_root = rop.RUNS_ROOT  # captured BEFORE _restore(), so the
    # verification path below reflects where the run actually happened, not
    # the real RUNS_ROOT this gets reset back to immediately after.

    manifest = rop.run_one_paper(_make_paper_file(_tmp2_path))
    _restore()

    stage1_saved_path = _accounting_runs_root / manifest["run_id"] / "stage1" / "10-9999_test-paper.json"
    check("accounting: Stage 1 result file was written to disk regardless of routing outcome",
          stage1_saved_path.is_file(), detail=str(stage1_saved_path))
    if stage1_saved_path.is_file():
        saved = json.loads(stage1_saved_path.read_text())
        check("accounting: saved Stage 1 result preserves the EXACT original classification "
              "(relevance/extractability untouched by the routing fix)",
              saved.get("stage1_relevance") == "ambiguous_relevance"
              and saved.get("stage1_extractability") == "figure_only", detail=str(saved))
    check("accounting: figure_only ambiguous case is still excluded (Stage 2 never reached)",
          manifest["status"] == "excluded_at_stage1"
          and manifest.get("stage1_routing_decision") == "exclude_stage1",
          detail=str((manifest["status"], manifest.get("stage1_routing_decision"))))


print()
print("=" * 70)
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
