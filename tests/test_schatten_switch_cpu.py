"""CPU-only unit tests for Schatten-switch early-search guards, candidate loading, and launch scope."""

import dataclasses
from pathlib import Path
import json

import numpy as np
import pytest

from research.polar_opt import eval_heldout_nesterov as eval_mod
from research.polar_opt import schatten_switch_search as search
from research.polar_opt.eval_heldout_nesterov import _candidate_methods, _resolve_eval_output
from research.polar_opt.schatten_switch_blackwell import (
    SCHATTEN_SWITCH_CIRCUITS,
    SchattenSwitchCircuit,
    theoretical_flops,
)


def test_early_srank_spectra_are_synthetic_and_near_rank_one():
    spectra = search.early_srank_spectra()
    again = search.early_srank_spectra()
    assert len(spectra) == len(search.EARLY_SIZES) * len(search.EARLY_SRANKS) * (
        len(search.EARLY_POWERS) + len(search.EARLY_SECONDARIES) * len(search.EARLY_FLOORS)
    )
    for row, other in zip(spectra, again):
        s = row["s"]
        assert np.array_equal(s, other["s"]) and len(s) == row["n"]
        assert 1.03 - 1e-9 <= np.sum(s ** 2) / s[0] ** 2 <= 1.60 + 1e-9
        assert s[0] == s.max() == 1.0 and s.min() > 0
    assert "harvested_nesterov_real" not in Path(search.__file__).read_text()


def test_resolve_search_output_guards_seed_artifact(tmp_path):
    default_seed = Path("results/schatten_switch_search_cycle2.json")
    assert search._resolve_search_output(None, False, default_seed) == default_seed
    early_default = search._resolve_search_output(None, True, default_seed)
    assert early_default == Path("results/schatten_switch_early_polish.json")
    assert early_default != default_seed

    with pytest.raises(ValueError, match="must differ from --seed-artifact"):
        search._resolve_search_output(default_seed, True, default_seed)
    with pytest.raises(ValueError, match="must differ from --seed-artifact"):
        search._resolve_search_output(None, True, early_default)

    custom_out = tmp_path / "early_out.json"
    assert search._resolve_search_output(custom_out, True, tmp_path / "seed.json") == custom_out


def test_resolve_eval_output_guards_baseline_and_candidates(tmp_path):
    default_out = eval_mod.ROOT / "results" / "cycle2_heldout_nesterov_eval.json"
    rel_default = Path("results/cycle2_heldout_nesterov_eval.json")
    assert _resolve_eval_output(None, None) == default_out

    cand = tmp_path / "search.json"
    with pytest.raises(ValueError, match="--output is required"):
        _resolve_eval_output(None, cand)
    with pytest.raises(ValueError, match="must not overwrite baseline"):
        _resolve_eval_output(default_out, cand)
    with pytest.raises(ValueError, match="must not overwrite baseline"):
        _resolve_eval_output(rel_default, cand)
    with pytest.raises(ValueError, match="must differ from --candidates"):
        _resolve_eval_output(cand, cand)

    custom_out = tmp_path / "eval.json"
    assert _resolve_eval_output(custom_out, cand) == custom_out


def test_candidate_methods_filters_failed_fp16_and_checks_bf16_record(tmp_path):
    c_pass_fp16 = dataclasses.replace(SCHATTEN_SWITCH_CIRCUITS["SS3_54"], name="PASS_FP16")
    c_pass_both = dataclasses.replace(SCHATTEN_SWITCH_CIRCUITS["SS3_35"], name="PASS_BOTH")
    c_bf16_blocked = dataclasses.replace(SCHATTEN_SWITCH_CIRCUITS["SS3_35"], name="BF16_BLOCKED")
    c_fail_fp16 = dataclasses.replace(SCHATTEN_SWITCH_CIRCUITS["SS3_54"], name="FAIL_FP16")
    c_missing_flag = dataclasses.replace(SCHATTEN_SWITCH_CIRCUITS["SS3_54"], name="MISSING_FLAG")

    payload = {
        "validated": {
            "PASS_FP16": {"circuit": c_pass_fp16.to_dict(), "fp16_passed": True, "bf16_verified": False},
            "PASS_BOTH": {"circuit": c_pass_both.to_dict(), "fp16_passed": True, "bf16_verified": True},
            "BF16_BLOCKED": {"circuit": c_bf16_blocked.to_dict(), "fp16_passed": True, "bf16_verified": False},
            "FAIL_FP16": {"circuit": c_fail_fp16.to_dict(), "fp16_passed": False, "bf16_verified": False},
            "MISSING_FLAG": {"circuit": c_missing_flag.to_dict()},
        }
    }
    path = tmp_path / "search.json"
    path.write_text(json.dumps(payload))

    default_methods = _candidate_methods(path)
    assert sorted(default_methods) == [
        "cand_BF16_BLOCKED_fp16",
        "cand_PASS_BOTH_bf16",
        "cand_PASS_BOTH_fp16",
        "cand_PASS_FP16_fp16",
    ]

    all_methods = _candidate_methods(path, include_failed=True)
    assert sorted(all_methods) == [
        "cand_BF16_BLOCKED_fp16",
        "cand_FAIL_FP16_fp16",
        "cand_MISSING_FLAG_fp16",
        "cand_PASS_BOTH_bf16",
        "cand_PASS_BOTH_fp16",
        "cand_PASS_FP16_fp16",
    ]


@pytest.mark.parametrize("name", sorted(SCHATTEN_SWITCH_CIRCUITS))
def test_kernel_launches_matches_registered_zero_residual_circuits(name):
    circuit = SCHATTEN_SWITCH_CIRCUITS[name]
    record = search.kernel_launches(circuit, 512)
    assert record["matrix_products"] == theoretical_flops(name, (512, 2048))["matrix_kernel_launches"]
    assert record["total"] == record["matrix_products"] + 4
    assert record["host_reads"] == 0


def test_kernel_launches_rejects_nonzero_seg1_residual():
    res_circuit = SchattenSwitchCircuit(
        name="TEST_RES",
        table=((0.05, ((1, 0, 0, 1.0, -1.0, 0.0), (0, 1, 0, 0.0, 0.5, 0.45))),),
        buffers=2,
        u_safe=1.05,
        lo=0.5,
        error=0.5,
    )
    with pytest.raises(ValueError, match="zero-residual segment-1"):
        search.kernel_launches(res_circuit, 512)
