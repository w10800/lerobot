from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts" / "research" / "crp_vla"
sys.path.insert(0, str(SCRIPTS))

from task12_common import (  # noqa: E402
    bootstrap_median_difference,
    jaccard_distance,
    outcome_stratum,
    progress_bin,
    validate_task11_pass,
)


def test_outcome_strata_are_directional() -> None:
    assert outcome_stratum(True, False) == "base_only_success"
    assert outcome_stratum(False, True) == "snap_only_success"


def test_jaccard_and_progress_bins() -> None:
    assert jaccard_distance({1, 2}, {2, 3}) == pytest.approx(2 / 3)
    assert [progress_bin(index, 10) for index in range(10)] == list(range(10))


def test_bootstrap_is_reproducible() -> None:
    first = bootstrap_median_difference([2, 3, 4], [0, 1, 2], repeats=100, seed=7)
    second = bootstrap_median_difference([2, 3, 4], [0, 1, 2], repeats=100, seed=7)
    assert first == second
    assert first["point"] == pytest.approx(2.0)


def test_pass_validation_is_fail_closed() -> None:
    validate_task11_pass(
        {
            "status": "FORMAL_CONFIRMATION1200_PASS",
            "case_count": 1200,
            "noninferiority_pass": True,
            "frozen_protocol_modified": False,
        }
    )
    with pytest.raises(RuntimeError):
        validate_task11_pass(
            {
                "status": "FORMAL_CONFIRMATION1200_FAIL",
                "case_count": 1200,
                "noninferiority_pass": False,
                "frozen_protocol_modified": False,
            }
        )
