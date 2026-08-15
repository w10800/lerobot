from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[3] / "scripts" / "research" / "crp_vla"
sys.path.insert(0, str(SCRIPT_DIR))

from finalize_task12 import compare_replays  # noqa: E402


def replay(successes: dict[str, list[bool]]) -> dict:
    results = []
    nfe = {"snap10": 10, "snap2": 2, "snap1": 1}
    for arm, values in successes.items():
        for index in range(40):
            success = values[index % len(values)]
            results.append(
                {
                    "suite": "suite",
                    "task_id": index,
                    "init_state_id": 4,
                    "env_seed": 0,
                    "arm": arm,
                    "nfe": nfe[arm],
                    "success": success,
                    "steps_run": 10 + index,
                    "initial_sim_state_sha256": f"state-{index}",
                    "canonical_input_sha256": f"input-{index}",
                    "noise_schedule_sha256": f"noise-{index}",
                    "evaluator_sha256": "evaluator",
                }
            )
    return {"status": "MATURATION_REPLAY_COMPLETED", "results": results}


def test_compare_replays_reports_probe_minus_original() -> None:
    original = replay({"snap10": [True, False], "snap2": [True, False], "snap1": [True, False]})
    probe = replay({"snap10": [True, False], "snap2": [True, True], "snap1": [False, False]})
    result = compare_replays(probe, original, repeats=100)
    assert result["snap10"]["probe_successes"] == 20
    assert result["snap2"]["probe_successes"] == 40
    assert result["snap1"]["probe_successes"] == 0
    assert result["snap1"]["probe_minus_original"]["paired_success_difference"] == -0.5


def test_compare_replays_rejects_invariant_drift() -> None:
    original = replay({arm: [True, False] for arm in ("snap10", "snap2", "snap1")})
    probe = replay({arm: [True, False] for arm in ("snap10", "snap2", "snap1")})
    probe["results"][0]["noise_schedule_sha256"] = "changed"
    with pytest.raises(ValueError, match="invariant mismatch"):
        compare_replays(probe, original, repeats=100)
