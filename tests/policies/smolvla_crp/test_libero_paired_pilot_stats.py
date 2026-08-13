import importlib.util
from pathlib import Path

SCRIPT = Path("scripts/research/crp_vla/run_libero_paired_four_arm_pilot.py")
SPEC = importlib.util.spec_from_file_location("run_libero_paired_four_arm_pilot", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_paired_summary_counts_discordant_outcomes():
    results = []
    outcomes = [(True, False, True, False), (False, True, False, False)]
    for case_id, values in enumerate(outcomes):
        for arm, success in zip(MODULE.ARM_SPECS, values, strict=True):
            results.append(
                {
                    "suite": "suite",
                    "task_id": case_id,
                    "init_state_id": 0,
                    "env_seed": 0,
                    "arm": arm,
                    "success": success,
                }
            )
    summary = MODULE.paired_summary(results, bootstrap_repeats=100, bootstrap_seed=0)
    assert summary["arm_success"]["base10"]["rate"] == 0.5
    assert summary["paired_vs_base10"]["base1"]["base_success_candidate_failure"] == 1
    assert summary["paired_vs_base10"]["base1"]["base_failure_candidate_success"] == 1
    assert summary["paired_vs_base10"]["base1"]["discordant_pair_rate"] == 1.0
    assert summary["paired_vs_base10"]["base1"]["paired_bootstrap_ci95"] == [-1.0, 1.0]
