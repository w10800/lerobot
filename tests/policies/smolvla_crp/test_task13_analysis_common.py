from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[3] / "scripts" / "research" / "crp_vla"
sys.path.insert(0, str(SCRIPT_DIR))

from task13_analysis_common import (  # noqa: E402
    auroc,
    average_precision,
    contrast,
    outcome_label,
)


def test_task13_outcome_labels_are_frozen() -> None:
    assert outcome_label(True, True) == "preserved"
    assert outcome_label(True, False) == "harmful"
    assert outcome_label(False, True) == "student_only_success"
    assert outcome_label(False, False) == "both_fail"


def test_task13_contrast_preserves_case_and_task_clusters() -> None:
    rows = []
    for task in range(4):
        rows.extend(
                [
                {"task_id": f"task:{task}", "outcome_label": "harmful", "value": 3.0},
                {"task_id": f"task:{task}", "outcome_label": "preserved", "value": 1.0},
            ]
        )
    result = contrast(rows, "value", repeats=100, seed=17)
    assert result["mean_difference"] == 2.0
    assert result["case_bootstrap_ci95"][0] > 0
    assert result["task_cluster_bootstrap_ci95"][0] > 0
    assert result["leave_one_task_out"]["positive_count"] == 4
    assert result["cluster_units"] == {
        "case": True,
        "task": True,
        "state_as_independent_sample": False,
    }


def test_task13_prediction_metrics_handle_ranked_scores() -> None:
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    assert auroc(labels, scores) == 1.0
    assert average_precision(labels, scores) == 1.0
