import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[3] / "scripts/research/crp_vla/summarize_maturation.py"
SPEC = importlib.util.spec_from_file_location("summarize_maturation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_training_metric_parser_reads_registered_step_and_sub_losses():
    lines = [
        f"INFO ot_train.py:769 step:{step} loss:0.999 grdn:0.999 loss/fm:0.999"
        for step in range(50, 1000, 50)
    ]
    # The exact 1k record has the same rounded label as later records; record order is the
    # immutable identifier for this uninterrupted log_freq=50 run.
    lines.append(
        "INFO ot_train.py:769 step:1K smpl:4K loss:0.123 grdn:0.456 loss/fm:0.111 "
        "loss/shortcut:0.120 loss/accounted_total:0.123"
    )
    lines.append("INFO ot_train.py:769 step:1K loss:0.777 grdn:0.777 loss/fm:0.777")
    text = "\n".join(lines)
    metrics, failures = MODULE.parse_training_metrics(text)
    assert metrics[1000]["loss/fm"] == 0.111
    assert metrics[1000]["grdn"] == 0.456
    assert failures == []


def test_frozen_tie_breaker_is_earlier_step():
    rows = [
        {"step": 1000, "eligible": True, "successes": {"snap1": 20}},
        {"step": 3000, "eligible": True, "successes": {"snap1": 21}},
        {"step": 5000, "eligible": True, "successes": {"snap1": 21}},
    ]
    selected = min(
        [row for row in rows if row["eligible"]],
        key=lambda row: (-row["successes"]["snap1"], row["step"]),
    )
    assert selected["step"] == 3000


def test_non_finite_nested_value_fails_closed():
    assert MODULE.all_finite({"ok": [1.0, 2], "nested": {"x": 3.0}})
    assert not MODULE.all_finite({"loss": float("nan")})
