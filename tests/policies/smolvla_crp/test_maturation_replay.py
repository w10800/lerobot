import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).parents[3] / "scripts/research/crp_vla"
sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location(
    "run_maturation_replay_v2", SCRIPT_DIR / "run_maturation_replay_v2.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_validate_capsule_metadata_accepts_frozen_development_case():
    case = {
        "suite": "libero_goal",
        "task_id": 3,
        "init_state_id": 4,
        "env_seed": 0,
        "max_steps": 300,
    }
    metadata = {
        "phase": "development",
        "suite": "libero_goal",
        "task_id": 3,
        "init_state_id": 4,
        "env_seed": 0,
        "maximum_episode_length": 300,
    }
    MODULE.validate_capsule_metadata(metadata, case)


def test_validate_capsule_metadata_fails_closed():
    case = {
        "suite": "libero_goal",
        "task_id": 3,
        "init_state_id": 4,
        "env_seed": 0,
        "max_steps": 300,
    }
    metadata = {
        "phase": "development",
        "suite": "libero_goal",
        "task_id": 3,
        "init_state_id": 5,
        "env_seed": 0,
        "maximum_episode_length": 300,
    }
    with pytest.raises(ValueError, match="Capsule/design mismatch"):
        MODULE.validate_capsule_metadata(metadata, case)


def test_trace_root_is_checkpoint_scoped():
    output = Path("artifacts/crp_vla/maturation/replay/step_003000.json")
    assert MODULE.checkpoint_trace_root(output, 3000) == Path(
        "artifacts/crp_vla/maturation/replay/traces/step_003000"
    )
    assert MODULE.checkpoint_trace_root(output, 3000) != MODULE.checkpoint_trace_root(output, 5000)
