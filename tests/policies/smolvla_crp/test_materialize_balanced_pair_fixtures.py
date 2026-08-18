import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path("scripts/research/crp_vla/materialize_balanced_pair_fixtures.py")
SPEC = importlib.util.spec_from_file_location("materialize_balanced_pair_fixtures", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_select_episode_uses_lowest_exact_task_match():
    episodes = [
        {"episode_index": 9, "length": 11},
        {"episode_index": 2, "length": 13},
        {"episode_index": 1, "length": 7},
    ]
    assert MODULE.select_episode(episodes, {9: 4, 2: 4, 1: 3}, 4) == (2, 13)


def test_select_episode_rejects_missing_exact_task():
    episodes = [{"episode_index": 0, "length": 4}]
    with pytest.raises(ValueError, match="No episode"):
        MODULE.select_episode(episodes, {0: 2}, 3)
