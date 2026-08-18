import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[3] / "scripts/research/crp_vla/materialize_maturation_fixture.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("materialize_maturation_fixture", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_select_latest_episode_per_task():
    episodes = [
        {"episode_index": 2, "length": 20},
        {"episode_index": 7, "length": 70},
        {"episode_index": 9, "length": 90},
        {"episode_index": 11, "length": 110},
    ]
    mapping = {2: 0, 7: 1, 9: 0, 11: 1}
    assert MODULE.select_latest_episodes(episodes, mapping, expected_tasks=2) == {
        0: (9, 90),
        1: (11, 110),
    }


def test_select_latest_episode_requires_complete_task_coverage():
    with pytest.raises(ValueError, match="missing"):
        MODULE.select_latest_episodes([{"episode_index": 0, "length": 1}], {0: 0}, expected_tasks=2)


class EpochSampler:
    def __init__(self):
        self.epoch = 0

    def __iter__(self):
        epoch = self.epoch
        self.epoch += 1
        return iter((epoch * 3, epoch * 3 + 1, epoch * 3 + 2))


def test_data_order_hash_covers_exact_step_prefixes():
    first = MODULE.hash_sampler_prefixes(EpochSampler(), 2, (1, 3, 4))
    second = MODULE.hash_sampler_prefixes(EpochSampler(), 2, (1, 3, 4))
    assert first == second
    assert set(first) == {1, 3, 4}
    assert len(set(first.values())) == 3


def test_data_order_hash_rejects_nonmonotonic_schedule():
    with pytest.raises(ValueError, match="strictly increasing"):
        MODULE.hash_sampler_prefixes(EpochSampler(), 2, (3, 1))
