import pytest
import torch

from lerobot.policies.smolvla_crp.paired_inputs import (
    assert_matched_batches,
    evaluate_condition_pair,
    tensor_sha256,
)
from lerobot.policies.smolvla_crp.response_metrics import compute_response_metrics


def test_identical_teacher_student_response_is_perfect():
    teacher_factual = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
    teacher_counterfactual = torch.zeros_like(teacher_factual)
    metrics = compute_response_metrics(
        teacher_factual,
        teacher_counterfactual,
        teacher_factual.clone(),
        teacher_counterfactual.clone(),
    )
    assert torch.allclose(metrics["global"]["response_norm_ratio"], torch.tensor(1.0))
    assert torch.allclose(metrics["global"]["response_cosine"], torch.tensor(1.0))
    assert torch.allclose(metrics["global"]["normalized_response_error"], torch.tensor(0.0))
    assert torch.allclose(metrics["horizon"]["response_norm_ratio"], torch.ones(2))


def test_zero_student_response_reports_loss_of_response():
    teacher_factual = torch.ones(2, 3, 1)
    zeros = torch.zeros_like(teacher_factual)
    metrics = compute_response_metrics(teacher_factual, zeros, zeros, zeros)
    assert torch.allclose(metrics["global"]["response_norm_ratio"], torch.tensor(0.0))
    assert torch.allclose(metrics["global"]["normalized_response_error"], torch.tensor(1.0))


def test_matched_batch_rejects_observation_or_state_change():
    factual = {
        "observation.state": torch.tensor([[1.0]]),
        "observation.language.tokens": torch.tensor([[1, 2]]),
    }
    counterfactual = {
        "observation.state": torch.tensor([[2.0]]),
        "observation.language.tokens": torch.tensor([[3, 4]]),
    }
    with pytest.raises(ValueError, match="invariant"):
        assert_matched_batches(factual, counterfactual)


def test_matched_batch_allows_only_language_change_and_noise_hash_is_stable():
    state = torch.tensor([[1.0]])
    factual = {
        "observation.state": state,
        "observation.language.tokens": torch.tensor([[1, 2]]),
    }
    counterfactual = {
        "observation.state": state.clone(),
        "observation.language.tokens": torch.tensor([[3, 4]]),
    }
    assert_matched_batches(factual, counterfactual)
    assert tensor_sha256(state) == tensor_sha256(state.clone())


class DummyPolicy:
    class Config:
        num_steps = 10

    def __init__(self):
        self.config = self.Config()
        self.noises = []

    def reset(self):
        pass

    def predict_action_chunk(self, batch, noise, **kwargs):
        self.noises.append(noise.clone())
        return noise + batch["observation.language.tokens"].float().sum()


def test_pair_evaluation_reuses_noise_and_restores_num_steps():
    policy = DummyPolicy()
    state = torch.tensor([[1.0]])
    factual = {
        "observation.state": state,
        "observation.language.tokens": torch.tensor([[1, 2]]),
    }
    counterfactual = {
        "observation.state": state.clone(),
        "observation.language.tokens": torch.tensor([[3, 4]]),
    }
    noise = torch.randn(1, 2, 1)
    evaluate_condition_pair(policy, factual, counterfactual, noise, num_steps=1)
    assert torch.equal(policy.noises[0], policy.noises[1])
    assert torch.equal(policy.noises[0], noise)
    assert policy.config.num_steps == 10
