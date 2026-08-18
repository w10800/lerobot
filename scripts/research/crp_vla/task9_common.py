#!/usr/bin/env python
"""Pure, fail-closed helpers for CRP-VLA Task 9.

The module intentionally contains no simulator or policy-loading code so its
manifest, overlap, statistics, and aggregation gates can be unit tested on any
host.  Task 9 is a baseline-validation preflight; none of these helpers can
start training or a confirmation rollout.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

SELECTED_STEP = 20_000
SELECTED_MODEL_SHA256 = "3523ff36091fdba82a97b798621b4ecee141554fcfe418816a6fbb1cf95f2b53"
REGISTERED_STEPS = (1_000, 3_000, 5_000, 10_000, 20_000, 30_000)
NFES = (10, 2, 1)
PRIMARY_ARMS = ("base10", "snap1_20k")
DIAGNOSTIC_ARMS = ("base2", "base1", "snap10_20k", "snap2_20k")
NONINFERIORITY_MARGIN = -0.03


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    hasher = hashlib.sha256()
    hasher.update(str(array.dtype).encode())
    hasher.update(json.dumps(array.shape).encode())
    hasher.update(array.tobytes())
    return hasher.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except Exception as error:
        raise ValueError(f"Cannot parse JSON input {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return value


def require_model_hash(path: Path, expected: str = SELECTED_MODEL_SHA256) -> str:
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"Model SHA-256 mismatch: expected {expected}, observed {actual}")
    return actual


def effective_processor_contract(checkpoint: Path) -> dict[str, Any]:
    """Hash the effective state/action processor contract.

    Maturation checkpoints carry extra unused image-stat tensors and a local
    tokenizer path, so byte-level processor manifests differ from the Hub base
    even though replay-v2 proved exact canonical tensors for all 40 task
    prompts.  Task 9 therefore compares the state/action normalization tensors,
    the postprocessor configuration, and the preprocessor step configuration
    after removing tokenizer storage-location metadata.  Live preflight still
    verifies a canonical batch and both model calls.
    """
    from safetensors.torch import load_file

    pre_json = load_json(checkpoint / "policy_preprocessor.json")
    post_json = load_json(checkpoint / "policy_postprocessor.json")
    tensors = load_file(checkpoint / "policy_preprocessor_step_5_normalizer_processor.safetensors")
    required = {
        key: hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        for key, tensor in sorted(tensors.items())
        if key.startswith("action.") or key.startswith("observation.state.")
    }
    if not required or not any(key.startswith("action.") for key in required):
        raise ValueError(f"Processor contract lacks action normalization tensors: {checkpoint}")

    def remove_storage_metadata(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: remove_storage_metadata(child)
                for key, child in value.items()
                if key not in {"artifacts", "tokenizer_name"}
            }
        if isinstance(value, list):
            return [remove_storage_metadata(child) for child in value]
        return value

    payload = {
        "state_action_tensor_sha256": required,
        "preprocessor_config_without_tokenizer_storage_metadata": remove_storage_metadata(pre_json),
        "postprocessor_config": post_json,
    }
    return {"sha256": canonical_json_sha256(payload), "payload": payload}


def exact_mcnemar(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left = np.asarray(left, dtype=np.int8)
    right = np.asarray(right, dtype=np.int8)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("McNemar inputs must be same-length vectors")
    left_win = int(np.sum((left == 1) & (right == 0)))
    right_win = int(np.sum((left == 0) & (right == 1)))
    discordant = left_win + right_win
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, index) for index in range(min(left_win, right_win) + 1))
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    return {
        "n_10": left_win,
        "n_01": right_win,
        "discordant": discordant,
        "two_sided_exact_p": float(p_value),
    }


def bootstrap_interval(values: np.ndarray, repeats: int, seed: int) -> list[float]:
    """The exact percentile paired-case procedure used by the formal gate."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Bootstrap requires a non-empty vector")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(repeats, len(values)))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def task_cluster_bootstrap_interval(
    values: np.ndarray, task_ids: Iterable[str], repeats: int, seed: int
) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    task_ids = np.asarray(list(task_ids), dtype=object)
    if len(values) != len(task_ids) or len(values) == 0:
        raise ValueError("Cluster bootstrap inputs must be non-empty and aligned")
    clusters = sorted(set(task_ids.tolist()))
    grouped = {cluster: values[task_ids == cluster] for cluster in clusters}
    generator = np.random.default_rng(seed)
    samples = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        selected = generator.integers(0, len(clusters), size=len(clusters))
        draw = np.concatenate([grouped[clusters[item]] for item in selected])
        samples[index] = draw.mean()
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def successful_jaccard(left: np.ndarray, right: np.ndarray) -> float:
    left_set = set(np.flatnonzero(np.asarray(left, dtype=bool)).tolist())
    right_set = set(np.flatnonzero(np.asarray(right, dtype=bool)).tolist())
    union = left_set | right_set
    return 1.0 if not union else len(left_set & right_set) / len(union)


def paired_flip_summary(
    left: np.ndarray,
    right: np.ndarray,
    task_ids: Iterable[str],
    *,
    repeats: int = 10_000,
    seed: int = 20260815,
) -> dict[str, Any]:
    left = np.asarray(left, dtype=np.int8)
    right = np.asarray(right, dtype=np.int8)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("Paired vectors must be aligned")
    differences = left.astype(np.float64) - right.astype(np.float64)
    mcnemar = exact_mcnemar(left, right)
    return {
        "case_count": int(len(left)),
        "n_11": int(np.sum((left == 1) & (right == 1))),
        "n_00": int(np.sum((left == 0) & (right == 0))),
        "n_10": mcnemar["n_10"],
        "n_01": mcnemar["n_01"],
        "paired_success_difference": float(differences.mean()),
        "exact_mcnemar_p": mcnemar["two_sided_exact_p"],
        "case_bootstrap_ci95": bootstrap_interval(differences, repeats, seed),
        "task_cluster_bootstrap_ci95": task_cluster_bootstrap_interval(
            differences, task_ids, repeats, seed + 10_000
        ),
        "successful_case_jaccard": float(successful_jaccard(left, right)),
    }


def select_checkpoint(success_vectors: dict[int, np.ndarray], eligible_steps: Iterable[int]) -> int:
    eligible = sorted(int(step) for step in eligible_steps)
    if not eligible:
        raise ValueError("No eligible checkpoint")
    missing = [step for step in eligible if step not in success_vectors]
    if missing:
        raise ValueError(f"Missing success vectors: {missing}")
    return min(eligible, key=lambda step: (-int(np.asarray(success_vectors[step]).sum()), step))


def selection_bootstrap(
    success_vectors: dict[int, np.ndarray],
    eligible_steps: Iterable[int],
    task_ids: Iterable[str],
    *,
    repeats: int = 10_000,
    seed: int = 20260815,
) -> dict[str, Any]:
    eligible = sorted(int(step) for step in eligible_steps)
    arrays = {step: np.asarray(success_vectors[step], dtype=np.int8) for step in eligible}
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("Checkpoint vectors are not aligned")
    case_count = lengths.pop()
    task_ids = np.asarray(list(task_ids), dtype=object)
    if len(task_ids) != case_count:
        raise ValueError("Task IDs are not aligned to checkpoint vectors")
    observed = select_checkpoint(arrays, eligible)

    def choose(indices: np.ndarray) -> int:
        return min(eligible, key=lambda step: (-int(arrays[step][indices].sum()), step))

    rng = np.random.default_rng(seed)
    case_counts = dict.fromkeys(eligible, 0)
    for _ in range(repeats):
        indices = rng.integers(0, case_count, size=case_count)
        case_counts[choose(indices)] += 1

    clusters = sorted(set(task_ids.tolist()))
    cluster_indices = {cluster: np.flatnonzero(task_ids == cluster) for cluster in clusters}
    cluster_counts = dict.fromkeys(eligible, 0)
    for _ in range(repeats):
        selected_clusters = rng.integers(0, len(clusters), size=len(clusters))
        indices = np.concatenate([cluster_indices[clusters[index]] for index in selected_clusters])
        cluster_counts[choose(indices)] += 1

    loo_case = []
    for omitted in range(case_count):
        indices = np.delete(np.arange(case_count), omitted)
        loo_case.append(choose(indices))
    loo_task = []
    for cluster in clusters:
        indices = np.flatnonzero(task_ids != cluster)
        loo_task.append(choose(indices))

    return {
        "observed_selected_step": observed,
        "eligible_steps": eligible,
        "case_bootstrap_replicates": repeats,
        "case_selection_counts": {str(step): case_counts[step] for step in eligible},
        "case_selection_frequencies": {str(step): case_counts[step] / repeats for step in eligible},
        "task_cluster_bootstrap_replicates": repeats,
        "task_cluster_selection_counts": {str(step): cluster_counts[step] for step in eligible},
        "task_cluster_selection_frequencies": {
            str(step): cluster_counts[step] / repeats for step in eligible
        },
        "leave_one_case_out_selected_steps": loo_case,
        "leave_one_case_out_20k_retention": loo_case.count(SELECTED_STEP) / len(loo_case),
        "leave_one_task_out_selected_steps": loo_task,
        "leave_one_task_out_task_ids": clusters,
        "leave_one_task_out_20k_retention": loo_task.count(SELECTED_STEP) / len(loo_task),
        "single_task_determines_20k": any(step != SELECTED_STEP for step in loo_task),
        "frozen_selected_step_unchanged": SELECTED_STEP,
    }


def spearman(x: Iterable[float], y: Iterable[float]) -> dict[str, float | int | None]:
    from scipy.stats import spearmanr

    x_values = np.asarray(list(x), dtype=np.float64)
    y_values = np.asarray(list(y), dtype=np.float64)
    result = spearmanr(x_values, y_values)
    statistic = float(result.statistic) if np.isfinite(result.statistic) else None
    p_value = float(result.pvalue) if np.isfinite(result.pvalue) else None
    return {"rho": statistic, "two_sided_p": p_value, "n": int(len(x_values))}


def standardized_mean_difference(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 2 or len(right) < 2:
        return None
    pooled = math.sqrt(
        ((len(left) - 1) * left.var(ddof=1) + (len(right) - 1) * right.var(ddof=1))
        / (len(left) + len(right) - 2)
    )
    return None if pooled == 0 else float((left.mean() - right.mean()) / pooled)


def _fit_logistic(x: np.ndarray, y: np.ndarray, regularization: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    design = np.column_stack([np.ones(len(x)), x])
    beta = np.zeros(design.shape[1], dtype=np.float64)
    penalty = np.eye(design.shape[1], dtype=np.float64) * regularization
    penalty[0, 0] = 0.0
    for _ in range(100):
        linear = np.clip(design @ beta, -30, 30)
        probabilities = 1.0 / (1.0 + np.exp(-linear))
        weights = np.clip(probabilities * (1 - probabilities), 1e-8, None)
        gradient = design.T @ (probabilities - y) + penalty @ beta
        hessian = design.T @ (design * weights[:, None]) + penalty
        update = np.linalg.solve(hessian, gradient)
        beta -= update
        if float(np.max(np.abs(update))) < 1e-10:
            break
    return beta


def leave_one_task_out_predictions(
    features: np.ndarray, outcomes: np.ndarray, task_ids: Iterable[str]
) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    if features.ndim == 1:
        features = features[:, None]
    outcomes = np.asarray(outcomes, dtype=np.int8)
    task_ids = np.asarray(list(task_ids), dtype=object)
    if len(features) != len(outcomes) or len(outcomes) != len(task_ids):
        raise ValueError("Prediction inputs are not aligned")
    predictions = np.empty(len(outcomes), dtype=np.float64)
    for task in sorted(set(task_ids.tolist())):
        train = task_ids != task
        test = ~train
        if len(set(outcomes[train].tolist())) < 2:
            predictions[test] = float(outcomes[train].mean())
            continue
        means = features[train].mean(axis=0)
        scales = features[train].std(axis=0)
        scales[scales == 0] = 1.0
        train_x = (features[train] - means) / scales
        test_x = (features[test] - means) / scales
        beta = _fit_logistic(train_x, outcomes[train])
        design = np.column_stack([np.ones(len(test_x)), test_x])
        predictions[test] = 1.0 / (1.0 + np.exp(-np.clip(design @ beta, -30, 30)))
    return np.clip(predictions, 1e-9, 1 - 1e-9)


def prediction_metrics(outcomes: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    from scipy.stats import rankdata

    outcomes = np.asarray(outcomes, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    positives = int(outcomes.sum())
    negatives = int(len(outcomes) - positives)
    if positives and negatives:
        ranks = rankdata(probabilities, method="average")
        auroc = float(
            (ranks[outcomes == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives)
        )
    else:
        auroc = None
    order = np.argsort(-probabilities, kind="stable")
    sorted_y = outcomes[order]
    if positives:
        precision = np.cumsum(sorted_y) / np.arange(1, len(sorted_y) + 1)
        auprc = float((precision * sorted_y).sum() / positives)
    else:
        auprc = None
    brier = float(np.mean((probabilities - outcomes) ** 2))
    bins = np.linspace(0, 1, 6)
    calibration = 0.0
    for lower, upper in zip(bins[:-1], bins[1:], strict=True):
        mask = (probabilities >= lower) & (probabilities < upper if upper < 1 else probabilities <= upper)
        if mask.any():
            calibration += mask.mean() * abs(probabilities[mask].mean() - outcomes[mask].mean())
    sensitivity = None
    specificity = 0.8
    if positives and negatives:
        negative_scores = probabilities[outcomes == 0]
        threshold = float(np.quantile(negative_scores, specificity, method="higher"))
        achieved_specificity = float(np.mean(negative_scores < threshold))
        sensitivity = float(np.mean(probabilities[outcomes == 1] >= threshold))
    else:
        threshold = None
        achieved_specificity = None
    return {
        "n": int(len(outcomes)),
        "positive_failures": positives,
        "prevalence": float(outcomes.mean()),
        "auroc": auroc,
        "auprc": auprc,
        "brier_score": brier,
        "expected_calibration_error_5bin": float(calibration),
        "fixed_specificity_target": specificity,
        "threshold": threshold,
        "achieved_specificity": achieved_specificity,
        "sensitivity_at_fixed_specificity": sensitivity,
    }


def validate_confirmation_manifest(record: dict[str, Any]) -> None:
    if record.get("schema_version") != 1:
        raise ValueError("Confirmation manifest schema mismatch")
    cases = record.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Confirmation manifest contains no cases")
    required = {
        "case_id",
        "task_id",
        "task_name",
        "initial_state_source",
        "initial_state_hash",
        "simulator_state_hash",
        "qpos_qvel_hash",
        "observation_hash",
        "processor_hash",
        "evaluator_hash",
        "runtime_metadata",
        "noise_schedule_hash",
        "formal100_overlap_check",
        "dev40_overlap_check",
    }
    for case in cases:
        missing = sorted(required - set(case))
        if missing:
            raise ValueError(f"Confirmation case missing fields: {missing}")
        if case["formal100_overlap_check"] or case["dev40_overlap_check"]:
            raise ValueError(f"Confirmation overlap detected: {case['case_id']}")
    for field in ("case_id", "initial_state_hash", "simulator_state_hash", "qpos_qvel_hash"):
        values = [case[field] for case in cases]
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate confirmation {field}")


def deterministic_diagnostic_subset(
    cases: list[dict[str, Any]], *, per_task: int = 5, seed: int = 20260815
) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_task[str(case["task_id"])].append(case)
    selected: list[str] = []
    for task in sorted(by_task):
        candidates = sorted(by_task[task], key=lambda case: case["case_id"])
        if len(candidates) < per_task:
            raise ValueError(f"Task {task} has fewer than {per_task} confirmation cases")
        ranked = sorted(
            candidates,
            key=lambda case: hashlib.sha256(f"{seed}:{case['case_id']}".encode()).hexdigest(),
        )
        selected.extend(case["case_id"] for case in ranked[:per_task])
    return {
        "schema_version": 1,
        "selection_rule": "per-task SHA-256 rank of frozen seed and case_id; outcome-independent",
        "seed": seed,
        "per_task": per_task,
        "case_count": len(selected),
        "case_ids": sorted(selected),
    }


def assert_no_state_overlap(
    cases: list[dict[str, Any]], formal_hashes: set[str], dev_hashes: set[str]
) -> None:
    for case in cases:
        hashes = {
            case["initial_state_hash"],
            case["simulator_state_hash"],
            case["qpos_qvel_hash"],
        }
        if hashes & formal_hashes:
            raise ValueError(f"formal100 state overlap: {case['case_id']}")
        if hashes & dev_hashes:
            raise ValueError(f"dev40 state overlap: {case['case_id']}")


def detect_duplicate_trace_paths_and_hashes(results: list[dict[str, Any]]) -> None:
    paths = [result["trace_manifest"]["path"] for result in results]
    hashes = [result["trace_manifest"]["sha256"] for result in results]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate trace manifest path")
    if len(hashes) != len(set(hashes)):
        raise ValueError("Duplicate trace manifest hash")


def validate_arm_input_equality(records: list[dict[str, Any]], required_arms: set[str]) -> None:
    if {record["arm"] for record in records} != required_arms:
        raise ValueError("Arm set mismatch")
    keys = (
        "initial_sim_state_sha256",
        "canonical_input_sha256",
        "noise_schedule_sha256",
        "evaluator_sha256",
    )
    reference = records[0]
    for record in records[1:]:
        mismatches = [key for key in keys if record.get(key) != reference.get(key)]
        if mismatches:
            raise ValueError(f"Arm-input mismatch: {mismatches}")


def guarded_aggregate(
    run_records: list[dict[str, Any]], primary_case_ids: set[str], primary_arms: set[str]
) -> dict[str, Any]:
    primary = [
        record
        for record in run_records
        if record.get("case_id") in primary_case_ids and record.get("arm") in primary_arms
    ]
    expected = len(primary_case_ids) * len(primary_arms)
    if len(primary) != expected:
        raise RuntimeError(f"Primary aggregation refused: {len(primary)}/{expected} records")
    if any(record.get("status") != "COMPLETED" for record in primary):
        raise RuntimeError("Primary aggregation refused: incomplete/invalid status")
    keys = [(record["case_id"], record["arm"]) for record in primary]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Primary aggregation refused: duplicate case/arm")
    return {
        "case_count": len(primary_case_ids),
        "arms": {
            arm: {
                "successes": sum(bool(record["success"]) for record in primary if record["arm"] == arm),
                "rollouts": len(primary_case_ids),
            }
            for arm in sorted(primary_arms)
        },
    }


def simulate_paired_design(
    *,
    n: int,
    true_delta: float,
    discordance_rate: float,
    trials: int,
    bootstrap_repeats: int,
    seed: int,
) -> dict[str, Any]:
    """Prospective paired-Bernoulli simulation with the frozen bootstrap CI.

    Discordant probabilities are p10=(d+delta)/2 and p01=(d-delta)/2 for
    candidate-minus-base outcomes.  Remaining mass is concordant and does not
    affect the paired difference or its interval.
    """
    p10 = (discordance_rate + true_delta) / 2
    p01 = (discordance_rate - true_delta) / 2
    if min(p10, p01) < 0 or p10 + p01 > 1:
        raise ValueError("Infeasible delta/discordance combination")
    generator = np.random.default_rng(seed)
    widths = np.empty(trials, dtype=np.float64)
    passed = np.empty(trials, dtype=bool)
    estimates = np.empty(trials, dtype=np.float64)
    for trial in range(trials):
        draws = generator.choice(np.asarray([1.0, -1.0, 0.0]), size=n, p=[p10, p01, 1 - p10 - p01])
        ci = bootstrap_interval(draws, bootstrap_repeats, seed + trial + 1)
        estimates[trial] = draws.mean()
        widths[trial] = ci[1] - ci[0]
        passed[trial] = ci[0] >= NONINFERIORITY_MARGIN
    return {
        "n": n,
        "true_delta": true_delta,
        "discordance_rate": discordance_rate,
        "trials": trials,
        "bootstrap_repeats_per_trial": bootstrap_repeats,
        "mean_estimated_delta": float(estimates.mean()),
        "expected_ci_width": float(widths.mean()),
        "median_ci_width": float(np.median(widths)),
        "pass_probability": float(passed.mean()),
    }
