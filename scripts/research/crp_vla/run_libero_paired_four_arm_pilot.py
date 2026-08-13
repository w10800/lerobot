#!/usr/bin/env python
"""Run a small, strictly paired four-arm pilot on ordinary LIBERO."""

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ARM_SPECS = {
    "base10": ("base", 10, None),
    "base1": ("base", 1, None),
    "snap10": ("snap", 10, None),
    "snap1": ("snap", 1, 0.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--snap-checkpoint", type=Path, required=True)
    parser.add_argument("--snap-revision", required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--execution-horizon", type=int, default=10)
    parser.add_argument("--noise-seed", type=int, default=20260813)
    parser.add_argument("--bootstrap-repeats", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260813)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def git_repository_value(repository: Path, *args: str) -> str:
    resolved = repository.resolve()
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={resolved}", "-C", str(resolved), *args],
        text=True,
    ).strip()


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def observation_hashes(observation: dict[str, Any]) -> dict[str, str]:
    hashes = {}

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, item in sorted(value.items()):
                visit(f"{prefix}.{key}" if prefix else key, item)
        elif isinstance(value, np.ndarray):
            hashes[prefix] = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()

    visit("", observation)
    return hashes


def add_batch_dimension(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: add_batch_dimension(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return np.expand_dims(value, axis=0)
    return value


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [float("nan"), float("nan")]
    probability = successes / total
    denominator = 1 + z**2 / total
    center = (probability + z**2 / (2 * total)) / denominator
    half_width = z * math.sqrt(
        probability * (1 - probability) / total + z**2 / (4 * total**2)
    ) / denominator
    return [center - half_width, center + half_width]


def paired_summary(results: list[dict], bootstrap_repeats: int, bootstrap_seed: int) -> dict:
    by_case = {}
    for result in results:
        key = (result["suite"], result["task_id"], result["init_state_id"], result["env_seed"])
        by_case.setdefault(key, {})[result["arm"]] = result["success"]
    if any(set(arms) != set(ARM_SPECS) for arms in by_case.values()):
        raise ValueError("Every pilot case must contain exactly four arms")
    output = {"arm_success": {}, "paired_vs_base10": {}}
    for arm in ARM_SPECS:
        values = [int(arms[arm]) for arms in by_case.values()]
        successes = sum(values)
        output["arm_success"][arm] = {
            "successes": successes,
            "rollouts": len(values),
            "rate": successes / len(values),
            "wilson_ci95": wilson_interval(successes, len(values)),
        }
    for arm in ("base1", "snap10", "snap1"):
        paired_differences = np.asarray(
            [int(arms[arm]) - int(arms["base10"]) for arms in by_case.values()],
            dtype=np.float64,
        )
        generator = np.random.default_rng(bootstrap_seed + list(ARM_SPECS).index(arm))
        bootstrap_indices = generator.integers(
            0,
            len(paired_differences),
            size=(bootstrap_repeats, len(paired_differences)),
        )
        bootstrap_means = paired_differences[bootstrap_indices].mean(axis=1)
        base_success_candidate_failure = sum(
            bool(arms["base10"]) and not bool(arms[arm]) for arms in by_case.values()
        )
        base_failure_candidate_success = sum(
            not bool(arms["base10"]) and bool(arms[arm]) for arms in by_case.values()
        )
        output["paired_vs_base10"][arm] = {
            "success_rate_difference": (
                output["arm_success"][arm]["rate"] - output["arm_success"]["base10"]["rate"]
            ),
            "paired_bootstrap_ci95": [
                float(np.quantile(bootstrap_means, 0.025)),
                float(np.quantile(bootstrap_means, 0.975)),
            ],
            "base_success_candidate_failure": base_success_candidate_failure,
            "base_failure_candidate_success": base_failure_candidate_success,
            "discordant_pair_rate": (
                base_success_candidate_failure + base_failure_candidate_success
            )
            / len(by_case),
        }
    return output


def configure_standard_libero(root: Path, config_dir: Path) -> None:
    package_parent = root / "libero"
    benchmark_root = package_parent / "libero"
    required = (
        benchmark_root / "assets" / "scenes" / "libero_tabletop_base_style.xml",
        benchmark_root / "bddl_files" / "libero_spatial",
        benchmark_root / "init_files" / "libero_spatial",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Pinned LIBERO checkout is incomplete: {missing}")
    config_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "benchmark_root": str(benchmark_root.resolve()),
        "bddl_files": str((benchmark_root / "bddl_files").resolve()),
        "init_states": str((benchmark_root / "init_files").resolve()),
        "datasets": str((package_parent / "datasets").resolve()),
        "assets": str((benchmark_root / "assets").resolve()),
    }
    (config_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir.resolve())


def main() -> None:
    args = parse_args()
    if args.execution_horizon < 1:
        raise ValueError("execution-horizon must be positive")
    if args.bootstrap_repeats < 1:
        raise ValueError("bootstrap-repeats must be positive")
    if git_value("status", "--porcelain"):
        raise ValueError("Repository must be clean before the admitted paired pilot")
    design = json.loads(args.design.read_text())
    cases = design.get("cases", [])
    if not cases:
        raise ValueError("Pilot design contains no cases")
    case_keys = [(case["suite"], case["task_id"], case["init_state_id"], case["env_seed"]) for case in cases]
    if len(case_keys) != len(set(case_keys)):
        raise ValueError("Pilot design contains duplicate cases")
    configure_standard_libero(args.libero_root, args.output.parent / "libero_standard_config")
    repository_commit = git_value("rev-parse", "HEAD")
    libero_commit = git_repository_value(args.libero_root, "rev-parse", "HEAD")
    base_checkpoint_sha256 = file_sha256(args.base_checkpoint / "model.safetensors")
    snap_checkpoint_sha256 = file_sha256(args.snap_checkpoint / "model.safetensors")

    import libero.libero as libero_module
    from libero.libero import benchmark

    expected_assets = (args.libero_root / "libero" / "libero" / "assets").resolve()
    # This LIBERO release's ``get_assets_path`` bypasses ``get_libero_path``
    # and only checks site-packages or its private cache. Pin the cache to the
    # same versioned checkout already validated by ``configure_standard_libero``.
    libero_module._assets_path_cache = str(expected_assets)
    if Path(libero_module.get_assets_path()).resolve() != expected_assets:
        raise RuntimeError("LIBERO asset path did not resolve to the pinned checkout")

    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.libero import TASK_SUITE_MAX_STEPS, LiberoEnv
    from lerobot.envs.utils import preprocess_observation
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.constants import ACTION

    def load_policy(checkpoint: Path, revision: str):
        config = SmolVLAConfig.from_pretrained(checkpoint)
        config.device = args.device
        config.load_vlm_weights = False
        config.n_action_steps = args.execution_horizon
        policy = SmolVLAPolicy.from_pretrained(checkpoint, config=config, revision=revision, strict=False)
        preprocessor, postprocessor = make_pre_post_processors(
            config,
            str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": args.device}},
        )
        return policy, preprocessor, postprocessor

    policies = {
        "base": load_policy(args.base_checkpoint, args.base_revision),
        "snap": load_policy(args.snap_checkpoint, args.snap_revision),
    }
    benchmark_factories = benchmark.get_benchmark_dict()
    requested_suites = sorted({case["suite"] for case in cases})
    unknown_suites = [name for name in requested_suites if name not in benchmark_factories]
    if unknown_suites:
        raise ValueError(f"Unknown suites: {unknown_suites}")
    suites = {name: benchmark_factories[name]() for name in requested_suites}
    results = []
    for case_index, case in enumerate(cases):
        suite_name = case["suite"]
        if suite_name not in suites:
            raise ValueError(f"Unknown suite: {suite_name}")
        suite = suites[suite_name]
        task_id = int(case["task_id"])
        init_state_id = int(case["init_state_id"])
        env_seed = int(case["env_seed"])
        env_config = LiberoEnvConfig(
            task=suite_name,
            task_ids=[task_id],
            observation_height=256,
            observation_width=256,
        )
        env_preprocessor, env_postprocessor = env_config.get_env_processors()
        max_steps = int(case.get("max_steps", TASK_SUITE_MAX_STEPS[suite_name]))
        env = LiberoEnv(
            task_suite=suite,
            task_id=task_id,
            task_suite_name=suite_name,
            episode_length=max_steps,
            camera_name=env_config.camera_name,
            obs_type=env_config.obs_type,
            render_mode=env_config.render_mode,
            observation_width=env_config.observation_width,
            observation_height=env_config.observation_height,
            init_states=env_config.init_states,
            episode_index=init_state_id,
            n_envs=1,
            num_steps_wait=10,
            camera_name_mapping=env_config.camera_name_mapping,
            control_freq=env_config.fps,
            control_mode=env_config.control_mode,
            is_libero_plus=env_config.is_libero_plus,
            hard_reset=env_config.hard_reset,
        )
        try:
            env.init_state_id = init_state_id
            env.reset(seed=env_seed)
            if env._env is None:
                raise RuntimeError("LIBERO inner environment was not initialized")
            initial_sim_state = np.asarray(env._env.get_sim_state()).copy()
            initial_sim_state_sha256 = hashlib.sha256(initial_sim_state.tobytes()).hexdigest()
            canonical_raw = env._env.set_init_state(initial_sim_state)
            canonical_observation = env._format_raw_obs(canonical_raw)
            initial_observation_hash = observation_hashes(canonical_observation)
            prompt = suite.get_task(task_id).language

            for arm, (policy_key, num_steps, target_time) in ARM_SPECS.items():
                policy, preprocessor, postprocessor = policies[policy_key]
                policy.reset()
                policy.config.num_steps = num_steps
                env.init_state_id = init_state_id
                env.reset(seed=env_seed)
                env._env.set_init_state(initial_sim_state)
                restored_state = np.asarray(env._env.get_sim_state())
                if hashlib.sha256(restored_state.tobytes()).hexdigest() != initial_sim_state_sha256:
                    raise RuntimeError(f"Exact simulator state restore failed for {arm}")
                observation = copy.deepcopy(canonical_observation)
                if observation_hashes(observation) != initial_observation_hash:
                    raise RuntimeError(f"Canonical initial observation mismatch for {arm}")

                action_queue: deque[torch.Tensor] = deque()
                noise_hashes = []
                action_hasher = hashlib.sha256()
                success = False
                steps_run = 0
                for step in range(max_steps):
                    if not action_queue:
                        batch = preprocess_observation(add_batch_dimension(observation))
                        batch["task"] = [prompt]
                        batch = preprocessor(env_preprocessor(batch))
                        replan_seed = args.noise_seed + case_index * 100_000 + step
                        generator = torch.Generator(device=args.device).manual_seed(replan_seed)
                        noise = torch.randn(
                            (1, policy.config.chunk_size, policy.config.max_action_dim),
                            generator=generator,
                            device=args.device,
                            dtype=torch.float32,
                        )
                        noise_hashes.append(tensor_sha256(noise))
                        kwargs = {} if target_time is None else {"target_time": target_time}
                        with torch.inference_mode():
                            chunk = policy.predict_action_chunk(batch, noise=noise, **kwargs)
                        action_queue.extend(chunk[0, : args.execution_horizon])
                    action = postprocessor(action_queue.popleft().unsqueeze(0))
                    action = env_postprocessor({ACTION: action})[ACTION]
                    action_numpy = action.detach().cpu().numpy()[0]
                    action_hasher.update(np.ascontiguousarray(action_numpy).tobytes())
                    observation, _, terminated, truncated, info = env.step(action_numpy)
                    steps_run = step + 1
                    success = bool(info["is_success"])
                    if terminated or truncated:
                        break
                results.append(
                    {
                        "suite": suite_name,
                        "task_id": task_id,
                        "task": prompt,
                        "init_state_id": init_state_id,
                        "env_seed": env_seed,
                        "arm": arm,
                        "checkpoint": policy_key,
                        "num_steps": num_steps,
                        "target_time": target_time,
                        "execution_horizon": args.execution_horizon,
                        "max_steps": max_steps,
                        "steps_run": steps_run,
                        "success": success,
                        "initial_sim_state_sha256": initial_sim_state_sha256,
                        "initial_observation_hashes": initial_observation_hash,
                        "noise_sha256_per_replan": noise_hashes,
                        "action_stream_sha256": action_hasher.hexdigest(),
                    }
                )
                partial = {
                    "schema_version": 1,
                    "status": "RUNNING_PARTIAL",
                    "repository_commit": repository_commit,
                    "repository_dirty": False,
                    "libero_commit": libero_commit,
                    "design_sha256": file_sha256(args.design),
                    "base_checkpoint_sha256": base_checkpoint_sha256,
                    "snap_checkpoint_sha256": snap_checkpoint_sha256,
                    "completed_rollout_count": len(results),
                    "planned_rollout_count": len(cases) * len(ARM_SPECS),
                    "results": results,
                }
                partial_path = args.output.with_suffix(".partial.json")
                partial_path.parent.mkdir(parents=True, exist_ok=True)
                partial_path.write_text(json.dumps(partial, indent=2, sort_keys=True) + "\n")
                print(
                    f"completed_case={case_index + 1}/{len(cases)} arm={arm} success={int(success)} steps={steps_run}",
                    flush=True,
                )
        finally:
            env.close()

    for case_key in case_keys:
        case_results = [
            result
            for result in results
            if (result["suite"], result["task_id"], result["init_state_id"], result["env_seed"])
            == case_key
        ]
        reference = case_results[0]
        for result in case_results[1:]:
            if result["initial_sim_state_sha256"] != reference["initial_sim_state_sha256"]:
                raise RuntimeError(f"Initial simulator state mismatch within case {case_key}")
            if result["initial_observation_hashes"] != reference["initial_observation_hashes"]:
                raise RuntimeError(f"Initial policy observation mismatch within case {case_key}")

    output = {
        "schema_version": 1,
        "status": "PILOT_ANALYZED",
        "repository_commit": repository_commit,
        "repository_dirty": False,
        "design": str(args.design.resolve()),
        "design_sha256": file_sha256(args.design),
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "snap_checkpoint_sha256": snap_checkpoint_sha256,
        "libero_repository": str(args.libero_root.resolve()),
        "libero_commit": libero_commit,
        "execution_horizon": args.execution_horizon,
        "noise_seed": args.noise_seed,
        "bootstrap_repeats": args.bootstrap_repeats,
        "bootstrap_seed": args.bootstrap_seed,
        "case_count": len(cases),
        "rollout_count": len(results),
        "summary": paired_summary(results, args.bootstrap_repeats, args.bootstrap_seed),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
