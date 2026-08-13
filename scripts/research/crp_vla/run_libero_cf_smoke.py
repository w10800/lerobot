#!/usr/bin/env python
"""Run two LIBERO-CF conditions from one initial state with a SmolVLA policy."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--libero-cf-root", type=Path, required=True)
    parser.add_argument("--suite", default="libero_cf_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--env-seed", type=int, default=0)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--execution-horizon", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--factual-prompt", required=True)
    parser.add_argument("--factual-condition", required=True)
    parser.add_argument("--counterfactual-prompt", required=True)
    parser.add_argument("--counterfactual-condition", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def git_value(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def configure_libero_cf(root: Path, config_dir: Path) -> None:
    package_parent = root / "libero"
    benchmark_root = package_parent / "libero"
    if not (benchmark_root / "conditions" / "libero_cf_spatial.json").is_file():
        raise FileNotFoundError(f"Invalid LIBERO-CF checkout: {root}")
    config_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "benchmark_root": str(benchmark_root.resolve()),
        "bddl_files": str((benchmark_root / "bddl_files").resolve()),
        "init_states": str((benchmark_root / "init_files").resolve()),
        "datasets": str((package_parent / "datasets").resolve()),
        "assets": str((benchmark_root / "assets").resolve()),
        "conditions": str((benchmark_root / "conditions").resolve()),
    }
    (config_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir.resolve())

    # LIBERO-CF's unused legacy vector-env module still imports ``gym`` during
    # package initialization. The active environment uses the compatible API
    # exposed by Gymnasium, so provide an import alias without installing an
    # obsolete Gym release into the locked training environment.
    import gymnasium
    import libero

    sys.modules.setdefault("gym", gymnasium)
    path = str(package_parent.resolve())
    if path not in libero.__path__:
        libero.__path__.insert(0, path)
    for module_name in tuple(sys.modules):
        if module_name.startswith("libero.libero"):
            del sys.modules[module_name]


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


def main() -> None:
    args = parse_args()
    if args.num_steps < 1 or args.execution_horizon < 1 or args.max_steps < 1:
        raise ValueError("num-steps, execution-horizon, and max-steps must be positive")
    checkpoint_model = args.checkpoint / "model.safetensors"
    if not checkpoint_model.is_file():
        raise FileNotFoundError(checkpoint_model)

    config_dir = args.output.parent / "libero_config"
    configure_libero_cf(args.libero_cf_root, config_dir)

    from libero.libero import benchmark

    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.libero import LiberoEnv
    from lerobot.envs.utils import preprocess_observation
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.constants import ACTION

    suites = benchmark.get_benchmark_dict()
    if args.suite not in suites:
        raise ValueError(f"Unknown LIBERO-CF suite {args.suite!r}")
    suite = suites[args.suite]()
    if not 0 <= args.task_id < suite.n_tasks:
        raise IndexError(f"task-id {args.task_id} outside suite with {suite.n_tasks} tasks")

    policy_config = SmolVLAConfig.from_pretrained(args.checkpoint)
    policy_config.device = args.device
    policy_config.load_vlm_weights = False
    policy_config.num_steps = args.num_steps
    policy_config.n_action_steps = args.execution_horizon
    policy = SmolVLAPolicy.from_pretrained(
        args.checkpoint,
        config=policy_config,
        revision=args.checkpoint_revision,
        strict=False,
    )
    preprocessor, postprocessor = make_pre_post_processors(
        policy_config,
        str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    env_config = LiberoEnvConfig(
        task=args.suite,
        task_ids=[args.task_id],
        observation_height=256,
        observation_width=256,
    )
    env_preprocessor, env_postprocessor = env_config.get_env_processors()

    branches = (
        ("factual", args.factual_prompt, args.factual_condition),
        ("counterfactual", args.counterfactual_prompt, args.counterfactual_condition),
    )
    initial_hashes = None
    results = []
    for branch_name, prompt, condition in branches:
        env = LiberoEnv(
            task_suite=suite,
            task_id=args.task_id,
            task_suite_name=args.suite,
            episode_length=args.max_steps,
            camera_name=env_config.camera_name,
            obs_type=env_config.obs_type,
            render_mode=env_config.render_mode,
            observation_width=env_config.observation_width,
            observation_height=env_config.observation_height,
            init_states=env_config.init_states,
            episode_index=args.init_state_id,
            n_envs=1,
            num_steps_wait=10,
            camera_name_mapping=env_config.camera_name_mapping,
            control_freq=env_config.fps,
            control_mode=env_config.control_mode,
            is_libero_plus=env_config.is_libero_plus,
            hard_reset=env_config.hard_reset,
        )
        try:
            policy.reset()
            observation, _ = env.reset(seed=args.env_seed)
            branch_hashes = observation_hashes(observation)
            if initial_hashes is None:
                initial_hashes = branch_hashes
            elif branch_hashes != initial_hashes:
                raise RuntimeError("LIBERO-CF branches did not start from byte-identical observations")
            if env._env is None:
                raise RuntimeError("LIBERO-CF inner environment was not initialized")
            env._env.set_success_any_conditions([condition])

            action_hasher = hashlib.sha256()
            noise_hashes = []
            terminated = False
            truncated = False
            steps_run = 0
            for step in range(args.max_steps):
                batch = preprocess_observation(observation)
                batch["task"] = [prompt]
                batch = env_preprocessor(batch)
                batch = preprocessor(batch)
                noise = None
                if policy._check_get_actions_condition():
                    generator = torch.Generator(device=args.device).manual_seed(args.noise_seed + step)
                    noise = torch.randn(
                        (1, policy_config.chunk_size, policy_config.max_action_dim),
                        generator=generator,
                        device=args.device,
                        dtype=torch.float32,
                    )
                    noise_hashes.append(tensor_sha256(noise))
                with torch.inference_mode():
                    action = policy.select_action(batch, noise=noise)
                action = postprocessor(action)
                action = env_postprocessor({ACTION: action})[ACTION]
                action_numpy = action.detach().cpu().numpy()[0]
                action_hasher.update(np.ascontiguousarray(action_numpy).tobytes())
                observation, _, terminated, truncated, _ = env.step(action_numpy)
                steps_run = step + 1
                if terminated or truncated:
                    break
            condition_success = bool(env._env.evaluate_conditions([condition]).get(condition, False))
            results.append(
                {
                    "branch": branch_name,
                    "prompt": prompt,
                    "condition": condition,
                    "steps_run": steps_run,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "condition_success": condition_success,
                    "action_stream_sha256": action_hasher.hexdigest(),
                    "noise_sha256_per_replan": noise_hashes,
                    "initial_observation_hashes": branch_hashes,
                }
            )
        finally:
            env.close()

    output = {
        "schema_version": 1,
        "status": "SMOKE",
        "repository_commit": git_value("rev-parse", "HEAD"),
        "repository_dirty": bool(git_value("status", "--porcelain")),
        "libero_cf_commit": git_value("rev-parse", "HEAD", cwd=args.libero_cf_root),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_revision": args.checkpoint_revision,
        "checkpoint_sha256": file_sha256(checkpoint_model),
        "suite": args.suite,
        "task_id": args.task_id,
        "init_state_id": args.init_state_id,
        "env_seed": args.env_seed,
        "noise_seed": args.noise_seed,
        "num_steps": args.num_steps,
        "execution_horizon": args.execution_horizon,
        "max_steps": args.max_steps,
        "device": args.device,
        "branches_share_exact_initial_observation": True,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
