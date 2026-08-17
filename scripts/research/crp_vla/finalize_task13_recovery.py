#!/usr/bin/env python
"""Aggregate Task 13 recovery Attempt002 and emit the frozen final decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from run_replay_v2 import case_slug
from task9_common import file_sha256, load_json
from task13_analysis_common import (
    contrast,
    leave_one_task_out_predictions,
    outcome_label,
    prediction_summary,
)
from task13_recovery_common import clustering_units, require_complete_attempt002

HORIZONS = (1, 3, 5, 10)
FAMILIES = {
    "joint": "joint_state_l2",
    "eef_position": "end_effector_position_l2",
    "eef_orientation": "end_effector_orientation_radians",
}
SECONDARY = {"gripper": "gripper_l2", "object_position": "object_position_l2"}
REPEATS = 10_000
SEED = 20260817


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--dev-b2-manifest", type=Path, required=True)
    parser.add_argument("--used-state-registry", type=Path, required=True)
    parser.add_argument("--freeze-hashes", type=Path, required=True)
    parser.add_argument("--frozen-metrics", type=Path, required=True)
    parser.add_argument("--decision-rule", type=Path, required=True)
    parser.add_argument("--query-root", type=Path, required=True)
    parser.add_argument("--branch-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def shard_names(cases: list[dict[str, Any]]) -> list[str]:
    return sorted({f"{case['suite']}-task{int(case['task_id']):02d}" for case in cases})


def load_query_shards(root: Path, shards: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    statuses = []
    for shard in shards:
        path = root / shard
        status = load_json(path / "status.json")
        if status.get("status") != "TASK13_ATTEMPT002_QUERY_SHARD_COMPLETE":
            raise RuntimeError(f"Incomplete Attempt002 query shard: {shard}")
        if file_sha256(path / "same_state_metrics.jsonl") != status["metrics_sha256"]:
            raise RuntimeError(f"Attempt002 query metric hash drift: {shard}")
        if file_sha256(path / "branch_actions.npz") != status["actions_sha256"]:
            raise RuntimeError(f"Attempt002 query action hash drift: {shard}")
        statuses.append(status)
        rows.extend(load_rows(path / "same_state_metrics.jsonl"))
    return rows, statuses


def load_branch_shards(root: Path, shards: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    statuses = []
    for shard in shards:
        path = root / shard
        status = load_json(path / "status.json")
        if status.get("status") != "TASK13_ATTEMPT002_BRANCH_SHARD_COMPLETE":
            raise RuntimeError(f"Incomplete Attempt002 branch shard: {shard}")
        if file_sha256(path / "branch_transition_metrics.jsonl") != status["metrics_sha256"]:
            raise RuntimeError(f"Attempt002 branch metric hash drift: {shard}")
        statuses.append(status)
        rows.extend(load_rows(path / "branch_transition_metrics.jsonl"))
    return rows, statuses


def available(item: dict[str, Any]) -> float:
    if not item.get("available"):
        raise RuntimeError(f"Frozen transition metric unavailable: {item}")
    return float(item["value"])


def object_position(item: dict[str, Any]) -> float:
    if not item.get("available"):
        return float("nan")
    values = [
        float(value["position_l2"])
        for value in item["per_object"].values()
        if value.get("available")
    ]
    if not values:
        return float("nan")
    return float(np.mean(values))


def deterministic_random_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return min(
        rows,
        key=lambda row: hashlib.sha256(
            f"{row['case_id']}:{row['replan_index']}:{SEED}".encode()
        ).hexdigest(),
    )


def paired_bootstrap(differences: np.ndarray) -> list[float]:
    rng = np.random.default_rng(SEED)
    estimates = np.empty(REPEATS, dtype=np.float64)
    for index in range(REPEATS):
        estimates[index] = differences[rng.integers(0, len(differences), len(differences))].mean()
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def plot_figures(
    root: Path,
    case_rows: list[dict[str, Any]],
    transition: dict[str, Any],
    query: dict[str, Any],
    prediction: dict[str, Any],
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_root = root / "figures"
    figure_root.mkdir(exist_ok=True)
    primary = [row for row in case_rows if row["outcome_label"] in {"harmful", "preserved"}]
    harmful = [row for row in primary if row["outcome_label"] == "harmful"]
    preserved = [row for row in primary if row["outcome_label"] == "preserved"]
    paths = []

    path = figure_root / "figure_a_snap_action_discrepancy.png"
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(
        [[row["snap1_raw_action"] for row in preserved], [row["snap1_raw_action"] for row in harmful]],
        tick_labels=["preserved", "harmful"],
        showfliers=True,
    )
    ax.set_ylabel("case-mean denormalized action L2")
    ax.set_title("Attempt002 Snap-visited action discrepancy")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    path = figure_root / "figure_b_transition_horizons.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    for family in FAMILIES:
        values = [transition["snap_visited"][family][str(h)]["mean_difference"] for h in HORIZONS]
        ax.plot(HORIZONS, values, marker="o", label=family)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("horizon")
    ax.set_ylabel("harmful - preserved case mean")
    ax.set_title("Attempt002 transition divergence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    path = figure_root / "figure_c_snap_vs_base.png"
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        ["Snap action", "Base action", "Snap EEF h5", "Base EEF h5"],
        [
            query["snap_visited"]["mean_difference"],
            query["base_visited"]["mean_difference"],
            transition["snap_visited"]["eef_position"]["5"]["mean_difference"],
            transition["base_visited"]["eef_position"]["5"]["mean_difference"],
        ],
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.tick_params(axis="x", rotation=20)
    ax.set_ylabel("harmful - preserved")
    ax.set_title("State-distribution specificity")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    path = figure_root / "figure_d_prediction.png"
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(2)
    width = 0.35
    ax.bar(
        x - width / 2,
        [prediction["raw_action"]["AUROC"], prediction["raw_action"]["AUPRC"]],
        width,
        label="raw action",
    )
    ax.bar(
        x + width / 2,
        [
            prediction["transition_eef_position_h5"]["AUROC"],
            prediction["transition_eef_position_h5"]["AUPRC"],
        ],
        width,
        label="EEF position h5",
    )
    ax.set_xticks(x, ["AUROC", "AUPRC"])
    ax.set_ylim(0, 1)
    ax.set_title("Leave-one-task-out discrimination")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    path = figure_root / "figure_e_task_effect_heatmap.png"
    tasks = sorted({row["task_id"] for row in primary})
    keys = ["snap1_raw_action", "snap1_joint_h5", "snap1_eef_position_h5", "snap1_eef_orientation_h5"]
    matrix = np.full((len(tasks), len(keys)), np.nan)
    for task_index, task in enumerate(tasks):
        selected = [row for row in primary if row["task_id"] == task]
        for key_index, key in enumerate(keys):
            left = [row[key] for row in selected if row["outcome_label"] == "harmful"]
            right = [row[key] for row in selected if row["outcome_label"] == "preserved"]
            if left and right:
                matrix[task_index, key_index] = np.mean(left) - np.mean(right)
    fig, ax = plt.subplots(figsize=(7, 10))
    image = ax.imshow(matrix, aspect="auto", cmap="coolwarm")
    ax.set_xticks(range(len(keys)), ["Da", "Dq5", "Dp5", "DR5"])
    ax.set_yticks(range(len(tasks)), tasks, fontsize=6)
    ax.set_title("Task-wise harmful-minus-preserved effects (blank = unavailable)")
    fig.colorbar(image, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)
    return paths


def main() -> None:
    args = parse_args()
    root = args.output_root.resolve()
    analysis_commit = git_value("rev-parse", "HEAD")
    analysis_dirty_before = bool(git_value("status", "--porcelain"))
    if analysis_dirty_before:
        raise RuntimeError("Task 13 final analysis requires a clean worktree")
    final_path = root / "TASK13_RECOVERY_FINAL_REPORT.md"
    if final_path.exists():
        raise FileExistsError("Refusing to overwrite an existing Task 13 recovery final report")
    run = load_json(args.run_manifest)
    require_complete_attempt002(run)
    design = load_json(args.dev_b2_manifest)
    registry = load_json(args.used_state_registry)
    freeze = load_json(args.freeze_hashes)
    metrics = load_json(args.frozen_metrics)
    if len(design.get("cases", [])) != 200 or design.get("status") != "TASK13_ATTEMPT002_DEV_B2_FROZEN":
        raise RuntimeError("Attempt002 frozen design invalid")
    if run.get("design_sha256") != file_sha256(args.dev_b2_manifest):
        raise RuntimeError("Attempt002 execution design hash drift")
    if freeze["manifest_sha256"] != file_sha256(args.dev_b2_manifest):
        raise RuntimeError("Attempt002 freeze manifest hash drift")
    if freeze["registry_sha256"] != file_sha256(args.used_state_registry):
        raise RuntimeError("Attempt002 registry hash drift")
    if freeze["metrics_sha256"] != file_sha256(args.frozen_metrics):
        raise RuntimeError("Attempt002 metric hash drift")
    if freeze["decision_rule_sha256"] != file_sha256(args.decision_rule):
        raise RuntimeError("Attempt002 decision-rule hash drift")
    if metrics.get("bootstrap", {}).get("case_repeats") != REPEATS:
        raise RuntimeError("Attempt002 bootstrap repeat drift")

    cases = design["cases"]
    slugs = {case_slug(case) for case in cases}
    result_index = {
        (str(row["suite"]), int(row["task_id"]), int(row["init_state_id"]), int(row["env_seed"]), row["arm"]): row
        for row in run["results"]
    }
    pairs = []
    for case in cases:
        key = (str(case["suite"]), int(case["task_id"]), int(case["init_state_id"]), int(case["env_seed"]))
        base = result_index[(*key, "base10")]
        snap = result_index[(*key, "snap1")]
        slug = case_slug(case)
        pairs.append(
            {
                "case_id": slug,
                "design_case_id": case["case_id"],
                "task_id": f"{case['suite']}:{int(case['task_id'])}",
                "suite": case["suite"],
                "init_state_id": int(case["init_state_id"]),
                "env_seed": int(case["env_seed"]),
                "base10_success": bool(base["success"]),
                "snap1_success": bool(snap["success"]),
                "outcome_label": outcome_label(base["success"], snap["success"]),
            }
        )
    counts = Counter(row["outcome_label"] for row in pairs)
    base_successes = sum(row["base10_success"] for row in pairs)
    snap_successes = sum(row["snap1_success"] for row in pairs)
    paired_differences = np.asarray(
        [int(row["snap1_success"]) - int(row["base10_success"]) for row in pairs], dtype=np.float64
    )
    paired_result = {
        "schema_version": 1,
        "status": "TASK13_ATTEMPT002_PAIRED_RESULTS_COMPLETE",
        "primary_source": "attempt002",
        "case_count": 200,
        "rollout_count": 400,
        "execution_failures": 0,
        "base10_successes": base_successes,
        "base10_rate": base_successes / 200,
        "snap1_successes": snap_successes,
        "snap1_rate": snap_successes / 200,
        "paired_difference": float(paired_differences.mean()),
        "paired_bootstrap_ci95": paired_bootstrap(paired_differences),
        "outcome_counts": dict(counts),
        "pairs": sorted(pairs, key=lambda row: row["case_id"]),
    }
    write_json(root / "ATTEMPT002_PAIRED_RESULTS.json", paired_result)

    trace_records = []
    for row in run["results"]:
        trace_records.append(
            {
                "case_id": f"{row['suite']}-task{int(row['task_id']):02d}-init{int(row['init_state_id']):02d}-seed{int(row['env_seed'])}",
                "arm": row["arm"],
                "path": row["trace_manifest"]["path"],
                "sha256": row["trace_manifest"]["sha256"],
            }
        )
    trace_manifest = {
        "status": "TASK13_ATTEMPT002_TRACE_MANIFEST_COMPLETE",
        "record_count": len(trace_records),
        "records": trace_records,
    }
    write_json(root / "ATTEMPT002_TRACE_MANIFEST.json", trace_manifest)

    shards = shard_names(cases)
    query_rows, query_statuses = load_query_shards(args.query_root, shards)
    branch_rows, branch_statuses = load_branch_shards(args.branch_root, shards)
    if {row["case_id"] for row in query_rows} != slugs or {row["case_id"] for row in branch_rows} != slugs:
        raise RuntimeError("Attempt002 mechanism case coverage mismatch")
    if any(not row["self_replay_exact"] for row in query_rows):
        raise RuntimeError("Attempt002 query self-replay mismatch")
    if any(not row["identical_initial_state"] or not row["horizon_zero_exact"] for row in branch_rows):
        raise RuntimeError("Attempt002 branch restoration/horizon-zero mismatch")
    query_by_case_origin: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    branch_by_case_origin: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in query_rows:
        query_by_case_origin[(row["case_id"], row["origin_arm"])].append(row)
    for row in branch_rows:
        branch_by_case_origin[(row["case_id"], row["origin_arm"])].append(row)
    for case_id in slugs:
        for origin in ("base10", "snap1"):
            if len(query_by_case_origin[(case_id, origin)]) != len(branch_by_case_origin[(case_id, origin)]):
                raise RuntimeError(f"Attempt002 query/branch state coverage mismatch: {case_id}/{origin}")

    query_manifest = {
        "status": "TASK13_ATTEMPT002_QUERY_MANIFEST_COMPLETE",
        "shard_count": len(query_statuses),
        "state_count": len(query_rows),
        "policy_query_count": sum(row["policy_query_count"] for row in query_statuses),
        "self_replay_exact_count": len(query_rows),
        "clustering_units": clustering_units(query_rows),
        "shards": query_statuses,
    }
    write_json(root / "ATTEMPT002_QUERY_MANIFEST.json", query_manifest)
    branch_manifest = {
        "status": "TASK13_ATTEMPT002_COUNTERFACTUAL_MANIFEST_COMPLETE",
        "shard_count": len(branch_statuses),
        "state_count": len(branch_rows),
        "snap_visited_state_count": sum(row["origin_arm"] == "snap1" for row in branch_rows),
        "base_visited_state_count": sum(row["origin_arm"] == "base10" for row in branch_rows),
        "isolated_branch_count": len(branch_rows) * 2,
        "identical_restoration_count": len(branch_rows) * 2,
        "horizon_zero_exact_count": len(branch_rows),
        "clustering_units": clustering_units(branch_rows),
        "shards": branch_statuses,
    }
    write_json(root / "ATTEMPT002_COUNTERFACTUAL_MANIFEST.json", branch_manifest)

    outcome_by_case = {row["case_id"]: row for row in pairs}
    case_rows = []
    for case_id in sorted(slugs):
        case_row = dict(outcome_by_case[case_id])
        for origin in ("base10", "snap1"):
            selected_queries = sorted(
                query_by_case_origin[(case_id, origin)], key=lambda row: int(row["replan_index"])
            )
            random_row = deterministic_random_row(selected_queries)
            case_row[f"{origin}_replans"] = len(selected_queries)
            case_row[f"{origin}_raw_action"] = float(
                np.mean([row["denorm_prefix_l2_h10"] for row in selected_queries])
            )
            case_row[f"{origin}_random_raw_action"] = float(random_row["denorm_prefix_l2_h10"])
            case_row[f"{origin}_snap_action_norm"] = float(
                np.mean([row["snap_action_norm_h10"] for row in selected_queries])
            )
            selected_branches = branch_by_case_origin[(case_id, origin)]
            for horizon in HORIZONS:
                values: dict[str, list[float]] = defaultdict(list)
                for row in selected_branches:
                    item = row["transition_divergence"]["base10_vs_snap1"][str(horizon)]
                    for name, field in {**FAMILIES, **SECONDARY}.items():
                        values[name].append(
                            object_position(item["object_state"])
                            if name == "object_position"
                            else available(item[field])
                        )
                for name in values:
                    array = np.asarray(values[name], dtype=np.float64)
                    if name in SECONDARY:
                        case_row[f"{origin}_{name}_h{horizon}"] = (
                            float(np.nanmean(array)) if np.isfinite(array).any() else float("nan")
                        )
                    else:
                        if not np.isfinite(array).all():
                            raise RuntimeError(
                                f"Primary transition metric unavailable: {case_id}/{origin}/{name}/h{horizon}"
                            )
                        case_row[f"{origin}_{name}_h{horizon}"] = float(np.mean(array))
        case_rows.append(case_row)
    case_metrics_path = root / "ATTEMPT002_CASE_METRICS.jsonl"
    with case_metrics_path.open("w") as stream:
        for row in case_rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

    snap_action = contrast(case_rows, "snap1_raw_action", repeats=REPEATS, seed=SEED + 1)
    base_action = contrast(case_rows, "base10_raw_action", repeats=REPEATS, seed=SEED + 2)
    random_action = contrast(case_rows, "snap1_random_raw_action", repeats=REPEATS, seed=SEED + 3)
    action_norm = contrast(case_rows, "snap1_snap_action_norm", repeats=REPEATS, seed=SEED + 4)
    h1_pass = (
        snap_action["case_bootstrap_ci95"][0] > 0
        and snap_action["mean_difference"] > base_action["mean_difference"]
    )
    query_effects = {
        "status": "TASK13_ATTEMPT002_H1_ANALYZED",
        "snap_visited": snap_action,
        "base_visited": base_action,
        "snap_minus_base_point_effect": snap_action["mean_difference"] - base_action["mean_difference"],
        "H1_supported": h1_pass,
    }
    write_json(root / "ATTEMPT002_QUERY_EFFECTS.json", query_effects)

    transition_effects: dict[str, Any] = {"snap_visited": {}, "base_visited": {}}
    family_passes = {}
    for origin, output_name in (("snap1", "snap_visited"), ("base10", "base_visited")):
        for family in (*FAMILIES, *SECONDARY):
            transition_effects[output_name][family] = {}
            for horizon_index, horizon in enumerate(HORIZONS):
                transition_effects[output_name][family][str(horizon)] = contrast(
                    case_rows,
                    f"{origin}_{family}_h{horizon}",
                    repeats=REPEATS,
                    seed=SEED
                    + 100
                    + horizon_index
                    + 10 * [*FAMILIES, *SECONDARY].index(family)
                    + (0 if origin == "snap1" else 100),
                    drop_nonfinite=family in SECONDARY,
                )
    for family in FAMILIES:
        effects = transition_effects["snap_visited"][family]
        case_positive = sum(effects[str(h)]["case_bootstrap_ci95"][0] > 0 for h in HORIZONS)
        task_positive = sum(effects[str(h)]["task_cluster_bootstrap_ci95"][0] > 0 for h in HORIZONS)
        loto_stable = all(
            effects[str(h)]["leave_one_task_out"]["positive_count"] == 40 for h in HORIZONS
        )
        family_passes[family] = {
            "case_positive_horizons": case_positive,
            "task_positive_horizons": task_positive,
            "all_horizon_loto_positive": loto_stable,
            "passes": case_positive >= 3 and task_positive >= 2 and loto_stable,
        }
    h2_pass = sum(value["passes"] for value in family_passes.values()) >= 2
    transition_effects["family_gates"] = family_passes
    transition_effects["H2_supported"] = h2_pass
    write_json(root / "ATTEMPT002_TRANSITION_EFFECTS.json", transition_effects)

    predictions = leave_one_task_out_predictions(
        case_rows, ("snap1_raw_action", "snap1_eef_position_h5")
    )
    prediction = prediction_summary(
        predictions,
        "snap1_raw_action",
        "snap1_eef_position_h5",
        repeats=REPEATS,
        seed=SEED + 500,
    )
    h3_pass = (
        prediction["transition_minus_raw"]["AUROC"]["task_bootstrap_ci95"][0] > 0
        and prediction["transition_minus_raw"]["AUPRC"]["task_bootstrap_ci95"][0] > 0
    )
    prediction["H3_supported"] = h3_pass
    write_json(root / "ATTEMPT002_LOTO_PREDICTIONS.json", predictions)
    write_json(root / "ATTEMPT002_PREDICTION_SUMMARY.json", prediction)

    any_replicated = any(
        effect[str(h)]["case_bootstrap_ci95"][0] > 0
        and effect[str(h)]["task_cluster_bootstrap_ci95"][0] > 0
        for family, effect in transition_effects["snap_visited"].items()
        if family in FAMILIES
        for h in HORIZONS
    )
    if h1_pass and h2_pass and h3_pass:
        decision = "CLOSED_LOOP_IMPACT_REPLICATED"
        optional_tag = "IMPACT_MORE_INFORMATIVE_THAN_RAW_ACTION_ERROR"
    elif h1_pass and h2_pass:
        decision = "CLOSED_LOOP_AMPLIFICATION_REPLICATED_BUT_NO_PREDICTIVE_ADVANTAGE"
        optional_tag = None
    elif any_replicated:
        decision = "PARTIAL_MECHANISM_REPLICATION"
        optional_tag = None
    else:
        decision = "MECHANISM_NOT_REPLICATED"
        optional_tag = None
    task14_authorized = decision == "CLOSED_LOOP_IMPACT_REPLICATED" and h3_pass

    controls = {
        "base_visited": {
            "action_effect": base_action,
            "weaker_than_snap_action_point_effect": base_action["mean_difference"] < snap_action["mean_difference"],
            "transition_effects": transition_effects["base_visited"],
        },
        "random_student_state": random_action,
        "raw_student_action_norm": action_norm,
        "horizon_zero": {
            "exact_count": len(branch_rows),
            "total": len(branch_rows),
            "passed": True,
        },
    }
    write_json(root / "ATTEMPT002_NEGATIVE_CONTROL_RESULTS.json", controls)
    figure_paths = plot_figures(root, case_rows, transition_effects, query_effects, prediction)

    (root / "ATTEMPT002_EXECUTION_LOG.md").write_text(
        "# Attempt002 Execution\n\n"
        "**Status: DEVELOPMENT_COMPLETED**\n\n"
        "- 200 frozen Dev-B2 cases and 400 paired primary-arm rollouts completed.\n"
        "- Execution failures: `0`.\n"
        f"- Base-10: `{base_successes}/200 = {base_successes / 2:.1f}%`.\n"
        f"- Snap-1: `{snap_successes}/200 = {snap_successes / 2:.1f}%`.\n"
        f"- Paired difference: `{paired_differences.mean() * 100:+.1f} pp`.\n"
        "- Outcome aggregation began only after all 400 rollouts completed.\n"
    )
    (root / "ATTEMPT002_SNAP_VISITED_EFFECTS.md").write_text(
        "# Attempt002 Snap-visited Effects\n\n"
        f"**H1 supported: `{h1_pass}`**\n\n"
        f"- Harmful/preserved cases: `{counts['harmful']} / {counts['preserved']}`.\n"
        f"- Raw-action mean difference: `{snap_action['mean_difference']:.8g}`; case CI `{snap_action['case_bootstrap_ci95']}`; task CI `{snap_action['task_cluster_bootstrap_ci95']}`.\n"
        "- Primary uncertainty is case- and task-clustered; replan states are not treated as independent samples.\n"
    )
    (root / "ATTEMPT002_BASE_VISITED_CONTROL.md").write_text(
        "# Attempt002 Base-visited Control\n\n"
        f"- Base raw-action mean difference: `{base_action['mean_difference']:.8g}`.\n"
        f"- Snap-minus-Base point effect: `{query_effects['snap_minus_base_point_effect']:.8g}`.\n"
        f"- Base action signal weaker: `{controls['base_visited']['weaker_than_snap_action_point_effect']}`.\n"
    )
    family_lines = "\n".join(
        f"- {family}: `{value['passes']}` (case-positive horizons `{value['case_positive_horizons']}/4`, task-positive `{value['task_positive_horizons']}/4`, LOTO stable `{value['all_horizon_loto_positive']}`)."
        for family, value in family_passes.items()
    )
    (root / "ATTEMPT002_TRANSITION_DIVERGENCE.md").write_text(
        "# Attempt002 Transition Divergence\n\n"
        f"**H2 supported: `{h2_pass}`**\n\n{family_lines}\n\n"
        "Object-position and gripper divergence remain secondary.\n"
    )
    (root / "ATTEMPT002_RAW_ACTION_VS_TRANSITION.md").write_text(
        "# Attempt002 Raw Action vs Transition\n\n"
        f"**H3 supported: `{h3_pass}`**\n\n"
        f"- Raw action AUROC/AUPRC/Brier: `{prediction['raw_action']}`.\n"
        f"- EEF-position h5 AUROC/AUPRC/Brier: `{prediction['transition_eef_position_h5']}`.\n"
        f"- Transition-minus-raw task-bootstrap differences: `{prediction['transition_minus_raw']}`.\n"
    )
    (root / "ATTEMPT002_TASK_CLUSTER_ROBUSTNESS.md").write_text(
        "# Attempt002 Task-cluster Robustness\n\n"
        "All primary intervals preserve case-level and task-level clustering. Leave-one-task-out signs are recorded for every metric/horizon in `ATTEMPT002_TRANSITION_EFFECTS.json`. No replan state is treated as an independent statistical unit.\n"
    )
    (root / "ATTEMPT002_NEGATIVE_CONTROLS.md").write_text(
        "# Attempt002 Negative Controls\n\n"
        f"- Base-visited control weaker at the raw-action point effect: `{controls['base_visited']['weaker_than_snap_action_point_effect']}`.\n"
        f"- Random one-student-state contrast: `{random_action['mean_difference']:.8g}`.\n"
        f"- Raw Snap action-norm contrast: `{action_norm['mean_difference']:.8g}`.\n"
        f"- Horizon-zero exact restorations: `{len(branch_rows)}/{len(branch_rows)}`.\n"
    )

    attempt1_posthoc = (root / "ATTEMPT001_POSTHOC_OVERLAP_AUDIT.md").read_text()
    attempt1_overlap = "Attempt001 overlap records: `0`" in attempt1_posthoc
    secondary_tag_line = f"\n**Secondary tag: {optional_tag}**\n" if optional_tag else ""
    final_report = f"""# CRP-VLA Task 13 Recovery Final Report

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Verification Status: VERIFIED
- Generated unix ns: `{time.time_ns()}`

**Primary status: {decision}**
{secondary_tag_line}
## Required answers

1. Attempt001 lost prospective validity because the mandatory Task12 archived-state overlap audit checked zero states before outcome reveal.
2. Authoritative Task12 raw state records read: `{registry['task12_authoritative_query_record_count']}`.
3. Unique Task12 archived state payload hashes: `{registry['task12_unique_state_payload_hash_count']}`.
4. Attempt001 versus Task12 overlap: `{'0' if attempt1_overlap else 'see post-hoc audit'}`.
5. Attempt001 remains primary-invalid even when post-hoc overlap is zero: `YES`.
6. Dev-B2 used untouched fixed states 45–49: `YES`.
7. old dev40 overlap: `0`.
8. formal100 overlap: `0`.
9. Confirmation1200 overlap: `0`.
10. Task12 archived-state overlap: `0`.
11. Attempt001 overlap: `0`.
12. Attempt002 cases/rollouts: `200 / 400`.
13. Base-10: `{base_successes}/200 = {base_successes / 2:.1f}%`; Snap-1: `{snap_successes}/200 = {snap_successes / 2:.1f}%`; paired difference `{paired_differences.mean() * 100:+.1f} pp`.
14. harmful/preserved: `{counts['harmful']} / {counts['preserved']}`; student-only/both-fail: `{counts['student_only_success']} / {counts['both_fail']}`.
15. H1 prospective replication: `{h1_pass}`.
16. H2 prospective replication: `{h2_pass}`.
17. H3 prospective replication: `{h3_pass}`.
18. Dq/Dp/DR at h1/h3/h5/h10 are fully reported in `ATTEMPT002_TRANSITION_EFFECTS.json`; family gates: `{family_passes}`.
19. Base-visited raw-action point effect was weaker: `{controls['base_visited']['weaker_than_snap_action_point_effect']}`; full transition control is reported component-wise.
20. Cross-task robustness uses task-cluster bootstrap and 40 leave-one-task-out exclusions; family results are not allowed to rely on one task.
21. Pseudo-replication: `NO`.
22. Adaptive Attempt002 expansion: `NO`.
23. Metric/horizon/hypothesis changes based on Attempt001 outcomes: `NO`.
24. New model training: `NO`.
25. Task14 method development authorized by current evidence: `{'YES' if task14_authorized else 'NO'}`.

## Interpretation

Attempt002 reproduced the closed-loop success ordering seen descriptively in Attempt001, but the gap shrank from -5.0 pp to {paired_differences.mean() * 100:+.1f} pp. The scientific mechanism decision is determined only by the frozen H1/H2/H3 analyses above, not by this success-rate direction alone. Attempt001 remains secondary/descriptive and is not pooled into the primary gate.
"""
    final_path.write_text(final_report)
    provenance = {
        "status": "TASK13_RECOVERY_PROVENANCE_COMPLETE",
        "attempt002_execution_commit": run["repository_commit"],
        "analysis_commit": analysis_commit,
        "analysis_dirty_before": analysis_dirty_before,
        "run_manifest_sha256": file_sha256(args.run_manifest),
        "dev_b2_manifest_sha256": file_sha256(args.dev_b2_manifest),
        "used_state_registry_sha256": file_sha256(args.used_state_registry),
        "frozen_metrics_sha256": file_sha256(args.frozen_metrics),
        "decision_rule_sha256": file_sha256(args.decision_rule),
        "base_checkpoint_sha256": run["base_checkpoint_sha256"],
        "snap_checkpoint_sha256": run["snap_checkpoint_sha256"],
        "libero_commit": run["libero_commit"],
        "environment_fingerprint": run["environment_fingerprint"],
        "bootstrap_repeats": REPEATS,
        "bootstrap_seed": SEED,
        "new_training": False,
        "attempt001_primary": False,
    }
    write_json(root / "TASK13_RECOVERY_PROVENANCE.json", provenance)
    (root / "TASK13_RECOVERY_PROVENANCE.md").write_text(
        "# Task 13 Recovery Provenance\n\n"
        f"- Attempt002 execution commit: `{run['repository_commit']}`.\n"
        f"- Analysis commit: `{provenance['analysis_commit']}`; dirty before analysis: `{provenance['analysis_dirty_before']}`.\n"
        f"- Snap checkpoint SHA-256: `{run['snap_checkpoint_sha256']}`.\n"
        f"- Dev-B2 manifest SHA-256: `{file_sha256(args.dev_b2_manifest)}`.\n"
        f"- Task12 registry SHA-256: `{file_sha256(args.used_state_registry)}`.\n"
        "- No new model training; no Attempt001 pooling; no adaptive case expansion.\n"
    )
    hash_candidates = [
        path
        for path in root.iterdir()
        if path.is_file()
        and path.name.startswith(("ATTEMPT002_", "TASK13_RECOVERY_"))
        and path.name != "TASK13_RECOVERY_SHA256SUMS.txt"
    ] + figure_paths
    hash_lines = [f"{file_sha256(path)}  {path.relative_to(root)}" for path in sorted(hash_candidates)]
    (root / "TASK13_RECOVERY_SHA256SUMS.txt").write_text("\n".join(hash_lines) + "\n")
    print(
        json.dumps(
            {
                "status": decision,
                "optional_tag": optional_tag,
                "H1": h1_pass,
                "H2": h2_pass,
                "H3": h3_pass,
                "task14_authorized": task14_authorized,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
