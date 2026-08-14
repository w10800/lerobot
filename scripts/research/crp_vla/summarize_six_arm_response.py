#!/usr/bin/env python
"""Create the six-arm response decomposition, controls, tables, and plots."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ARMS = ("base10", "base2", "base1", "snap10", "snap2", "snap1")
METRICS = (
    "factual_action_mse",
    "normalized_response_error",
    "response_cosine",
    "response_norm_ratio",
    "absolute_response_norm",
    "teacher_response_norm",
)
PAIRWISE = (
    "base_10_to_2",
    "base_2_to_1",
    "base_10_to_1",
    "snap_10_to_2",
    "snap_2_to_1",
    "snap_10_to_1",
    "snap10_vs_base10",
    "snap2_vs_base2",
    "snap1_vs_base1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--fixture-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260814)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric_value(values: dict, metric: str) -> float:
    return float(values["global"].get(metric, values.get(metric)))


def interval(values: np.ndarray, repeats: int, seed: int) -> list[float]:
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(repeats, len(values)))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def describe(values: list[float], repeats: int, seed: int) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p25": float(np.quantile(array, 0.25)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "pair_bootstrap_ci95": interval(array, repeats, seed),
        "n_semantic_pairs": len(array),
    }


def task_cluster_interval(rows: list[dict], value_key: str, repeats: int, seed: int) -> list[float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        grouped[int(row["dataset_task_index"])].append(float(row[value_key]))
    task_means = np.asarray([np.mean(values) for values in grouped.values()])
    return interval(task_means, repeats, seed)


def flatten_records(matrix: dict) -> list[dict]:
    rows = []
    for record in matrix["records"]:
        for arm in ARMS:
            arm_record = record["arms"][arm]
            row = {
                "pair_id": record["pair_id"],
                "source_suite": record["source_suite"],
                "source_task": record["source_task"],
                "dataset_task_index": int(record["dataset_task_index"]),
                "intervention_type": record["intervention_type"],
                "noise_seed": int(record["noise_seed"]),
                "noise_sha256": record["noise_sha256"],
                "arm": arm,
                "accepted": bool(record["accepted"]),
                "matched_noise_seed": int(record["matched_noise_control"]["alternate_noise_seed"]),
                "condition_teacher_norm": float(record["teacher_response_norm"]),
                "matched_noise_teacher_norm": float(
                    record["matched_noise_control"]["teacher_noise_response_norm"]
                ),
            }
            for metric in METRICS:
                row[f"condition_{metric}"] = metric_value(arm_record["condition_change"], metric)
                row[f"noise_{metric}"] = metric_value(arm_record["noise_change_matched"], metric)
            for metric, value in arm_record["g_condition"].items():
                row[f"g_condition_{metric}"] = float(value)
            for profile in ("horizon", "dimension"):
                for metric, values in arm_record["condition_change"][profile].items():
                    row[f"condition_{profile}_{metric}"] = [float(value) for value in values]
            rows.append(row)
    return rows


def aggregate_pairs(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in records:
        if row["accepted"]:
            grouped[(row["pair_id"], row["arm"])].append(row)
    output = []
    scalar_keys = [
        key
        for key in records[0]
        if key.startswith(("condition_", "noise_", "g_condition_"))
        and not isinstance(records[0][key], list)
        and isinstance(records[0][key], (int, float))
    ]
    array_keys = [key for key in records[0] if isinstance(records[0][key], list)]
    for (pair_id, arm), rows in sorted(grouped.items()):
        first = rows[0]
        item = {
            "pair_id": pair_id,
            "arm": arm,
            "source_suite": first["source_suite"],
            "source_task": first["source_task"],
            "dataset_task_index": first["dataset_task_index"],
            "intervention_type": first["intervention_type"],
            "noise_seed_count": len(rows),
        }
        item.update({key: float(np.mean([row[key] for row in rows])) for key in scalar_keys})
        item.update(
            {key: np.mean(np.asarray([row[key] for row in rows]), axis=0).tolist() for key in array_keys}
        )
        output.append(item)
    return output


def pairwise_statistics(matrix: dict, repeats: int, seed: int) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in matrix["records"]:
        if record["accepted"]:
            grouped[record["pair_id"]].append(record)
    output = {}
    for comparison_index, comparison in enumerate(PAIRWISE):
        output[comparison] = {}
        for metric_index, metric in enumerate(METRICS):
            pair_values = [
                float(
                    np.mean(
                        [
                            metric_value(record["pairwise_condition_change"][comparison], metric)
                            for record in records
                        ]
                    )
                )
                for records in grouped.values()
            ]
            output[comparison][metric] = describe(
                pair_values,
                repeats,
                seed + comparison_index * len(METRICS) + metric_index,
            )
    return output


def statistics(pair_rows: list[dict], matrix: dict, repeats: int, seed: int) -> dict:
    output = {
        "schema_version": 1,
        "status": "DIAGNOSTIC_ANALYZED",
        "independent_unit_primary": "semantic_pair_after_mean_over_3_noise_seeds",
        "independent_unit_secondary": "dataset_task_cluster",
        "semantic_pair_count": len({row["pair_id"] for row in pair_rows}),
        "noise_seed_count_per_pair": 3,
        "arms": {},
        "pairwise_condition_change": pairwise_statistics(matrix, repeats, seed + 1_000),
    }
    for arm_index, arm in enumerate(ARMS):
        rows = [row for row in pair_rows if row["arm"] == arm]
        output["arms"][arm] = {}
        for metric_index, metric in enumerate(METRICS):
            for control in ("condition", "noise"):
                key = f"{control}_{metric}"
                item = describe(
                    [row[key] for row in rows],
                    repeats,
                    seed + arm_index * 100 + metric_index * 2 + int(control == "noise"),
                )
                item["task_cluster_bootstrap_ci95"] = task_cluster_interval(
                    rows, key, repeats, seed + 10_000 + arm_index * 100 + metric_index
                )
                output["arms"][arm][key] = item
        for metric in (
            "normalized_response_error",
            "response_cosine",
            "response_norm_ratio",
        ):
            key = f"g_condition_{metric}"
            item = describe([row[key] for row in rows], repeats, seed + 20_000 + arm_index)
            item["task_cluster_bootstrap_ci95"] = task_cluster_interval(
                rows, key, repeats, seed + 30_000 + arm_index
            )
            output["arms"][arm][key] = item
        horizon = np.asarray([row["condition_horizon_normalized_response_error"] for row in rows]).mean(
            axis=0
        )
        thirds = np.array_split(np.arange(len(horizon)), 3)
        output["arms"][arm]["horizon_segments_response_error"] = {
            label: float(horizon[indices].mean())
            for label, indices in zip(("near", "middle", "far"), thirds, strict=True)
        }
        dimension = np.asarray([row["condition_dimension_normalized_response_error"] for row in rows]).mean(
            axis=0
        )
        output["arms"][arm]["action_group_response_error"] = {
            "translation": float(dimension[:3].mean()),
            "rotation": float(dimension[3:6].mean()),
            "gripper": float(dimension[6]),
        }
    for metric in METRICS:
        snap_reduction = np.asarray(
            [
                output["pairwise_condition_change"]["snap_10_to_1"][metric]["mean"],
                output["pairwise_condition_change"]["base_10_to_1"][metric]["mean"],
            ]
        )
        output.setdefault("snap_step_effect_minus_base_step_effect", {})[metric] = float(
            snap_reduction[0] - snap_reduction[1]
        )
    return output


def write_parquet(rows: list[dict], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def plot_lines(stats: dict, metric: str, ylabel: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    for prefix, label, color in (("base", "Base", "#2563eb"), ("snap", "Snap", "#dc2626")):
        values = [stats["arms"][f"{prefix}{nfe}"][f"condition_{metric}"]["mean"] for nfe in (10, 2, 1)]
        axis.plot((10, 2, 1), values, marker="o", label=label, color=color)
    axis.set_xticks((10, 2, 1))
    axis.set_xlabel("NFE")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def make_plots(pair_rows: list[dict], stats: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    plot_lines(
        stats, "normalized_response_error", "Normalized response error", output / "nfe_vs_response_error.png"
    )
    plot_lines(stats, "response_cosine", "Response cosine", output / "nfe_vs_response_cosine.png")
    plot_lines(stats, "response_norm_ratio", "Response norm ratio", output / "nfe_vs_response_norm_ratio.png")

    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    for arm in ARMS:
        rows = [row for row in pair_rows if row["arm"] == arm]
        values = np.asarray([row["condition_horizon_normalized_response_error"] for row in rows]).mean(axis=0)
        axis.plot(np.arange(len(values)), values, label=arm)
    axis.set(xlabel="Action horizon index", ylabel="Normalized response error")
    axis.grid(alpha=0.2)
    axis.legend(ncol=3, fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "horizon_wise_response_error.png", dpi=180)
    plt.close(figure)

    labels = ("translation", "rotation", "gripper")
    x = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(8, 4.4))
    width = 0.12
    for index, arm in enumerate(ARMS):
        values = [stats["arms"][arm]["action_group_response_error"][label] for label in labels]
        axis.bar(x + (index - 2.5) * width, values, width, label=arm)
    axis.set_xticks(x, labels)
    axis.set_ylabel("Normalized response error")
    axis.legend(ncol=3, fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "action_group_decomposition.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(ARMS))
    condition = [stats["arms"][arm]["condition_normalized_response_error"]["mean"] for arm in ARMS]
    noise = [stats["arms"][arm]["noise_normalized_response_error"]["mean"] for arm in ARMS]
    axis.bar(x - 0.18, condition, 0.36, label="Condition change")
    axis.bar(x + 0.18, noise, 0.36, label="Matched noise change")
    axis.set_xticks(x, ARMS, rotation=25)
    axis.set_ylabel("Normalized response error")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "condition_vs_matched_noise.png", dpi=180)
    plt.close(figure)

    snap1 = [row for row in pair_rows if row["arm"] == "snap1"]
    grouped: dict[int, list[float]] = defaultdict(list)
    names = {}
    for row in snap1:
        grouped[row["dataset_task_index"]].append(row["g_condition_normalized_response_error"])
        names[row["dataset_task_index"]] = row["source_task"]
    task_ids = sorted(grouped)
    means = [float(np.mean(grouped[index])) for index in task_ids]
    figure, axis = plt.subplots(figsize=(8, max(4, 0.45 * len(task_ids))))
    axis.scatter(means, np.arange(len(task_ids)), color="#7c3aed")
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_yticks(np.arange(len(task_ids)), [names[index][:55] for index in task_ids], fontsize=7)
    axis.set_xlabel("Snap-1 G_condition response error (task mean)")
    figure.tight_layout()
    figure.savefig(output / "task_level_effect_forest.png", dpi=180)
    plt.close(figure)


def write_paraphrase_candidates(fixtures: dict, output_dir: Path) -> None:
    records = []
    for fixture in fixtures["records"]:
        records.append(
            {
                "pair_id": fixture["pair_id"],
                "source_suite": fixture["source_suite"],
                "factual_prompt": fixture["factual_prompt"],
                "factual_paraphrase_candidate": f"Please {fixture['factual_prompt'].rstrip('.')}.",
                "counterfactual_prompt": fixture["counterfactual_prompt"],
                "counterfactual_paraphrase_candidate": f"Please {fixture['counterfactual_prompt'].rstrip('.')}.",
                "status": "CANDIDATE_ONLY_NOT_EVALUATED",
            }
        )
    catalog = {
        "schema_version": 1,
        "status": "CANDIDATE_ONLY_NOT_EVALUATED",
        "pair_count": len(records),
        "source_fixture_manifest_sha256": file_sha256(Path(fixtures["_path"])),
        "records": records,
    }
    (output_dir / "paraphrase_candidate_catalog.json").write_text(
        json.dumps(catalog, indent=2, sort_keys=True) + "\n"
    )
    duplicate_count = len(records) - len(
        {
            (record["factual_paraphrase_candidate"], record["counterfactual_paraphrase_candidate"])
            for record in records
        }
    )
    (output_dir / "PARAPHRASE_CANDIDATE_AUDIT.md").write_text(
        "# Paraphrase candidate audit\n\n"
        "**Status: CANDIDATE ONLY — NOT EVALUATED AND NOT INCLUDED IN STATISTICS.**\n\n"
        f"- Candidates: {len(records)}\n"
        f"- Duplicate candidate pairs: {duplicate_count}\n"
        "- Sources: standard LIBERO frozen fixtures only; no LIBERO-CF condition was added.\n"
        "- Transformation: polite surface-form prefix only; semantic equivalence remains unverified and requires freezing before use.\n"
    )


def write_report(stats: dict, path: Path) -> None:
    pairwise = stats["pairwise_condition_change"]
    snap1 = stats["arms"]["snap1"]
    segments = snap1["horizon_segments_response_error"]
    groups = snap1["action_group_response_error"]
    dominant_segment = max(segments, key=segments.get)
    dominant_group = max(groups, key=groups.get)
    cond = snap1["condition_normalized_response_error"]
    noise = snap1["noise_normalized_response_error"]
    gap = snap1["g_condition_normalized_response_error"]
    lines = [
        "# Six-arm conditional-response diagnostic",
        "",
        "**Status: DIAGNOSTIC. No training or checkpoint selection is authorized.**",
        "",
        "## Main arm results",
        "",
        "| Arm | Response error | Cosine | Norm ratio | Factual MSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        item = stats["arms"][arm]
        lines.append(
            f"| {arm} | {item['condition_normalized_response_error']['mean']:.4f} | "
            f"{item['condition_response_cosine']['mean']:.4f} | "
            f"{item['condition_response_norm_ratio']['mean']:.4f} | "
            f"{item['condition_factual_action_mse']['mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Required answers",
            "",
            f"1. Snap-1 response error is {cond['mean']:.4f}; its largest horizon segment is **{dominant_segment}** and largest action-group error is **{dominant_group}**. Direction and magnitude evidence are reported separately by cosine and norm ratio; the decomposition does not force a single-cause label.",
            f"2. Snap 10→2 direct error is {pairwise['snap_10_to_2']['normalized_response_error']['mean']:.4f}; Snap 2→1 is {pairwise['snap_2_to_1']['normalized_response_error']['mean']:.4f}. Base counterparts are {pairwise['base_10_to_2']['normalized_response_error']['mean']:.4f} and {pairwise['base_2_to_1']['normalized_response_error']['mean']:.4f}.",
            f"3. Snap-1 condition-change error ({cond['mean']:.4f}) versus matched-noise error ({noise['mean']:.4f}) gives G_condition={gap['mean']:+.4f}, pair-bootstrap 95% CI [{gap['pair_bootstrap_ci95'][0]:+.4f}, {gap['pair_bootstrap_ci95'][1]:+.4f}].",
            "4. Conditional specificity is supported only when G_condition is positive with compatible cosine and norm-ratio controls. The full machine-readable statistics preserve all three; no single scalar is treated as decisive.",
            "5. The offline fixture catalog does not provide a one-to-one mapping to the 15 closed-loop discordant cases, whose failure categories are UNKNOWN. Therefore alignment with spatial/object closed-loop clustering remains UNVERIFIED.",
            "",
            "## Statistical unit and controls",
            "",
            "Three noise seeds are averaged within each of 40 semantic pairs before inference. Primary intervals bootstrap semantic pairs; secondary intervals bootstrap the seven dataset-task clusters. Noise controls select, within each pair, the alternate registered seed whose Base-10 factual delta norm is closest to the condition-change teacher norm.",
            "",
            "Paraphrases are candidate-only and excluded from every statistic because no frozen paraphrase catalog existed.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    matrix = json.loads(args.input.read_text())
    fixtures = json.loads(args.fixture_manifest.read_text())
    fixtures["_path"] = str(args.fixture_manifest)
    if matrix.get("status") != "DIAGNOSTIC_ANALYZED" or matrix.get("record_count") != 120:
        raise ValueError("Expected a completed 40-pair × 3-seed diagnostic matrix")
    if set(matrix.get("arm_specs", {})) != set(ARMS):
        raise ValueError("Six-arm matrix is incomplete")
    records = flatten_records(matrix)
    pairs = aggregate_pairs(records)
    if len(records) != 720 or len(pairs) != 240:
        raise ValueError("Expected 720 arm-seed records and 240 pair-arm aggregates")
    stats = statistics(pairs, matrix, args.bootstrap_repeats, args.bootstrap_seed)
    stats.update(
        {
            "source": str(args.input.resolve()),
            "source_sha256": file_sha256(args.input),
            "fixture_manifest_sha256": file_sha256(args.fixture_manifest),
            "record_count": len(records),
            "pair_arm_count": len(pairs),
            "bootstrap_repeats": args.bootstrap_repeats,
            "bootstrap_seed": args.bootstrap_seed,
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(records, args.output_dir / "per_record_metrics.parquet")
    write_parquet(pairs, args.output_dir / "per_pair_metrics.parquet")
    (args.output_dir / "statistics.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    write_report(stats, args.output_dir / "RESPONSE_6ARM_SUMMARY.md")
    write_paraphrase_candidates(fixtures, args.output_dir)
    make_plots(pairs, stats, args.output_dir / "plots")
    files = sorted(
        path for path in args.output_dir.rglob("*") if path.is_file() and path.name != "sha256_manifest.txt"
    )
    (args.output_dir / "sha256_manifest.txt").write_text(
        "".join(f"{file_sha256(path)}  {path.relative_to(args.output_dir)}\n" for path in files)
    )
    print(json.dumps({"record_count": len(records), "pair_arm_count": len(pairs)}, indent=2))


if __name__ == "__main__":
    main()
