#!/usr/bin/env python
"""Audit registered CF-Train pairs against the pinned LIBERO-CF evaluation set."""

import argparse
import hashlib
import json
import re
from pathlib import Path


REQUIRED_FIELDS = {
    "schema_version",
    "pair_id",
    "dataset_repo",
    "dataset_revision",
    "source_suite",
    "source_bddl",
    "source_task",
    "frame_selector",
    "intervention_type",
    "generation_rule",
    "factual_prompt",
    "counterfactual_prompt",
    "factual_condition",
    "counterfactual_condition",
    "target_objects",
    "receptacles",
    "scene_entities",
    "review_status",
}


def normalize_text(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())


def condition_entities(condition: str) -> list[str]:
    return condition.lower().split()[1:]


def object_type(instance: str) -> str:
    return re.sub(r"_\d+(?:_.+)?$", "", instance)


def load_catalog(path: Path) -> list[dict]:
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        missing = REQUIRED_FIELDS - record.keys()
        if missing:
            raise ValueError(f"{path}:{line_number} missing fields: {sorted(missing)}")
        records.append(record)
    if not records:
        raise ValueError("CF-Train catalog is empty")
    return records


def load_eval_inventory(condition_dir: Path) -> dict:
    conditions: set[str] = set()
    prompts: set[str] = set()
    scenes: set[str] = set()
    targets: set[str] = set()
    ood_targets: set[str] = set()
    files = sorted(condition_dir.glob("libero_cf*.json"))
    if not files:
        raise FileNotFoundError(f"No LIBERO-CF condition files under {condition_dir}")
    for path in files:
        payload = json.loads(path.read_text())
        for scene, scene_conditions in payload.items():
            scenes.add(scene)
            stem = Path(scene).stem
            prompt = re.sub(r"^\d+-", "", stem).replace("_", " ")
            prompts.add(normalize_text(prompt))
            for condition in scene_conditions:
                normalized_condition = normalize_text(condition)
                conditions.add(normalized_condition)
                entities = condition_entities(condition)
                if entities:
                    target = object_type(entities[0])
                    targets.add(target)
                    if path.name == "libero_cf_ood.json":
                        ood_targets.add(target)
    return {
        "condition_files": [str(path) for path in files],
        "conditions": conditions,
        "prompts": prompts,
        "scenes": scenes,
        "targets": targets,
        "ood_targets": ood_targets,
    }


def audit(catalog: list[dict], inventory: dict, bddl_root: Path) -> dict:
    hard_failures = []
    warnings = []
    seen_ids = set()
    pair_reports = []
    for record in catalog:
        pair_id = record["pair_id"]
        pair_failures = []
        pair_warnings = []
        if pair_id in seen_ids:
            pair_failures.append("duplicate_pair_id")
        seen_ids.add(pair_id)
        if record["dataset_repo"] != "lerobot/libero":
            pair_failures.append("nonstandard_training_dataset")
        if record["source_suite"].startswith("libero_cf"):
            pair_failures.append("evaluation_suite_used_for_training")
        if record["source_bddl"] in inventory["scenes"]:
            pair_failures.append("evaluation_scene_overlap")
        for key in ("factual_condition", "counterfactual_condition"):
            if normalize_text(record[key]) in inventory["conditions"]:
                pair_failures.append(f"exact_eval_condition:{key}")
        for key in ("factual_prompt", "counterfactual_prompt"):
            if normalize_text(record[key]) in inventory["prompts"]:
                pair_failures.append(f"exact_eval_prompt:{key}")
        target_overlap = sorted(set(record["target_objects"]) & inventory["targets"])
        ood_overlap = sorted(set(record["target_objects"]) & inventory["ood_targets"])
        if ood_overlap:
            pair_failures.append(f"ood_target_overlap:{','.join(ood_overlap)}")
        if target_overlap:
            pair_warnings.append(f"shared_base_target_objects:{','.join(target_overlap)}")

        bddl_path = bddl_root / record["source_bddl"]
        if not bddl_path.is_file():
            pair_failures.append("missing_source_bddl")
        else:
            bddl_text = bddl_path.read_text().lower()
            absent = sorted(entity for entity in record["scene_entities"] if entity.lower() not in bddl_text)
            if absent:
                pair_failures.append(f"entities_absent_from_scene:{','.join(absent)}")
            for key in ("factual_condition", "counterfactual_condition"):
                absent_condition_entities = sorted(
                    entity for entity in condition_entities(record[key]) if entity not in bddl_text
                )
                if absent_condition_entities:
                    pair_failures.append(
                        f"condition_entities_absent:{key}:{','.join(absent_condition_entities)}"
                    )
        hard_failures.extend(f"{pair_id}:{item}" for item in pair_failures)
        warnings.extend(f"{pair_id}:{item}" for item in pair_warnings)
        pair_reports.append(
            {
                "pair_id": pair_id,
                "hard_failures": pair_failures,
                "warnings": pair_warnings,
                "target_object_overlap": target_overlap,
            }
        )
    return {
        "schema_version": 1,
        "status": "PASS" if not hard_failures else "FAIL",
        "pair_count": len(catalog),
        "hard_failures": hard_failures,
        "warnings": warnings,
        "pairs": pair_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--libero-cf-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    condition_dir = args.libero_cf_root / "libero" / "libero" / "conditions"
    bddl_root = args.libero_cf_root / "libero" / "libero" / "bddl_files"
    catalog = load_catalog(args.catalog)
    report = audit(catalog, load_eval_inventory(condition_dir), bddl_root)
    report["catalog_sha256"] = hashlib.sha256(args.catalog.read_bytes()).hexdigest()
    report["catalog"] = str(args.catalog.resolve())
    report["libero_cf_root"] = str(args.libero_cf_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
