#!/usr/bin/env python
"""Generate a deterministic balanced pilot catalog from standard LIBERO scenes."""

import argparse
import json
import re
from pathlib import Path

from audit_cf_train_leakage import load_eval_inventory, normalize_text, object_type

DATASET_REVISION = "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"
OBJECT_NAMES = {
    "akita_black_bowl": "black bowl",
    "black_book": "book",
    "chefmate_8_frypan": "frying pan",
    "porcelain_mug": "white mug",
    "white_yellow_mug": "yellow and white mug",
}
DESTINATIONS = (
    ("On", "plate_1", "on the plate"),
    ("On", "flat_stove_1_cook_region", "on the stove"),
    ("On", "wooden_cabinet_1_top_region", "on top of the cabinet"),
    ("On", "wine_rack_1_top_region", "on the rack"),
    ("In", "akita_black_bowl_1", "in the bowl"),
    ("On", "main_table_stove_front_region", "in front of the stove"),
    ("On", "main_table_cabinet_region", "by the cabinet"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero-cf-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--object-pairs", type=int, default=20)
    parser.add_argument("--relation-pairs", type=int, default=20)
    return parser.parse_args()


def block(text: str, name: str) -> str:
    match = re.search(rf"\(:{name}\b(.*?)(?=\n\s*\(:|\n\s*\)\s*$)", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Missing :{name} block")
    return match.group(1)


def language(text: str) -> str:
    match = re.search(r"\(:language\s+(.+?)\)", text, flags=re.DOTALL)
    if not match:
        raise ValueError("Missing :language")
    return " ".join(match.group(1).split()).lower()


def objects(text: str) -> dict[str, str]:
    parsed = {}
    for names, kind in re.findall(r"([a-zA-Z0-9_\s]+?)\s+-\s+([a-zA-Z0-9_]+)", block(text, "objects")):
        for name in names.split():
            parsed[name] = kind
    return parsed


def goals(text: str) -> list[tuple[str, str, str]]:
    return re.findall(r"\((On|In)\s+([^\s()]+)\s+([^\s()]+)\)", block(text, "goal"), re.I)


def display_object(instance: str, object_types: dict[str, str]) -> str:
    kind = object_types.get(instance, object_type(instance))
    return OBJECT_NAMES.get(kind, kind.replace("_", " "))


def condition(predicate: str, target: str, destination: str) -> str:
    return f"{predicate.lower()} {target.lower()} {destination.lower()}"


def destination_type(destination: str) -> str:
    if destination.endswith("_cook_region"):
        return "flat_stove"
    if destination.endswith("_top_region") and destination.startswith("wooden_cabinet"):
        return "wooden_cabinet"
    if destination.endswith("_top_region") and destination.startswith("wine_rack"):
        return "wine_rack"
    return object_type(destination)


def destination_phrase(destination: str) -> str:
    for _, known_destination, phrase in DESTINATIONS:
        if destination == known_destination:
            return phrase
    cleaned = destination
    for prefix in ("floor_", "living_room_table_", "kitchen_table_", "main_table_", "desk_"):
        if cleaned.startswith(prefix):
            cleaned = cleaned.removeprefix(prefix)
            break
    cleaned = re.sub(r"_region(?:_\d+)?$", "", cleaned).replace("_", " ")
    return f"at the {cleaned} scene location"


def destination_options(text: str, predicate: str, destination: str) -> list[tuple[str, str, str]]:
    options = [(predicate, destination, destination_phrase(destination))]
    options.extend(item for item in DESTINATIONS if item[1] in text)
    options.extend(
        (candidate_predicate, candidate_destination, destination_phrase(candidate_destination))
        for candidate_predicate, _, candidate_destination in re.findall(
            r"\((On|In)\s+([^\s()]+)\s+([^\s()]+)\)", block(text, "init"), re.I
        )
    )
    unique = []
    seen = set()
    for item in options:
        key = (item[0].lower(), item[1].lower())
        if key not in seen:
            unique.append(item)
            seen.add(key)
    return unique


def admissible(
    factual_condition: str,
    counterfactual_condition: str,
    factual_prompt: str,
    counterfactual_prompt: str,
    target_types: list[str],
    inventory: dict,
) -> bool:
    if normalize_text(factual_condition) in inventory["conditions"]:
        return False
    if normalize_text(counterfactual_condition) in inventory["conditions"]:
        return False
    if normalize_text(factual_prompt) in inventory["prompts"]:
        return False
    if normalize_text(counterfactual_prompt) in inventory["prompts"]:
        return False
    return not (set(target_types) & inventory["ood_targets"])


def make_record(
    *,
    index: int,
    source_suite: str,
    source_path: Path,
    source_task: str,
    intervention_type: str,
    factual_prompt: str,
    counterfactual_prompt: str,
    factual_condition: str,
    counterfactual_condition: str,
    targets: list[str],
    receptacles: list[str],
    entities: list[str],
) -> dict:
    return {
        "schema_version": 1,
        "pair_id": f"cftrain-v2-{intervention_type}-{index:03d}",
        "dataset_repo": "lerobot/libero",
        "dataset_revision": DATASET_REVISION,
        "source_suite": source_suite,
        "source_bddl": f"{source_suite}/{source_path.name}",
        "source_task": source_task,
        "frame_selector": {
            "episode_rule": "lowest_episode_for_exact_source_task",
            "frame_rule": "middle_frame",
            "seed": 0,
        },
        "intervention_type": intervention_type,
        "generation_rule": "balanced_scene_present_v2",
        "factual_prompt": factual_prompt,
        "counterfactual_prompt": counterfactual_prompt,
        "factual_condition": factual_condition,
        "counterfactual_condition": counterfactual_condition,
        "target_objects": sorted(set(targets)),
        "receptacles": sorted(set(receptacles)),
        "scene_entities": sorted(set(entities)),
        "review_status": "approved_balanced_pilot",
    }


def generate_object_pairs(bddl_root: Path, inventory: dict, count: int) -> list[dict]:
    candidates = []
    seen_signatures = set()
    for source_suite in ("libero_10", "libero_goal", "libero_object"):
        for path in sorted((bddl_root / source_suite).glob("*.bddl")):
            text = path.read_text().lower()
            object_types = objects(text)
            source_task = language(text)
            movable = sorted(object_types)
            for source_predicate, _, source_destination in goals(text):
                for predicate, destination, phrase in destination_options(
                    text, source_predicate, source_destination
                ):
                    for target in movable:
                        if target == destination or object_types[target] in {"basket", "plate"}:
                            continue
                        factual_prompt = f"put the {display_object(target, object_types)} {phrase}"
                        factual_condition = condition(predicate, target, destination)
                        for alternative in movable:
                            signature = (
                                predicate.lower(),
                                destination,
                                tuple(sorted((object_types[target], object_types[alternative]))),
                            )
                            if signature in seen_signatures:
                                continue
                            if alternative in {target, destination} or object_types[alternative] in {
                                "basket",
                                "plate",
                            }:
                                continue
                            if object_types[alternative] == object_types[target]:
                                continue
                            counterfactual_prompt = (
                                f"put the {display_object(alternative, object_types)} {phrase}"
                            )
                            counterfactual_condition = condition(predicate, alternative, destination)
                            target_types = [object_types[target], object_types[alternative]]
                            if not admissible(
                                factual_condition,
                                counterfactual_condition,
                                factual_prompt,
                                counterfactual_prompt,
                                target_types,
                                inventory,
                            ):
                                continue
                            seen_signatures.add(signature)
                            candidates.append(
                                make_record(
                                    index=len(candidates),
                                    source_suite=source_suite,
                                    source_path=path,
                                    source_task=source_task,
                                    intervention_type="object_target_swap",
                                    factual_prompt=factual_prompt,
                                    counterfactual_prompt=counterfactual_prompt,
                                    factual_condition=factual_condition,
                                    counterfactual_condition=counterfactual_condition,
                                    targets=target_types,
                                    receptacles=[destination_type(destination)],
                                    entities=[target, alternative, destination],
                                )
                            )
    if len(candidates) < count:
        raise ValueError(f"Only generated {len(candidates)} admissible object pairs, need {count}")
    return candidates[:count]


def generate_relation_pairs(bddl_root: Path, inventory: dict, count: int) -> list[dict]:
    candidates = []
    seen_signatures = set()
    for path in sorted((bddl_root / "libero_goal").glob("*.bddl")):
        text = path.read_text().lower()
        object_types = objects(text)
        source_task = language(text)
        destinations = [item for item in DESTINATIONS if item[1] in text]
        for target in sorted(object_types):
            if object_types[target] == "basket":
                continue
            target_name = display_object(target, object_types)
            for factual_index, (factual_predicate, factual_destination, factual_phrase) in enumerate(
                destinations
            ):
                if factual_destination == target:
                    continue
                for counterfactual_predicate, counterfactual_destination, counterfactual_phrase in (
                    destinations[factual_index + 1 :]
                ):
                    if counterfactual_destination == target:
                        continue
                    signature = (
                        object_types[target],
                        factual_predicate.lower(),
                        factual_destination,
                        counterfactual_predicate.lower(),
                        counterfactual_destination,
                    )
                    if signature in seen_signatures:
                        continue
                    factual_condition = condition(factual_predicate, target, factual_destination)
                    factual_prompt = f"put the {target_name} {factual_phrase}"
                    counterfactual_condition = condition(
                        counterfactual_predicate,
                        target,
                        counterfactual_destination,
                    )
                    counterfactual_prompt = f"put the {target_name} {counterfactual_phrase}"
                    target_types = [object_types[target]]
                    if not admissible(
                        factual_condition,
                        counterfactual_condition,
                        factual_prompt,
                        counterfactual_prompt,
                        target_types,
                        inventory,
                    ):
                        continue
                    seen_signatures.add(signature)
                    candidates.append(
                        make_record(
                            index=len(candidates),
                            source_suite="libero_goal",
                            source_path=path,
                            source_task=source_task,
                            intervention_type="relation_location_swap",
                            factual_prompt=factual_prompt,
                            counterfactual_prompt=counterfactual_prompt,
                            factual_condition=factual_condition,
                            counterfactual_condition=counterfactual_condition,
                            targets=target_types,
                            receptacles=[
                                destination_type(factual_destination),
                                destination_type(counterfactual_destination),
                            ],
                            entities=[target, factual_destination, counterfactual_destination],
                        )
                    )
    if len(candidates) < count:
        raise ValueError(f"Only generated {len(candidates)} admissible relation pairs, need {count}")
    selected = candidates[:count]
    for index, record in enumerate(selected, start=20):
        record["pair_id"] = f"cftrain-v2-relation_location_swap-{index:03d}"
    return selected


def main() -> None:
    args = parse_args()
    bddl_root = args.libero_cf_root / "libero" / "libero" / "bddl_files"
    condition_dir = args.libero_cf_root / "libero" / "libero" / "conditions"
    inventory = load_eval_inventory(condition_dir)
    records = [
        *generate_object_pairs(bddl_root, inventory, args.object_pairs),
        *generate_relation_pairs(bddl_root, inventory, args.relation_pairs),
    ]
    if len({(record["factual_prompt"], record["counterfactual_prompt"]) for record in records}) != len(
        records
    ):
        raise ValueError("Generated catalog contains duplicate semantic prompt pairs")
    if len(
        {(record["factual_condition"], record["counterfactual_condition"]) for record in records}
    ) != len(records):
        raise ValueError("Generated catalog contains duplicate condition pairs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))
    print(
        json.dumps(
            {
                "status": "GENERATED",
                "output": str(args.output.resolve()),
                "pair_count": len(records),
                "object_pairs": args.object_pairs,
                "relation_pairs": args.relation_pairs,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
