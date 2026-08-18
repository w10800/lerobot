#!/usr/bin/env python
"""Compare the refactored cached velocity API with the legacy denoise body."""

import argparse
import json
from pathlib import Path

import torch

from lerobot.policies.common.vla_utils import make_att_2d_masks
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE


def build_prefix_cache(policy: SmolVLAPolicy, batch: dict):
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    model = policy.model
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images,
        img_masks,
        batch[OBS_LANGUAGE_TOKENS],
        batch[OBS_LANGUAGE_ATTENTION_MASK],
        state=state,
    )
    attention = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    _, cache = model.vlm_with_expert.forward(
        attention_mask=attention,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=model.config.use_cache,
    )
    return prefix_pad_masks, cache


def legacy_denoise_body(model, prefix_pad_masks, past_key_values, x_t, timestep):
    suffix_embs, suffix_pad_masks, suffix_att_masks = model.embed_suffix(x_t, timestep)
    suffix_len = suffix_pad_masks.shape[1]
    batch_size = prefix_pad_masks.shape[0]
    prefix_len = prefix_pad_masks.shape[1]
    prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
    suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
    full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
    prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
    position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
    outputs_embeds, _ = model.vlm_with_expert.forward(
        attention_mask=full_att_2d_masks,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=[None, suffix_embs],
        use_cache=model.config.use_cache,
    )
    if past_key_values is not None:
        past_key_values.crop(prefix_len)
    suffix_out = outputs_embeds[1][:, -model.config.chunk_size :].to(dtype=torch.float32)
    return model.action_out_proj(suffix_out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--batch-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--atol", type=float, default=0.0)
    args = parser.parse_args()

    config = SmolVLAConfig.from_pretrained(args.checkpoint, revision=args.revision)
    config.device = args.device
    config.use_target_time_embedding = False
    config.training_objective = "flow_matching"
    if Path(args.checkpoint).is_dir():
        config.load_vlm_weights = False
    policy = SmolVLAPolicy.from_pretrained(
        args.checkpoint, config=config, revision=args.revision, strict=False
    )
    batch = torch.load(args.batch_file, map_location="cpu", weights_only=True)
    if not isinstance(batch, dict) or OBS_STATE not in batch:
        raise ValueError("Expected a processed SmolVLA tensor batch")
    batch = {
        key: value.to(args.device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    generator = torch.Generator(device=args.device).manual_seed(args.noise_seed)
    x_t = torch.randn(
        (batch[OBS_STATE].shape[0], config.chunk_size, config.max_action_dim),
        generator=generator,
        device=args.device,
        dtype=torch.float32,
    )
    timestep = torch.full((x_t.shape[0],), 0.5, device=args.device, dtype=torch.float32)
    with torch.no_grad():
        prefix_masks, cache = build_prefix_cache(policy, batch)
        legacy = legacy_denoise_body(policy.model, prefix_masks, cache, x_t, timestep)
        prefix_masks, cache = build_prefix_cache(policy, batch)
        refactored = policy.model.predict_velocity(prefix_masks, cache, x_t, timestep)
    max_abs_error = torch.max(torch.abs(legacy - refactored)).item()
    result = {
        "schema_version": 1,
        "status": "SMOKE",
        "checkpoint": args.checkpoint,
        "revision": args.revision,
        "noise_seed": args.noise_seed,
        "atol": args.atol,
        "exact_equal": torch.equal(legacy, refactored),
        "max_abs_error": max_abs_error,
        "passed": max_abs_error <= args.atol,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not result["passed"]:
        raise RuntimeError(f"Velocity API equivalence failed: max_abs_error={max_abs_error}")


if __name__ == "__main__":
    main()
