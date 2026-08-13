# CF-Train pair and cache protocol

Status: frozen schema v1; pilot only until the readiness decision is logged.

CF-Train is constructed only from standard LIBERO training scenes. LIBERO-CF
conditions, scenes, and reported prompts are evaluation-only and must never be
used as teacher-label sources.

## Deterministic pair generation

Each JSONL record in `configs/crp_vla/cf_train_pairs.jsonl` fixes a standard
LIBERO BDDL scene, two executable conditions whose referenced entities occur in
that scene, prompts, intervention class, and a frame selector. Generation v1:

1. resolve the pinned dataset revision;
2. select the lowest-numbered episode whose exact source task matches the BDDL
   language;
3. select `floor((episode_length - 1) / 2)` as the source frame;
4. keep observation, robot state, normalization, action noise, and execution
   settings unchanged;
5. replace only the task prompt between branches;
6. reject a pair unless every condition entity exists in the source BDDL;
7. run `audit_cf_train_leakage.py` against the pinned LIBERO-CF checkout before
   producing any teacher responses.

The pilot catalog freezes the schema and audit path; it does not authorize full
cache generation or training.

## Leakage policy

Hard failures are: a LIBERO-CF source scene, an exact LIBERO-CF condition, an
exact normalized evaluation prompt/template, an OOD evaluation target object,
duplicate pair IDs, missing scene entities, or use of any dataset other than the
pinned standard LIBERO revision. Target-object overlap with non-OOD evaluation
suites is reported and must be minimized, but is not automatically fatal because
base LIBERO and LIBERO-CF intentionally share part of the object vocabulary.

## Response cache manifest

`build_response_cache_manifest.py` creates a manifest over existing diagnostic
archives without copying images. Every entry records pair identity, episode and
frame selector, factual and counterfactual prompts, noise seed/hash, teacher
checkpoint revision/hash/NFE, action archive hash, dataset revision, and the
combined normalization-config hash. Records below the pre-registered teacher
response threshold remain listed as rejected records.

