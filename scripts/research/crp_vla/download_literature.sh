#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
paper_dir="$repo_root/literature/papers"
manifest="$repo_root/literature/manifest.tsv"
mkdir -p "$paper_dir"

printf 'priority\tarxiv_id\tcitation_key\ttitle\tsource_url\tlocal_path\tsha256\tbytes\tstatus\n' > "$manifest"

download_paper() {
    priority="$1"
    arxiv_id="$2"
    citation_key="$3"
    title="$4"
    filename="${priority}_${arxiv_id}_${citation_key}.pdf"
    destination="$paper_dir/$filename"
    temporary="$destination.part"
    source_url="https://arxiv.org/pdf/${arxiv_id}"

    if [ ! -s "$destination" ]; then
        curl --fail --location --retry 3 --retry-delay 2 "$source_url" --output "$temporary"
        mv "$temporary" "$destination"
    fi

    if ! file "$destination" | grep -q 'PDF document'; then
        printf 'Downloaded file is not a PDF: %s\n' "$destination" >&2
        exit 1
    fi

    sha256="$(shasum -a 256 "$destination" | awk '{print $1}')"
    bytes="$(wc -c < "$destination" | tr -d ' ')"
    relative_path="literature/papers/$filename"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tDOWNLOADED\n' \
        "$priority" "$arxiv_id" "$citation_key" "$title" "$source_url" \
        "$relative_path" "$sha256" "$bytes" >> "$manifest"
}

download_paper P1 2506.01844 smolvla 'SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics'
download_paper P1 2604.05656 snapflow 'SnapFlow: One-Step Action Generation for Flow-Matching VLAs via Progressive Self-Distillation'
download_paper P1 2602.17659 libero_cf 'When Vision Overrides Language: Evaluating and Mitigating Counterfactual Failures in VLAs'
download_paper P1 2410.24164 pi0 'pi0: A Vision-Language-Action Flow Model for General Robot Control'
download_paper P1 2210.02747 flow_matching 'Flow Matching for Generative Modeling'
download_paper P1 2303.01469 consistency_models 'Consistency Models'
download_paper P2 2608.04396 cofactvla 'CofactVLA: Deconfounding Vision-Language-Action Models via Counterfactual Intervention'
download_paper P2 2608.04510 guard 'GUARD: Grounding Uncertainty and Ablation-Based Risk Detection for Diffusion-Based VLAs'
download_paper P2 2605.30117 vla_trace 'VLA-Trace: Diagnosing Vision-Language-Action Models through Representation and Behavior Tracing'
download_paper P2 2603.28301 libero_para 'LIBERO-Para: A Diagnostic Benchmark and Metrics for Paraphrase Robustness in VLA Models'
download_paper P2 2603.12480 one_step_flow_policy 'One-Step Flow Policy: Self-Distillation for Fast Visuomotor Policies'
download_paper P3 1803.00443 jacobian_matching 'Knowledge Transfer with Jacobian Matching'
download_paper P3 1904.05068 relational_kd 'Relational Knowledge Distillation'

printf 'Literature manifest written to %s\n' "$manifest"
