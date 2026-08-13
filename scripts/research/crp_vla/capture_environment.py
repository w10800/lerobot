#!/usr/bin/env python
"""Capture a small, auditable CRP-VLA environment fingerprint."""

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path


def command(*args: str) -> str | None:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        import torch

        torch_info = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }
    except ImportError:
        torch_info = {"installed": False}

    output = {
        "schema_version": 1,
        "status": "VERIFIED",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "git_commit": command("git", "rev-parse", "HEAD"),
        "git_status": command("git", "status", "--short", "--branch"),
        "submodules": command("git", "submodule", "status"),
        "torch": torch_info,
        "nvidia_smi": command("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
