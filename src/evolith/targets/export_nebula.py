from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evolith.lab.zstep_sail import SailPolicy
from evolith.targets.registry import EvolutionProfile, load_profile


def export_profile_for_nebula(
    profile_path: Path,
    *,
    image: str,
    model_uid: str,
    model_dir: str,
    output_dir: Path,
    policy_override_path: Path | None = None,
    world_size: int | None = None,
    replicas: int = 1,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Export an approved Evolith profile to a Nebula-ready deployment bundle."""
    profile: EvolutionProfile = load_profile(profile_path)
    root = repo_root or profile_path.resolve().parent.parent

    env_map: dict[str, str] = {
        "ZSTEP_BACKEND": "sail" if profile.docker_backend == "sail" else "cuda",
    }

    policy_file: Path | None = policy_override_path
    if policy_file is None and profile.approved_policy_path:
        candidate = Path(profile.approved_policy_path)
        policy_file = candidate if candidate.is_absolute() else (root / candidate)

    if policy_file is not None and policy_file.is_file():
        raw_policy = json.loads(policy_file.read_text(encoding="utf-8"))
        if not isinstance(raw_policy, dict):
            raise ValueError(f"Policy file {policy_file} must contain a JSON object")
        if profile.docker_backend == "sail":
            policy = SailPolicy.model_validate(raw_policy)
            for key, val in policy.to_worker_policy_dict().items():
                env_map[f"ZSTEP_SAIL_{key.upper()}"] = str(val)
        else:
            for key, val in raw_policy.items():
                env_key = key if key.startswith("ZSTEP_") else f"ZSTEP_{key.upper()}"
                env_map[env_key] = str(val)

    output_dir.mkdir(parents=True, exist_ok=True)
    policy_env_path = output_dir / "policy.env"
    policy_env_path.write_text(
        "\n".join(f"{k}={v}" for k, v in sorted(env_map.items())) + "\n",
        encoding="utf-8",
    )

    tp = world_size if world_size is not None else profile.hardware.default_world_size
    extra_args = [
        "--world-size",
        str(tp),
        "--page-block-tokens",
        str(profile.model_family.page_block_tokens),
        "--enable-prefix-caching",
    ]
    if profile.nebula_speculative_method != "none":
        extra_args.extend(
            [
                "--speculative-method",
                profile.nebula_speculative_method,
                "--num-speculative-tokens",
                str(profile.nebula_num_spec_tokens),
            ]
        )

    deployment = {
        "profile": profile.name,
        "model_uid": model_uid,
        "model_name": model_dir,
        "engine_type": profile.nebula_engine_type,
        "hardware": profile.hardware.name,
        "model_family": profile.model_family.name,
        "docker_image": image,
        "dockerfile": profile.docker_dockerfile,
        "replicas": replicas,
        "extra_args": extra_args,
        "env": env_map,
    }
    manifest_path = output_dir / "nebula-deployment.json"
    manifest_path.write_text(json.dumps(deployment, indent=2) + "\n", encoding="utf-8")

    return {
        "status": "exported",
        "profile": profile.name,
        "policy_env": str(policy_env_path),
        "nebula_deployment": str(manifest_path),
        "docker_image": image,
    }
