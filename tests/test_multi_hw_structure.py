from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from evolith.cli import app
from evolith.hardware import get_hardware, list_hardware
from evolith.model_families import get_model_family, list_model_families
from evolith.targets import get_target, list_targets, load_profile
from evolith.targets.export_nebula import export_profile_for_nebula

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_hardware_model_and_target_registries() -> None:
    hw_names = {h.name for h in list_hardware()}
    assert {"nvidia_sm120", "thead_ppu_zw810e"}.issubset(hw_names)
    assert get_hardware("thead_ppu_zw810e").docker_backend == "sail"
    assert get_hardware("nvidia_sm120").docker_backend == "sm120"

    mf_names = {m.name for m in list_model_families()}
    assert {"deepseek_v41", "qwen35_moe"}.issubset(mf_names)
    assert get_model_family("deepseek_v41").page_block_tokens == 128

    sail_target = get_target("zstep", "thead_ppu_zw810e", "deepseek_v41")
    assert sail_target.capability_gaps(get_model_family("deepseek_v41").operator_families) == []
    assert sail_target.capability_gaps(["moe_experts", "unmodeled_custom_op"]) == [
        "unmodeled_custom_op"
    ]
    assert len(list_targets()) >= 3


def test_all_declarative_profiles_load_and_resolve() -> None:
    profiles_dir = REPO_ROOT / "profiles"
    profile_paths = sorted(profiles_dir.glob("*.json"))
    assert len(profile_paths) >= 3

    for path in profile_paths:
        profile = load_profile(path)
        assert profile.engine == "zstep"
        assert profile.docker_dockerfile.startswith("deploy/docker/Dockerfile.")
        if profile.approved_policy_path:
            assert (REPO_ROOT / profile.approved_policy_path).is_file()


def test_export_profile_for_nebula_sail_gen3(tmp_path: Path) -> None:
    profile_path = REPO_ROOT / "profiles" / "ppu_zw810e_dsv41_zstep.json"
    out_dir = tmp_path / "nebula_bundle"
    result = export_profile_for_nebula(
        profile_path,
        image="zstep:sail-zw810e-gen3",
        model_uid="deepseek-v4.1-flash",
        model_dir="/model/DeepSeek-V4.1-Flash",
        output_dir=out_dir,
        repo_root=REPO_ROOT,
    )
    assert result["status"] == "exported"

    manifest = json.loads((out_dir / "nebula-deployment.json").read_text(encoding="utf-8"))
    assert manifest["engine_type"] == "zstep"
    assert manifest["docker_image"] == "zstep:sail-zw810e-gen3"
    assert manifest["env"]["ZSTEP_BACKEND"] == "sail"
    assert manifest["env"]["ZSTEP_SAIL_MOE_PREFILL_BACKEND"] == "dequant_acblas"
    assert manifest["env"]["ZSTEP_SAIL_ENGRAM_INDEXER_BACKEND"] == "fused_hggc"
    assert "--speculative-method" in manifest["extra_args"]
    assert "dspark" in manifest["extra_args"]

    env_text = (out_dir / "policy.env").read_text(encoding="utf-8")
    assert "ZSTEP_SAIL_LM_HEAD_DTYPE=bf16" in env_text


def test_cli_profiles_and_export_nebula(tmp_path: Path) -> None:
    runner = CliRunner()
    res = runner.invoke(app, ["profiles", "--profiles-dir", str(REPO_ROOT / "profiles")])
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert len(data["profiles"]) >= 3

    out_dir = tmp_path / "cli_bundle"
    res2 = runner.invoke(
        app,
        [
            "export-nebula",
            "--profile",
            str(REPO_ROOT / "profiles" / "ppu_zw810e_dsv41_zstep.json"),
            "--image",
            "zstep:sail-zw810e-gen3",
            "--model-uid",
            "deepseek-v4.1-flash",
            "--model-dir",
            "/model/DeepSeek-V4.1-Flash",
            "--output",
            str(out_dir),
        ],
    )
    assert res2.exit_code == 0, res2.output
    out_json = json.loads(res2.output)
    assert out_json["status"] == "exported"
    assert Path(out_json["nebula_deployment"]).is_file()


def test_unknown_hardware_raises_clear_error() -> None:
    with pytest.raises(KeyError, match="Unknown hardware profile"):
        get_hardware("nonexistent_gpu")
