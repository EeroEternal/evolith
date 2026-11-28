import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from evolith.cli import app
from evolith.lab.zstep_sail import (
    SailPolicy,
    ZstepSailEvolution,
    ZstepSailEvolutionConfig,
    compose_sail_proposals,
    family_candidates,
    ordered_sail_targets,
)


def make_cfg(tmp_path: Path, **updates) -> ZstepSailEvolutionConfig:
    defaults = {
        "cycles": 1,
        "evaluations": 4,
        "families_per_cycle": 1,
    } | updates
    return ZstepSailEvolutionConfig(
        remote="root@10.8.141.36",
        local_zstep_repo=tmp_path,
        remote_root=Path("/data/zstep-evolith-sail"),
        model_dir=Path("/data/models/DeepSeek-V4.1-Flash-1017"),
        output=tmp_path / "results",
        **defaults,
    )


class FakeSailEvolution(ZstepSailEvolution):
    def __init__(self, config, *, confirm=True, tokens=True, faster=True, supported_share=0.45):
        super().__init__(config)
        self.confirm = confirm
        self.tokens = tokens
        self.faster = faster
        self.supported_share = supported_share
        self.micro_calls: list[tuple[str, str, str, int]] = []
        self.model_calls: list[str] = []
        self.observations: list[dict] = []

    def prepare(self) -> dict:
        return {"source": "src_hash", "controller": "ctrl_hash", "environment": "8x ZW810E"}

    def observe(self, policy: SailPolicy, cycle: int) -> dict:
        self.observations.append(policy.model_dump())
        attn_us = (
            150.0
            if policy.sparse_attn_backend == "triton_tp8"
            else 1000.0 * self.supported_share
        )
        tot_us = 300.0 + attn_us + 250.0
        return {
            "policy": policy.model_dump(),
            "applied_policy": policy.to_worker_policy_dict(),
            "tokens": [11] * 32,
            "output_throughput_tok_s": 100.0,
            "total_estimated_us": tot_us,
            "families": [
                {
                    "name": "engram_indexer_mhc",
                    "supported": False,
                    "calls_per_rank": 40,
                    "estimated_total_us": 300.0,
                    "share": round(300.0 / tot_us, 6),
                    "reason": "Stateless Engram/Indexer/mHC kernels have no parameterized policy family yet",
                },
                {
                    "name": "sparse_attn_tp8",
                    "supported": True,
                    "calls_per_rank": 40,
                    "estimated_total_us": attn_us,
                    "share": round(attn_us / tot_us, 6),
                    "reason": None,
                },
                {
                    "name": "icn_allreduce",
                    "supported": True,
                    "calls_per_rank": 88,
                    "estimated_total_us": 250.0,
                    "share": round(250.0 / tot_us, 6),
                    "reason": None,
                },
            ],
        }

    def micro(self, family: str, incumbent: SailPolicy, candidate: SailPolicy, seed: int, directory: Path) -> dict:
        self.micro_calls.append((family, incumbent.key, candidate.key, seed))
        base_us = 200.0
        if family == "sparse_attn_tp8" and candidate.sparse_attn_backend == "triton_tp8":
            cand_us = 80.0 if (self.confirm or seed == 31) else 210.0
        elif family == "icn_allreduce" and candidate.ar_oneshot_max_bytes == 262144:
            cand_us = 150.0 if (self.confirm or seed == 31) else 210.0
        else:
            cand_us = 215.0
        return {
            "family": family,
            "seed": seed,
            "rounds": self.config.rounds,
            "iterations": self.config.iterations,
            "baseline_config": incumbent.to_worker_policy_dict(),
            "config": candidate.to_worker_policy_dict(),
            "inputs_unchanged": True,
            "reference_ok": True,
            "ranks_agree": True,
            "baseline_us": [base_us] * self.config.rounds,
            "candidate_us": [cand_us] * self.config.rounds,
        }

    def model(self, policy: SailPolicy, directory: Path) -> dict:
        self.model_calls.append(policy.key)
        is_default = policy == SailPolicy()
        if is_default:
            wall = 1_000_000.0
        elif not self.faster:
            wall = 1_050_000.0
        else:
            wall = 1_000_000.0
            if policy.sparse_attn_backend == "triton_tp8":
                wall -= 220_000.0
            if policy.ar_oneshot_max_bytes == 262144:
                wall -= 100_000.0
        return {
            "applied_policy": policy.to_worker_policy_dict(),
            "tokens": [11 if (self.tokens or is_default) else 99] * self.config.decode_tokens,
            "tier_wall_us": wall,
            "output_throughput_tok_s": 1000.0 * 1e6 / wall,
            "total_throughput_tok_s": 5000.0 * 1e6 / wall,
            "mean_tpot_ms": 10.0,
            "median_ttft_ms": 40.0,
        }


def test_sail_policy_bounds_and_cli_help(tmp_path: Path) -> None:
    default_p = SailPolicy()
    assert default_p.sparse_attn_backend == "auto"
    for bad in (
        {"ar_blocks": 7},
        {"ar_threads": 64},
        {"sparse_attn_backend": "flashinfer"},
        {"moe_block_m": 128},
        {"extra_field": 1},
    ):
        with pytest.raises(ValidationError):
            SailPolicy.model_validate(bad)
    for fam in (
        "sparse_attn_tp8",
        "icn_allreduce",
        "mxfp4_moe_decode",
        "mxfp4_moe_prefill",
        "mhc_mixes",
        "attn_qkv_o_proj",
        "engram_compressor_indexer",
        "lm_head",
    ):
        cands = family_candidates(default_p, fam)
        assert cands and default_p not in cands
    # Legacy v1/v2 approved policy JSON validates cleanly with v3 default knobs
    v1_pol = SailPolicy.model_validate(
        {
            "version": 1,
            "moe_gemv_max_rows": 32,
            "moe_block_m": 16,
            "sparse_attn_backend": "triton_tp8",
        }
    )
    assert v1_pol.mhc_split_k == 1 and v1_pol.mhc_fused_max_rows == 16
    assert v1_pol.moe_prefill_backend == "triton_grouped"
    assert v1_pol.qkv_fusion_mode == "separate"
    assert v1_pol.engram_indexer_backend == "eager_py"
    assert v1_pol.lm_head_dtype == "fp32"
    assert v1_pol.ar_fuse_mhc_post == 0
    assert v1_pol.moe_router_mode == "fp32"
    assert v1_pol.mhc_backend == "legacy_2pass"
    assert CliRunner().invoke(app, ["zstep-sail-evolve", "--help"]).exit_code == 0


def test_target_ordering_follows_measured_cost_and_excludes_unsupported() -> None:
    families = [
        {"name": "engram_indexer_mhc", "supported": False, "estimated_total_us": 9999.0, "share": 0.60},
        {"name": "icn_allreduce", "supported": True, "estimated_total_us": 300.0, "share": 0.18},
        {"name": "sparse_attn_tp8", "supported": True, "estimated_total_us": 520.0, "share": 0.31},
    ]
    ordered = ordered_sail_targets(families, min_share=0.05)
    assert [f["name"] for f in ordered] == ["sparse_attn_tp8", "icn_allreduce"]


def test_fresh_seed_confirmation_full_model_gate_and_capability_gaps(tmp_path: Path) -> None:
    evo = FakeSailEvolution(make_cfg(tmp_path))
    res = evo.run()
    rep = json.loads(Path(res["report"]).read_text())
    assert res["deployment_approved"] is True
    assert res["incumbent"]["sparse_attn_backend"] == "triton_tp8"
    assert len(res["capability_gaps"]) == 1
    assert res["capability_gaps"][0]["name"] == "engram_indexer_mhc"
    # `triton_tp8` must run both seed=31 search and seed=73 confirmation
    triton_calls = [c for c in evo.micro_calls if "attn-triton_tp8" in c[2]]
    assert [c[3] for c in triton_calls] == [31, 73]
    # 5 alternating AB/BA pairs = 10 full-model evaluations
    assert len(evo.model_calls) == 10
    assert rep["model_evaluations"] == 1
    assert (tmp_path / "results" / res["run_id"] / "approved-sail-policy.json").is_file()


@pytest.mark.parametrize("kwargs", [{"confirm": False}, {"tokens": False}, {"faster": False}])
def test_rejected_candidates_never_replace_incumbent(tmp_path: Path, kwargs: dict) -> None:
    evo = FakeSailEvolution(make_cfg(tmp_path), **kwargs)
    res = evo.run()
    assert res["deployment_approved"] is False
    assert res["incumbent"] == SailPolicy().model_dump()
    if kwargs.get("confirm") is False:
        assert evo.model_calls == []
    if kwargs.get("tokens") is False:
        assert len(evo.model_calls) == 2


def test_second_cycle_compares_against_approved_incumbent_and_resumes_archive(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path, cycles=2, evaluations=8, families_per_cycle=1)
    evo = FakeSailEvolution(cfg)
    first_res = evo.run()
    assert first_res["incumbent"]["sparse_attn_backend"] == "triton_tp8"
    assert first_res["incumbent"]["ar_oneshot_max_bytes"] == 262144
    # Cycle 1 observed the newly approved triton_tp8 incumbent, not the original native policy!
    assert evo.observations[0]["sparse_attn_backend"] == "auto"
    assert evo.observations[1]["sparse_attn_backend"] == "triton_tp8"

    # Subsequent session under identical context resumes from the latest approved incumbent
    second = FakeSailEvolution(make_cfg(tmp_path))
    sec_res = second.run()
    assert second.observations[0] == first_res["incumbent"]
    assert json.loads(Path(sec_res["report"]).read_text())["parent_run"] == evo.run_id


def test_compose_sail_proposals_orders_by_predicted_savings() -> None:
    inc = SailPolicy()
    winners = [
        {
            "family": "icn_allreduce",
            "policy": inc.model_copy(update={"ar_blocks": 16, "ar_oneshot_max_bytes": 262144}).model_dump(),
            "predicted_saving_us": 40.0,
        },
        {
            "family": "sparse_attn_tp8",
            "policy": inc.model_copy(update={"sparse_attn_backend": "triton_tp8"}).model_dump(),
            "predicted_saving_us": 190.0,
        },
    ]
    props = compose_sail_proposals(inc, winners)
    assert len(props) == 2
    assert props[0].sparse_attn_backend == "triton_tp8" and props[0].ar_blocks == 16
    assert props[1].sparse_attn_backend == "triton_tp8" and props[1].ar_blocks == 8


def test_unapplied_sail_policy_fails_closed(tmp_path: Path) -> None:
    evo = ZstepSailEvolution(make_cfg(tmp_path))
    evo._idle = lambda d: None
    bad_resp = {
        "applied_policy": SailPolicy().to_worker_policy_dict(),
        "tokens": [11] * 32,
        "tier_wall_us": 500_000.0,
        "output_throughput_tok_s": 200.0,
        "total_throughput_tok_s": 800.0,
        "mean_tpot_ms": 8.0,
        "median_ttft_ms": 40.0,
    }
    evo._remote_bash = lambda *a, **kw: json.dumps(bad_resp)
    with pytest.raises(ValueError, match="Unapplied SAIL policy"):
        evo.model(SailPolicy(sparse_attn_backend="triton_tp8"), evo.directory / "m")


def test_run_ssh_retries_transient_exit_255(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess as sp

    evo = ZstepSailEvolution(make_cfg(tmp_path, control_path="/tmp/fake_sail_mux.sock"))
    monkeypatch.setattr("evolith.lab.zstep_sail.time.sleep", lambda *_a: None)
    calls = {"n": 0}

    class FakeProc:
        def __init__(self, rc: int, out: bytes) -> None:
            self.returncode = rc
            self._out = out
            self.pid = 12345

        def communicate(self, input=None, timeout=None):
            return self._out, b""

    def fake_popen(*_a, **_kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return FakeProc(255, b"")
        return FakeProc(0, b"ok\n")

    monkeypatch.setattr(sp, "Popen", fake_popen)
    out = evo._run_ssh("echo ok", tmp_path / "ssh_test")
    assert out == "ok\n"
    assert calls["n"] == 2
