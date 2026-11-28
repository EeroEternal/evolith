"""Measured, budgeted PPU/SAIL (`ZW810E * 8`) policy evolution for `zstep` + DeepSeek-V4.1-Flash.

Strictly follows Evolith's governing principles:
- Observes real per-rank CUDA-event operator-family costs on the target workload before choosing
  optimization targets; never hardcodes a single operator or shape as the permanent objective.
- Explicitly records unsupported bottlenecks (`engram_indexer_mhc`) as `capability_gaps`.
- Bounded parameter/backend selection (`SailPolicy`), not arbitrary source code mutation.
- Requires both fresh-seed isolated microbenchmark confirmation (`seed=31`, then `seed=73`) AND
  `>= 5` alternating AB/BA full-model `zstep` trials against the **current approved incumbent**
  using unchanged `decide()` (`99%` one-sided bootstrap lower bound `> 1.03`, `CV <= 0.10`) plus
  exact deterministic greedy token agreement.
- Preserves all accepted and rejected trials in atomic JSON reports and append-only `RunStore`.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import tarfile
import time
from typing import Literal

from pydantic import Field

from evolith.lab.contracts import StrictModel, content_hash
from evolith.lab.measurement import decide
from evolith.lab.store import RunStore, new_run_id, write_json

SUPPORTED_SAIL_FAMILIES: frozenset[str] = frozenset(
    {
        "sparse_attn_tp8",
        "icn_allreduce",
        "mxfp4_moe_decode",
        "mxfp4_moe_prefill",
        "mhc_mixes",
    }
)

DEFAULT_SAIL_CONTAINER_IMAGE: str = (
    "docker.io/library/asllm:1.8.1-pytorch2.8.0-ubuntu24.04-sail1.7.0-"
    "cuda12.9-vllm0.11.0-sglang0.5.3-py312"
)


class SailPolicy(StrictModel):
    """Bounded, typed PPU/SAIL runtime policy consumed via `ZSTEP_SAIL_POLICY_PATH`."""

    version: Literal[1, 2] = 2
    ar_blocks: Literal[4, 8, 16, 24, 32, 64] = 8
    ar_threads: Literal[128, 256, 512] = 256
    ar_oneshot_max_bytes: Literal[65536, 131072, 262144, 524288] = 131072
    ar_pccl_min_bytes: Literal[262144, 524288, 1048576, 4194304, 67108864] = 67108864
    moe_gemv_max_rows: Literal[8, 16, 32] = 16
    moe_gemv_threads: Literal[128, 256, 512] = 256
    moe_block_m: Literal[16, 32, 64] = 32
    moe_num_warps: Literal[4, 8] = 4
    sparse_attn_backend: Literal["auto", "triton_tp8", "hggc_tp8", "flash_mla"] = "auto"
    mhc_split_k: Literal[1, 4, 8, 16] = 1
    mhc_fused_max_rows: Literal[16, 64, 256, 4096] = 16

    @property
    def key(self) -> str:
        return (
            f"ar{self.ar_blocks}x{self.ar_threads}-os{self.ar_oneshot_max_bytes // 1024}k-"
            f"pccl{self.ar_pccl_min_bytes // 1024}k-"
            f"gemv{self.moe_gemv_max_rows}@{self.moe_gemv_threads}-"
            f"bm{self.moe_block_m}w{self.moe_num_warps}-attn-{self.sparse_attn_backend}-"
            f"mhc{self.mhc_split_k}r{self.mhc_fused_max_rows}"
        )

    def to_worker_policy_dict(self) -> dict[str, int | str]:
        return {
            "ar_blocks": self.ar_blocks,
            "ar_threads": self.ar_threads,
            "ar_oneshot_max_bytes": self.ar_oneshot_max_bytes,
            "ar_pccl_min_bytes": self.ar_pccl_min_bytes,
            "moe_gemv_max_rows": self.moe_gemv_max_rows,
            "moe_gemv_threads": self.moe_gemv_threads,
            "moe_block_m": self.moe_block_m,
            "moe_num_warps": self.moe_num_warps,
            "sparse_attn_backend": self.sparse_attn_backend,
            "mhc_split_k": self.mhc_split_k,
            "mhc_fused_max_rows": self.mhc_fused_max_rows,
        }


class ZstepSailEvolutionConfig(StrictModel):
    remote: str = Field(pattern=r"^[A-Za-z0-9._-]+@[A-Za-z0-9.-]+$")
    port: int = Field(default=22, ge=1, le=65535)
    local_zstep_repo: Path
    remote_root: Path
    model_dir: Path
    output: Path
    container_image: str = Field(default=DEFAULT_SAIL_CONTAINER_IMAGE, min_length=3, max_length=512)
    control_path: Path | None = None
    initial_policy_path: Path | None = None
    parent_run: str | None = None
    gpus: str = Field(default="0,1,2,3,4,5,6,7", pattern=r"^[0-7](,[0-7])*$")
    prompt_ids: str = Field(default="671,6102,294,8760,344", pattern=r"^[0-9]+(,[0-9]+)*$")
    decode_tokens: Literal[32, 64] = 32
    bench_tier: str = Field(
        default="evolve:8:16:256:64",
        pattern=r"^[A-Za-z0-9_-]+:[0-9]+:[0-9]+:[0-9]+:[0-9]+$",
    )
    observe_bench_tier: str = Field(
        default="observe:4:4:256:16",
        pattern=r"^[A-Za-z0-9_-]+:[0-9]+:[0-9]+:[0-9]+:[0-9]+$",
    )
    cycles: int = Field(default=2, ge=1, le=8)
    evaluations: int = Field(default=12, ge=1, le=64)
    model_evaluations: int = Field(default=3, ge=1, le=12)
    families_per_cycle: int = Field(default=2, ge=1, le=6)
    rounds: int = Field(default=11, ge=5, le=101)
    iterations: int = Field(default=25, ge=1, le=1000)
    model_pairs: int = Field(default=5, ge=5, le=19)
    min_share: float = Field(default=0.03, gt=0.0, lt=1.0)
    seconds: int = Field(default=3600, ge=60, le=7200)


def ordered_sail_targets(families: list[dict], *, min_share: float = 0.03) -> list[dict]:
    """Select supported operator families ordered strictly by measured device cost."""
    eligible = [
        fam
        for fam in families
        if fam.get("supported") is True
        and fam.get("name") in SUPPORTED_SAIL_FAMILIES
        and float(fam.get("estimated_total_us", 0.0)) > 0.0
        and float(fam.get("share", 0.0)) >= min_share
    ]
    return sorted(eligible, key=lambda fam: (-float(fam["estimated_total_us"]), str(fam["name"])))


def family_candidates(incumbent: SailPolicy, family: str) -> list[SailPolicy]:
    """Bounded single-family mutations around `incumbent`; never proposes `incumbent` itself."""
    candidates: list[SailPolicy] = []
    if family == "sparse_attn_tp8":
        for bk in ("triton_tp8", "hggc_tp8", "flash_mla"):
            if bk != incumbent.sparse_attn_backend:
                candidates.append(incumbent.model_copy(update={"sparse_attn_backend": bk}))
    elif family == "icn_allreduce":
        for blks, thrs, os_b, pccl_b in (
            (64, 512, 131072, 67108864),
            (32, 256, 131072, 67108864),
            (16, 256, 131072, 67108864),
            (16, 256, 262144, 67108864),
            (4, 256, 131072, 67108864),
        ):
            cand = incumbent.model_copy(
                update={
                    "ar_blocks": blks,
                    "ar_threads": thrs,
                    "ar_oneshot_max_bytes": os_b,
                    "ar_pccl_min_bytes": pccl_b,
                }
            )
            if cand != incumbent:
                candidates.append(cand)
    elif family == "mxfp4_moe_decode":
        for thrs, gmax in (
            (128, incumbent.moe_gemv_max_rows),
            (512, incumbent.moe_gemv_max_rows),
            (256, 32),
            (256, 8),
        ):
            cand = incumbent.model_copy(
                update={"moe_gemv_threads": thrs, "moe_gemv_max_rows": gmax}
            )
            if cand != incumbent:
                candidates.append(cand)
    elif family == "mxfp4_moe_prefill":
        for bm, nw in ((16, 4), (16, 8), (64, 4), (32, 8), (64, 8)):
            cand = incumbent.model_copy(update={"moe_block_m": bm, "moe_num_warps": nw})
            if cand != incumbent:
                candidates.append(cand)
    elif family == "mhc_mixes":
        for sk, mr in ((8, 4096), (16, 4096), (4, 4096), (8, 256), (4, 64)):
            cand = incumbent.model_copy(
                update={"mhc_split_k": sk, "mhc_fused_max_rows": mr}
            )
            if cand != incumbent:
                candidates.append(cand)
    else:
        raise ValueError(f"Unsupported SAIL policy family: {family!r}")

    deduped: list[SailPolicy] = []
    seen: set[str] = {incumbent.key}
    for c in candidates:
        if c.key not in seen:
            seen.add(c.key)
            deduped.append(c)
    return deduped


def compose_sail_proposals(incumbent: SailPolicy, winners: list[dict]) -> list[SailPolicy]:
    """Compose disjoint family winners ordered by descending predicted latency savings."""
    ordered = sorted(winners, key=lambda w: -float(w["predicted_saving_us"]))
    out: list[SailPolicy] = []
    seen: set[str] = {incumbent.key}
    for count in range(len(ordered), 0, -1):
        updates: dict[str, int | str] = {}
        for w in ordered[:count]:
            fam = w["family"]
            pol = SailPolicy.model_validate(w["policy"])
            if fam == "sparse_attn_backend" or fam == "sparse_attn_tp8":
                updates["sparse_attn_backend"] = pol.sparse_attn_backend
            elif fam == "icn_allreduce":
                updates.update(
                    ar_blocks=pol.ar_blocks,
                    ar_threads=pol.ar_threads,
                    ar_oneshot_max_bytes=pol.ar_oneshot_max_bytes,
                    ar_pccl_min_bytes=pol.ar_pccl_min_bytes,
                )
            elif fam == "mxfp4_moe_decode":
                updates.update(
                    moe_gemv_threads=pol.moe_gemv_threads,
                    moe_gemv_max_rows=pol.moe_gemv_max_rows,
                )
            elif fam == "mxfp4_moe_prefill":
                updates.update(
                    moe_block_m=pol.moe_block_m,
                    moe_num_warps=pol.moe_num_warps,
                )
            elif fam == "mhc_mixes":
                updates.update(
                    mhc_split_k=pol.mhc_split_k,
                    mhc_fused_max_rows=pol.mhc_fused_max_rows,
                )
        composed = incumbent.model_copy(update=updates)
        if composed.key not in seen:
            seen.add(composed.key)
            out.append(composed)
    return out


class ZstepSailEvolution:
    """Trusted measurement-first PPU/SAIL evolution controller."""

    def __init__(self, config: ZstepSailEvolutionConfig) -> None:
        if len(config.gpus.split(",")) != 8:
            raise ValueError("ZstepSailEvolution requires all 8 PPUs (`gpus='0,1,2,3,4,5,6,7'`).")
        self.config = config
        self.started = time.monotonic()
        self.run_id = new_run_id()
        self.directory = config.output.resolve() / self.run_id
        self.store = RunStore(config.output / "runs.sqlite3")

    def _remaining(self) -> float:
        rem = self.config.seconds - (time.monotonic() - self.started)
        if rem <= 0:
            raise TimeoutError("ZstepSailEvolution wall-time budget exhausted.")
        return rem

    def _ssh_base(self) -> list[str]:
        has_askpass = bool(os.environ.get("SSH_ASKPASS"))
        cmd = [
            "ssh",
            "-p",
            str(self.config.port),
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
        ]
        if not has_askpass:
            cmd += ["-o", "BatchMode=yes"]
        if self.config.control_path:
            cmd += ["-o", f"ControlPath={self.config.control_path}"]
        return cmd

    def _run_ssh(
        self,
        remote_cmd: str,
        directory: Path,
        *,
        timeout: int = 600,
        stdin_bytes: bytes | None = None,
    ) -> str:
        directory.mkdir(parents=True, exist_ok=True)
        bounded = min(timeout, max(1, int(self._remaining())))
        env = os.environ.copy()
        if env.get("SSH_ASKPASS"):
            env.setdefault("SSH_ASKPASS_REQUIRE", "force")
            env.setdefault("DISPLAY", ":0")
        proc = subprocess.Popen(
            [*self._ssh_base(), self.config.remote, remote_cmd],
            stdin=subprocess.PIPE if stdin_bytes is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            env=env,
        )
        try:
            out_b, err_b = proc.communicate(input=stdin_bytes, timeout=bounded)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                out_b, err_b = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                out_b, err_b = proc.communicate()
            (directory / "worker.log").write_bytes(out_b + err_b)
            raise TimeoutError(f"SSH command exceeded {bounded}s; terminated.") from None
        (directory / "worker.log").write_bytes(out_b + err_b)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Remote command exited {proc.returncode}; see {directory / 'worker.log'}"
            )
        return out_b.decode("utf-8", errors="replace")

    def _remote_bash(self, script: str, directory: Path, timeout: int = 600) -> str:
        bounded = min(timeout, max(1, int(self._remaining())))
        wrapped = f"timeout --signal=TERM --kill-after=10 {bounded + 15} bash -seuo pipefail"
        return self._run_ssh(
            wrapped, directory, timeout=bounded, stdin_bytes=script.encode("utf-8")
        )

    def _sync_tar(self, files_map: list[tuple[Path, str]], remote_dest: Path, directory: Path) -> None:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            for src, arcname in files_map:
                tf.add(str(src), arcname=arcname)
        cmd = f"mkdir -p {shlex.quote(str(remote_dest))} && tar -xzf - -C {shlex.quote(str(remote_dest))}"
        self._run_ssh(cmd, directory, timeout=180, stdin_bytes=buf.getvalue())

    def _idle(self, directory: Path) -> None:
        gpu_list = " ".join(shlex.quote(g.strip()) for g in self.config.gpus.split(","))
        procs = self._remote_bash(
            f"for g in {gpu_list}; do "
            f"ppu-smi --query-compute-apps=pid --format=csv,noheader -i \"$g\"; "
            f"done | grep -v '^$' || true",
            directory,
            timeout=30,
        ).strip()
        if procs:
            raise RuntimeError(f"Active PPU processes on {self.config.gpus}; refusing contended run.")

    def _source_fingerprint(self) -> str:
        digest = hashlib.sha256()
        zroot = self.config.local_zstep_repo.resolve()
        worker_pkg = zroot / "worker" / "zstep_worker"
        for p in sorted(worker_pkg.rglob("*.py")) + sorted(worker_pkg.rglob("*.cpp")) + sorted(
            worker_pkg.rglob("*.cu")
        ):
            if "__pycache__" in p.parts:
                continue
            digest.update(p.relative_to(zroot).as_posix().encode("utf-8"))
            digest.update(p.read_bytes())
        zbin = zroot / "target" / "x86_64-unknown-linux-musl" / "release" / "zstep"
        if not zbin.is_file():
            raise ValueError(f"Missing static zstep binary at {zbin}")
        digest.update(zbin.read_bytes())
        return digest.hexdigest()

    def _container_cmd(self, inner_py_args: str, *, extra_mounts: list[Path] | None = None) -> str:
        rr = shlex.quote(str(self.config.remote_root))
        mdir = shlex.quote(str(self.config.model_dir))
        img = shlex.quote(self.config.container_image)
        mounts = f"-v {mdir}:{mdir}:ro -v {rr}:{rr}"
        for mp_path in extra_mounts or []:
            q = shlex.quote(str(mp_path))
            mounts += f" -v {q}:{q}"
        return (
            f"/usr/local/bin/nerdctl -n k8s.io run --rm --network=none --ipc=host --privileged "
            f"--ulimit memlock=-1:-1 "
            f"-e CUDA_VISIBLE_DEVICES={shlex.quote(self.config.gpus)} "
            f"-e PYTHONPATH={shlex.quote(str(self.config.remote_root / 'worker'))} "
            f"-e TORCH_EXTENSIONS_DIR={shlex.quote(str(self.config.remote_root / 'cache' / 'torch_extensions'))} "
            f"-e TRITON_CACHE_DIR={shlex.quote(str(self.config.remote_root / 'cache' / 'triton'))} "
            f"{mounts} {img} python3 {inner_py_args}"
        )

    @staticmethod
    def _extract_last_json_line(stdout: str) -> dict:
        for line in reversed(stdout.strip().splitlines()):
            s = line.strip()
            if s.startswith("{") and s.endswith("}"):
                return json.loads(s)
        raise ValueError("Remote container command produced no JSON output line.")

    def prepare(self) -> dict:
        zroot = self.config.local_zstep_repo.resolve()
        zbin = zroot / "target" / "x86_64-unknown-linux-musl" / "release" / "zstep"
        worker_dir = zroot / "worker" / "zstep_worker"
        helper = Path(__file__).with_name("zstep_sail_bench.py")
        if not zbin.is_file() or not worker_dir.is_dir():
            raise ValueError("Local zstep repository is missing static binary or worker package.")

        self._idle(self.directory / "prepare" / "idle")
        files_to_sync: list[tuple[Path, str]] = [
            (zbin, "bin/zstep"),
            (helper, f"_evolith/{self.run_id}/zstep_sail_bench.py"),
        ]
        for p in sorted(worker_dir.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                rel = p.relative_to(zroot).as_posix()
                files_to_sync.append((p, rel))
        self._sync_tar(files_to_sync, self.config.remote_root, self.directory / "prepare" / "sync")
        first_gpu = shlex.quote(self.config.gpus.split(",")[0].strip())
        env_info = self._remote_bash(
            f"chmod +x {shlex.quote(str(self.config.remote_root / 'bin' / 'zstep'))} && "
            f"mkdir -p {shlex.quote(str(self.config.remote_root / 'cache' / 'torch_extensions'))} "
            f"{shlex.quote(str(self.config.remote_root / 'cache' / 'triton'))} && "
            f"ppu-smi -L | sed -n '1,8p' && ppu-smi -q -d VERSION -i {first_gpu} | sed -n '1p'",
            self.directory / "prepare" / "environment",
            timeout=60,
        ).strip()
        src_fp = self._source_fingerprint()
        ctrl_fp = hashlib.sha256(Path(__file__).read_bytes() + helper.read_bytes()).hexdigest()
        return {
            "source": src_fp,
            "controller": ctrl_fp,
            "environment": env_info,
        }

    def _helper_remote_path(self) -> Path:
        return self.config.remote_root / "_evolith" / self.run_id / "zstep_sail_bench.py"

    def _verify_applied_policy(self, result: dict, policy: SailPolicy) -> None:
        applied = result.get("applied_policy")
        if not isinstance(applied, dict):
            raise ValueError("Missing applied SAIL policy in worker log; failing closed.")
        expected = policy.to_worker_policy_dict()
        for k, v in expected.items():
            if applied.get(k) != v:
                raise ValueError(
                    f"Unapplied SAIL policy knob {k}: expected {v!r}, worker reported {applied.get(k)!r}"
                )

    def observe(self, policy: SailPolicy, cycle: int) -> dict:
        obs_dir = self.directory / f"cycle-{cycle:02d}" / "observe"
        self._idle(obs_dir / "idle")
        remote_work = self.config.remote_root / "_evolith" / self.run_id / f"observe-{cycle:02d}"
        inner = (
            f"{shlex.quote(str(self._helper_remote_path()))} model "
            f"--zstep-bin {shlex.quote(str(self.config.remote_root / 'bin' / 'zstep'))} "
            f"--worker-root {shlex.quote(str(self.config.remote_root / 'worker'))} "
            f"--model-dir {shlex.quote(str(self.config.model_dir))} "
            f"--policy {shlex.quote(json.dumps(policy.to_worker_policy_dict()))} "
            f"--work-dir {shlex.quote(str(remote_work))} "
            f"--prompt-ids {shlex.quote(self.config.prompt_ids)} "
            f"--decode-tokens {self.config.decode_tokens} "
            f"--bench-tier {shlex.quote(self.config.observe_bench_tier)} "
            f"--warmup-prompts 2 --observe"
        )
        raw = self._remote_bash(self._container_cmd(inner), obs_dir, timeout=600)
        parsed = self._extract_last_json_line(raw)
        self._verify_applied_policy(parsed, policy)
        obs = parsed.get("observation")
        if (
            not isinstance(obs, dict)
            or not obs.get("families")
            or float(obs.get("total_estimated_us", 0.0)) <= 0.0
        ):
            raise ValueError("Invalid SAIL workload observation payload.")
        for fam in obs["families"]:
            est = float(fam.get("estimated_total_us", -1.0))
            sh = float(fam.get("share", -1.0))
            if not math.isfinite(est) or est < 0.0 or not math.isfinite(sh) or not 0.0 <= sh <= 1.0:
                raise ValueError("Non-finite or out-of-bounds operator-family cost share.")
        payload = {
            "policy": policy.model_dump(),
            "applied_policy": parsed["applied_policy"],
            "tokens": parsed["tokens"],
            "output_throughput_tok_s": parsed["output_throughput_tok_s"],
            "total_estimated_us": obs["total_estimated_us"],
            "families": obs["families"],
        }
        write_json(obs_dir / "observation.json", payload)
        return payload

    def micro(
        self,
        family: str,
        incumbent: SailPolicy,
        candidate: SailPolicy,
        seed: int,
        directory: Path,
    ) -> dict:
        self._idle(directory / "idle")
        inner = (
            f"{shlex.quote(str(self._helper_remote_path()))} micro "
            f"--family {shlex.quote(family)} "
            f"--baseline {shlex.quote(json.dumps(incumbent.to_worker_policy_dict()))} "
            f"--candidate {shlex.quote(json.dumps(candidate.to_worker_policy_dict()))} "
            f"--seed {seed} --rounds {self.config.rounds} --iterations {self.config.iterations}"
        )
        raw = self._remote_bash(self._container_cmd(inner), directory, timeout=240)
        res = self._extract_last_json_line(raw)
        samples = res.get("baseline_us", []) + res.get("candidate_us", [])
        if (
            res.get("family") != family
            or res.get("seed") != seed
            or res.get("rounds") != self.config.rounds
            or res.get("baseline_config") != incumbent.to_worker_policy_dict()
            or res.get("config") != candidate.to_worker_policy_dict()
            or len(samples) != 2 * self.config.rounds
            or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in samples)
        ):
            raise ValueError("Microbenchmark response provenance mismatch.")
        if (
            res.get("inputs_unchanged") is not True
            or res.get("reference_ok") is not True
            or res.get("ranks_agree") is not True
        ):
            raise ValueError(f"Microbenchmark {family} failed reference/immutability/rank gate.")
        return res

    def model(self, policy: SailPolicy, directory: Path) -> dict:
        self._idle(directory / "idle")
        rel = directory.relative_to(self.directory)
        remote_work = self.config.remote_root / "_evolith" / self.run_id / rel
        inner = (
            f"{shlex.quote(str(self._helper_remote_path()))} model "
            f"--zstep-bin {shlex.quote(str(self.config.remote_root / 'bin' / 'zstep'))} "
            f"--worker-root {shlex.quote(str(self.config.remote_root / 'worker'))} "
            f"--model-dir {shlex.quote(str(self.config.model_dir))} "
            f"--policy {shlex.quote(json.dumps(policy.to_worker_policy_dict()))} "
            f"--work-dir {shlex.quote(str(remote_work))} "
            f"--prompt-ids {shlex.quote(self.config.prompt_ids)} "
            f"--decode-tokens {self.config.decode_tokens} "
            f"--bench-tier {shlex.quote(self.config.bench_tier)} "
            f"--warmup-prompts 4"
        )
        raw = self._remote_bash(self._container_cmd(inner), directory, timeout=600)
        res = self._extract_last_json_line(raw)
        self._verify_applied_policy(res, policy)
        toks = res.get("tokens")
        wall_us = res.get("tier_wall_us", 0.0)
        if (
            not isinstance(toks, list)
            or len(toks) != self.config.decode_tokens
            or any(type(t) is not int or t < 0 for t in toks)
            or not isinstance(wall_us, (int, float))
            or not math.isfinite(wall_us)
            or wall_us <= 0.0
        ):
            raise ValueError("Invalid full-model zstep response.")
        return res

    def verify(self, incumbent: SailPolicy, candidate: SailPolicy, directory: Path) -> dict:
        baseline_us: list[float] = []
        candidate_us: list[float] = []
        trials: list[dict] = []
        for pair in range(self.config.model_pairs):
            order = (("baseline", incumbent), ("candidate", candidate))
            if pair % 2 == 1:
                order = tuple(reversed(order))
            results = {
                name: self.model(pol, directory / f"pair-{pair:02d}-{idx}-{name}")
                for idx, (name, pol) in enumerate(order)
            }
            base_r, cand_r = results["baseline"], results["candidate"]
            same_tokens = base_r["tokens"] == cand_r["tokens"]
            baseline_us.append(float(base_r["tier_wall_us"]))
            candidate_us.append(float(cand_r["tier_wall_us"]))
            trials.append(
                {
                    "order": [name for name, _ in order],
                    "tokens_match": same_tokens,
                    "baseline_tier_wall_us": base_r["tier_wall_us"],
                    "candidate_tier_wall_us": cand_r["tier_wall_us"],
                    "baseline_out_tok_s": base_r["output_throughput_tok_s"],
                    "candidate_out_tok_s": cand_r["output_throughput_tok_s"],
                }
            )
            if not same_tokens:
                return {"approved": False, "tokens_match": False, "trials": trials}
        timing = decide(baseline_us, candidate_us, seed=2026)
        return {
            "approved": timing.accepted,
            "tokens_match": True,
            "trials": trials,
            "measurement": timing.model_dump(),
        }

    def run(self) -> dict:
        self.directory.mkdir(parents=True, exist_ok=False)
        incumbent = SailPolicy()
        if self.config.initial_policy_path is not None:
            raw_init = json.loads(
                self.config.initial_policy_path.read_text(encoding="utf-8")
            )
            incumbent = SailPolicy.model_validate(raw_init)
        report: dict = {
            "run_id": self.run_id,
            "status": "running",
            "deployment_approved": False,
            "scope": (
                "Observed PPU/SAIL operator-family parameter & backend policy evolution on "
                "8x PPU-ZW810E; bounded selection among verified implementations, not source mutation."
            ),
            "cycles": [],
            "capability_gaps": [],
            "evaluations": 0,
            "model_evaluations": 0,
            "incumbent": incumbent.model_dump(),
            "stop_reason": None,
        }
        if self.config.parent_run:
            report["parent_run"] = self.config.parent_run
        if self.config.initial_policy_path is not None:
            report["initial_policy_path"] = str(self.config.initial_policy_path)
        write_json(self.directory / "report.json", report)
        try:
            provenance = self.prepare()
            report["provenance"] = provenance
            context = content_hash(
                {
                    "source": provenance["source"],
                    "controller": provenance["controller"],
                    "environment": provenance["environment"],
                    "model_dir": str(self.config.model_dir),
                    "gpus": self.config.gpus,
                    "prompt_ids": self.config.prompt_ids,
                    "decode_tokens": self.config.decode_tokens,
                    "bench_tier": self.config.bench_tier,
                }
            )
            report["context"] = context

            tried = {
                entry["key"]
                for entry in self.store.recent("zstep_sail_micro", limit=256)
                if entry.get("context") == context
            }
            model_tried = {
                entry["key"]
                for entry in self.store.recent("zstep_sail_policy", limit=256)
                if entry.get("context") == context
            }
            if self.config.initial_policy_path is None:
                for session in self.store.recent("zstep_sail_session", limit=256):
                    if session.get("context") == context and session.get("deployment_approved"):
                        incumbent = SailPolicy.model_validate(session["incumbent"])
                        report["parent_run"] = session["run_id"]
                        report["incumbent"] = incumbent.model_dump()
                        break

            for cycle in range(self.config.cycles):
                if report["evaluations"] >= self.config.evaluations:
                    report["stop_reason"] = "evaluation_budget"
                    break
                if report["model_evaluations"] >= self.config.model_evaluations:
                    report["stop_reason"] = "model_budget"
                    break
                self._remaining()

                observation = self.observe(incumbent, cycle)
                gaps = [
                    fam for fam in observation["families"] if fam.get("supported") is not True
                ]
                report["capability_gaps"] = gaps
                targets = ordered_sail_targets(
                    observation["families"], min_share=self.config.min_share
                )
                cycle_rec: dict = {
                    "cycle": cycle,
                    "incumbent_at_start": incumbent.model_dump(),
                    "observation": observation,
                    "capability_gaps": gaps,
                    "selected_targets": [t["name"] for t in targets[: self.config.families_per_cycle]],
                    "micro": [],
                    "proposals": [],
                }
                report["cycles"].append(cycle_rec)

                if not targets:
                    report["stop_reason"] = (
                        "capability_gap" if gaps else "not_observed_bottleneck"
                    )
                    write_json(self.directory / "report.json", report)
                    break

                winners: list[dict] = []
                fam_count = 0
                for target_fam in targets:
                    fam_name = str(target_fam["name"])
                    options: list[tuple[SailPolicy, str]] = []
                    for cand in family_candidates(incumbent, fam_name):
                        m_key = content_hash(
                            {
                                "context": context,
                                "family": fam_name,
                                "incumbent": incumbent.model_dump(),
                                "candidate": cand.model_dump(),
                            }
                        )
                        if m_key not in tried:
                            options.append((cand, m_key))
                    if not options:
                        continue
                    fam_count += 1
                    best_fam_winner: dict | None = None
                    for cand, m_key in options:
                        if report["evaluations"] >= self.config.evaluations:
                            break
                        self._remaining()
                        report["evaluations"] += 1
                        tried.add(m_key)
                        m_dir = self.directory / f"cycle-{cycle:02d}" / f"{fam_name}-{cand.key}"
                        entry: dict = {
                            "context": context,
                            "key": m_key,
                            "family": fam_name,
                            "incumbent": incumbent.model_dump(),
                            "candidate": cand.model_dump(),
                            "accepted": False,
                        }
                        try:
                            search_res = self.micro(
                                fam_name, incumbent, cand, 31, m_dir / "search"
                            )
                            search_timing = decide(
                                search_res["baseline_us"], search_res["candidate_us"], seed=2026
                            )
                            entry.update(
                                benchmark=search_res,
                                measurement=search_timing.model_dump(),
                            )
                            if search_timing.accepted:
                                conf_res = self.micro(
                                    fam_name, incumbent, cand, 73, m_dir / "confirmation"
                                )
                                conf_timing = decide(
                                    conf_res["baseline_us"], conf_res["candidate_us"], seed=2027
                                )
                                entry.update(
                                    confirmation=conf_res,
                                    confirmation_measurement=conf_timing.model_dump(),
                                    accepted=conf_timing.accepted,
                                )
                                if conf_timing.accepted and (
                                    best_fam_winner is None
                                    or conf_timing.speedup > best_fam_winner["speedup"]
                                ):
                                    best_fam_winner = {
                                        "family": fam_name,
                                        "policy": cand.model_dump(),
                                        "speedup": conf_timing.speedup,
                                        "predicted_saving_us": float(
                                            target_fam["estimated_total_us"]
                                        )
                                        * (1.0 - 1.0 / conf_timing.speedup),
                                    }
                        except Exception as exc:
                            entry["error"] = f"{type(exc).__name__}: {exc}"
                            raise
                        finally:
                            cycle_rec["micro"].append(entry)
                            write_json(m_dir / "report.json", entry)
                            self.store.record(
                                f"{self.run_id}-micro-{report['evaluations']:02d}",
                                "zstep_sail_micro",
                                entry,
                            )
                            write_json(self.directory / "report.json", report)
                    if best_fam_winner is not None:
                        winners.append(best_fam_winner)
                    if fam_count >= self.config.families_per_cycle:
                        break

                cycle_rec["winners"] = winners
                for cand_policy in compose_sail_proposals(incumbent, winners):
                    if report["model_evaluations"] >= self.config.model_evaluations:
                        break
                    p_key = content_hash(
                        {
                            "context": context,
                            "incumbent": incumbent.model_dump(),
                            "candidate": cand_policy.model_dump(),
                        }
                    )
                    if p_key in model_tried:
                        continue
                    model_tried.add(p_key)
                    report["model_evaluations"] += 1
                    p_dir = self.directory / f"policy-{report['model_evaluations']:02d}"
                    p_entry: dict = {
                        "context": context,
                        "key": p_key,
                        "incumbent": incumbent.model_dump(),
                        "candidate": cand_policy.model_dump(),
                    }
                    try:
                        p_entry.update(self.verify(incumbent, cand_policy, p_dir))
                    except Exception as exc:
                        p_entry.update(approved=False, error=f"{type(exc).__name__}: {exc}")
                        raise
                    finally:
                        cycle_rec["proposals"].append(p_entry)
                        write_json(p_dir / "report.json", p_entry)
                        self.store.record(
                            f"{self.run_id}-policy-{report['model_evaluations']:02d}",
                            "zstep_sail_policy",
                            p_entry,
                        )
                    if p_entry["approved"]:
                        incumbent = cand_policy
                        report["incumbent"] = incumbent.model_dump()
                        report["deployment_approved"] = True
                        write_json(
                            self.directory / "approved-sail-policy.json",
                            incumbent.to_worker_policy_dict(),
                        )
                        break

                write_json(self.directory / "report.json", report)
                if not cycle_rec["micro"] and not cycle_rec["proposals"]:
                    report["stop_reason"] = "no_untried_supported_candidates"
                    break

            report["status"] = "stopped"
            report["stop_reason"] = report["stop_reason"] or "cycle_budget"
        except TimeoutError as exc:
            report.update(status="stopped", stop_reason="time_budget", error=str(exc))
        except Exception as exc:
            report.update(
                status="failed",
                stop_reason="execution_failure",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        finally:
            report["elapsed_sec"] = time.monotonic() - self.started
            write_json(self.directory / "report.json", report)
            self.store.record(self.run_id, "zstep_sail_session", report)

        return {
            "status": report["status"],
            "run_id": self.run_id,
            "report": str(self.directory / "report.json"),
            "deployment_approved": report["deployment_approved"],
            "incumbent": report["incumbent"],
            "capability_gaps": report["capability_gaps"],
            "stop_reason": report["stop_reason"],
        }
