"""HardwareProfile & BenchmarkProfile abstraction for multi-GPU Evolith × zstep.

Decouples Evolith's RSI loop, formal resource bounds, reflection cards, and remote
runner from hardcoded `8× RTX 6000D (SM120, TP=8)` assumptions so campaigns can
target SM120 (`PP=2×TP=4` / `TP=8` / `RTX 5090`), SM90 (`H100/H200`), or SM100 (`B200`)
with isolated lineage state and auto-probed topology metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from evolith.core.topology import PrecisionTrack


class BenchmarkProfile(BaseModel):
    profile_id: str
    description: str
    screen_tiers: str
    verify_tiers: str
    regime_phase: Literal["decode", "prefill", "mixed", "speculative"] = "mixed"
    precision_track: PrecisionTrack = PrecisionTrack.LOSSLESS
    num_kv_pages: int = Field(default=2560, ge=64)
    max_pages_per_seq: int = Field(default=256, ge=16)
    max_step_tokens: int = Field(default=4096, ge=192)
    env_overrides: dict[str, str] = Field(default_factory=dict)


BUILTIN_BENCHMARK_PROFILES: dict[str, BenchmarkProfile] = {
    "short-throughput": BenchmarkProfile(
        profile_id="short-throughput",
        description="Fast synthetic 512/1024-token multi-concurrency throughput & latency ladder",
        screen_tiers="in512_c1:1:8:512:64;in512_c32:32:64:512:64",
        verify_tiers=(
            "in512_c16:16:48:512:256;in512_c128:128:256:512:256;"
            "in1024_c32:32:200:1024:256;in1024_c128:128:400:1024:256"
        ),
        regime_phase="mixed",
        num_kv_pages=2560,
        max_pages_per_seq=256,
    ),
    "longctx-tp8": BenchmarkProfile(
        profile_id="longctx-tp8",
        description="TP=8 8K–32K long-context prefill & decode ladder within single-stage KV budget",
        screen_tiers="ctx8k_c2:2:4:8192:64;ctx16k_c8:8:16:16384:64",
        verify_tiers=(
            "ctx8k_c16:16:32:8192:128;ctx16k_c12:12:24:16384:128;"
            "ctx32k_c4:4:8:32768:128;ctx32k_c8:8:16:32768:128"
        ),
        regime_phase="prefill",
        num_kv_pages=2560,
        max_pages_per_seq=264,
        env_overrides={
            "ZSTEP_MAX_MODEL_LEN": "33792",
        },
    ),
    "longctx-agent": BenchmarkProfile(
        profile_id="longctx-agent",
        description="Long-context Coding-Agent 8K–32K chunked prefill, APC & speculative decode tiers",
        screen_tiers="ctx8k_c1:1:4:8192:64;ctx16k_c10:10:20:16384:64",
        verify_tiers=(
            "ctx8k_c16:16:32:8192:128;ctx16k_c20:20:40:16384:128;"
            "ctx32k_c10:10:20:32768:128;ctx32k_c20:20:40:32768:128"
        ),
        regime_phase="prefill",
        num_kv_pages=6144,
        max_pages_per_seq=264,
        env_overrides={
            "ZSTEP_MAX_MODEL_LEN": "33792",
        },
    ),
    "spec-dspark": BenchmarkProfile(
        profile_id="spec-dspark",
        description="Low-to-medium concurrency speculative decoding ladder (DSpark draft+verify)",
        screen_tiers="spec_c1:1:8:512:128;spec_c8:8:24:512:128",
        verify_tiers=(
            "spec_c2:2:12:512:256;spec_c4:4:16:512:256;spec_c16:16:48:1024:256"
        ),
        regime_phase="speculative",
        num_kv_pages=2560,
        max_pages_per_seq=256,
        env_overrides={
            "ZSTEP_SPEC_METHOD": "dspark",
            "ZSTEP_NUM_SPEC_TOKENS": "5",
        },
    ),
    "cold-prefill": BenchmarkProfile(
        profile_id="cold-prefill",
        description="Single-sequence 8K–32K cold-start TTFT & pure-prefill throughput ladder",
        screen_tiers="prefill8k_c1:1:4:8192:16;prefill16k_c1:1:4:16384:16",
        verify_tiers=(
            "prefill8k_v:1:6:8192:32;prefill16k_v:1:6:16384:32;prefill32k_v:1:4:32768:32"
        ),
        regime_phase="prefill",
        num_kv_pages=2560,
        max_pages_per_seq=264,
        max_step_tokens=8192,
        env_overrides={
            "ZSTEP_MAX_MODEL_LEN": "33792",
        },
    ),
    "nvfp4-dp2tp4": BenchmarkProfile(
        profile_id="nvfp4-dp2tp4",
        description="Isolated CALIBRATED_LOSSY single-island TP=4 / dual-island DP=2xTP=4 NVFP4 ladder",
        screen_tiers="prefill8k_c1:1:4:8192:16;in512_c16_s:16:32:512:64",
        verify_tiers=(
            "prefill8k_v:1:6:8192:32;prefill16k_v:1:6:16384:32;"
            "in512_c16_v:16:48:512:128;in512_c32_v:32:64:512:128"
        ),
        regime_phase="mixed",
        precision_track=PrecisionTrack.CALIBRATED_LOSSY,
        num_kv_pages=768,
        max_pages_per_seq=144,
        max_step_tokens=8192,
        env_overrides={
            "ZSTEP_MAX_MODEL_LEN": "18432",
            "ZSTEP_ENGRAM_GPU_LAYERS": "0",
        },
    ),
}


class HardwareProfile(BaseModel):
    profile_id: str = "sm120-8x6000d-tp8"
    gpu_name: str = "NVIDIA RTX 6000D"
    gpu_arch: str = "sm120"
    compute_capability: str = "12.0"
    world_size: int = Field(default=8, ge=1)
    dp_size: int = Field(default=1, ge=1)
    tp_size: int = Field(default=8, ge=1)
    pp_size: int = Field(default=1, ge=1)
    numa_nodes: int = Field(default=2, ge=1)
    vram_per_gpu_mib: int = Field(default=85651, ge=1024)
    interconnect: Literal["pcie_pxb_dual_numa", "pcie_pxb_single_numa", "nvswitch", "pcie_sys"] = (
        "pcie_pxb_dual_numa"
    )
    kernel_backend_dir: str = "worker/zstep_worker/kernels_sm120"
    overlay_script: str = "scripts/prepare_sm120_overlay.sh"
    decode_pipeline_1f1b: bool = False
    max_single_pxb_bytes: int = 180 * 1024 * 1024
    max_pp1_pxb_bytes: int = 256 * 1024 * 1024
    max_pp2_pxb_bytes: int = 384 * 1024 * 1024


BUILTIN_HARDWARE_PROFILES: dict[str, HardwareProfile] = {
    "sm120-8x6000d-tp8": HardwareProfile(
        profile_id="sm120-8x6000d-tp8",
        gpu_name="NVIDIA RTX 6000D",
        gpu_arch="sm120",
        compute_capability="12.0",
        world_size=8,
        tp_size=8,
        pp_size=1,
        numa_nodes=2,
        vram_per_gpu_mib=85651,
        interconnect="pcie_pxb_dual_numa",
    ),
    "sm120-8x6000d-pp2tp4": HardwareProfile(
        profile_id="sm120-8x6000d-pp2tp4",
        gpu_name="NVIDIA RTX 6000D",
        gpu_arch="sm120",
        compute_capability="12.0",
        world_size=8,
        tp_size=4,
        pp_size=2,
        numa_nodes=2,
        vram_per_gpu_mib=85651,
        interconnect="pcie_pxb_dual_numa",
        decode_pipeline_1f1b=True,
    ),
    "sm120-8x6000d-dp2tp4": HardwareProfile(
        profile_id="sm120-8x6000d-dp2tp4",
        gpu_name="NVIDIA RTX 6000D",
        gpu_arch="sm120",
        compute_capability="12.0",
        world_size=8,
        dp_size=2,
        tp_size=4,
        pp_size=1,
        numa_nodes=2,
        vram_per_gpu_mib=85651,
        interconnect="pcie_pxb_single_numa",
    ),
    "sm120-8x5090-tp8": HardwareProfile(
        profile_id="sm120-8x5090-tp8",
        gpu_name="NVIDIA GeForce RTX 5090",
        gpu_arch="sm120",
        compute_capability="12.0",
        world_size=8,
        tp_size=8,
        pp_size=1,
        numa_nodes=2,
        vram_per_gpu_mib=32607,
        interconnect="pcie_pxb_dual_numa",
        max_single_pxb_bytes=128 * 1024 * 1024,
        max_pp1_pxb_bytes=160 * 1024 * 1024,
        max_pp2_pxb_bytes=256 * 1024 * 1024,
    ),
    "sm90-8xh100-tp8": HardwareProfile(
        profile_id="sm90-8xh100-tp8",
        gpu_name="NVIDIA H100 80GB HBM3",
        gpu_arch="sm90",
        compute_capability="9.0",
        world_size=8,
        tp_size=8,
        pp_size=1,
        numa_nodes=2,
        vram_per_gpu_mib=81559,
        interconnect="nvswitch",
        kernel_backend_dir="worker/zstep_worker/kernels_sm90",
        overlay_script="",
        max_single_pxb_bytes=256 * 1024 * 1024,
        max_pp1_pxb_bytes=384 * 1024 * 1024,
        max_pp2_pxb_bytes=512 * 1024 * 1024,
    ),
    "sm100-8xb200-tp8": HardwareProfile(
        profile_id="sm100-8xb200-tp8",
        gpu_name="NVIDIA B200",
        gpu_arch="sm100",
        compute_capability="10.0",
        world_size=8,
        tp_size=8,
        pp_size=1,
        numa_nodes=2,
        vram_per_gpu_mib=183359,
        interconnect="nvswitch",
        kernel_backend_dir="worker/zstep_worker/kernels_sm100",
        overlay_script="",
        max_single_pxb_bytes=512 * 1024 * 1024,
        max_pp1_pxb_bytes=512 * 1024 * 1024,
        max_pp2_pxb_bytes=768 * 1024 * 1024,
    ),
}


def resolve_hardware_profile(profile_key: str, *, state_dir: Path | None = None) -> HardwareProfile:
    key = profile_key.strip().lower()
    if state_dir is not None and key in ("auto", "probed"):
        probed = state_dir / "hardware-profile.json"
        if probed.is_file():
            return HardwareProfile.model_validate_json(probed.read_text(encoding="utf-8"))
        if key == "auto":
            return BUILTIN_HARDWARE_PROFILES["sm120-8x6000d-tp8"]
    if key in BUILTIN_HARDWARE_PROFILES:
        return BUILTIN_HARDWARE_PROFILES[key]
    p = Path(profile_key).expanduser()
    if p.is_file():
        return HardwareProfile.model_validate_json(p.read_text(encoding="utf-8"))
    valid = ", ".join(["auto", *BUILTIN_HARDWARE_PROFILES])
    raise ValueError(f"Unknown hardware profile `{profile_key}`; valid presets: {valid}")


def resolve_benchmark_profile(profile_key: str) -> BenchmarkProfile:
    key = profile_key.strip().lower()
    if key in BUILTIN_BENCHMARK_PROFILES:
        return BUILTIN_BENCHMARK_PROFILES[key]
    valid = ", ".join(BUILTIN_BENCHMARK_PROFILES)
    raise ValueError(f"Unknown benchmark profile `{profile_key}`; valid presets: {valid}")


def infer_hardware_profile_from_probe(
    nvidia_smi_csv: str,
    topo_text: str,
    *,
    prefer_pp2_on_dual_socket_pcie: bool = True,
) -> HardwareProfile:
    """Parse `nvidia-smi --query-gpu=index,name,memory.total,compute_cap` + `topo -m` output."""
    rows = [ln.strip() for ln in nvidia_smi_csv.splitlines() if ln.strip()]
    if not rows:
        raise ValueError("Empty nvidia-smi GPU query output")

    world_size = len(rows)
    parts = [p.strip() for p in rows[0].split(",")]
    gpu_name = parts[1] if len(parts) > 1 else "Unknown GPU"
    mem_raw = parts[2].split()[0] if len(parts) > 2 else "81920"
    vram_mib = int(float(mem_raw))
    cc = parts[3] if len(parts) > 3 else "12.0"
    major_minor = cc.replace(".", "")
    if cc.startswith("12."):
        arch = "sm120"
    elif cc.startswith("10."):
        arch = "sm100"
    elif cc.startswith("9."):
        arch = "sm90"
    else:
        arch = f"sm{major_minor}"

    has_nvlink = "NV" in topo_text and any(tok.startswith("NV") and tok[2:].isdigit() for tok in topo_text.split())
    has_sys = "\tSYS" in topo_text or " SYS " in topo_text
    has_pxb = "\tPXB" in topo_text or " PXB " in topo_text

    if has_nvlink:
        interconnect: Literal["pcie_pxb_dual_numa", "pcie_pxb_single_numa", "nvswitch", "pcie_sys"] = "nvswitch"
        numa_nodes = 2 if has_sys else 1
    elif has_pxb and has_sys:
        interconnect = "pcie_pxb_dual_numa"
        numa_nodes = 2
    elif has_pxb:
        interconnect = "pcie_pxb_single_numa"
        numa_nodes = 1
    else:
        interconnect = "pcie_sys"
        numa_nodes = 2 if has_sys else 1

    use_pp2 = (
        prefer_pp2_on_dual_socket_pcie
        and world_size == 8
        and interconnect == "pcie_pxb_dual_numa"
        and vram_mib >= 60000
    )
    pp_size = 2 if use_pp2 else 1
    tp_size = max(1, world_size // pp_size)
    short_gpu = (
        "6000d"
        if "6000D" in gpu_name
        else ("5090" if "5090" in gpu_name else ("h100" if "H100" in gpu_name else ("b200" if "B200" in gpu_name else "gpu")))
    )
    topo_tag = f"pp{pp_size}tp{tp_size}" if pp_size > 1 else f"tp{tp_size}"
    profile_id = f"{arch}-{world_size}x{short_gpu}-{topo_tag}"

    single_pxb = 128 * 1024 * 1024 if vram_mib < 40000 else 180 * 1024 * 1024
    pp1_pxb = 160 * 1024 * 1024 if vram_mib < 40000 else 256 * 1024 * 1024
    pp2_pxb = 256 * 1024 * 1024 if vram_mib < 40000 else 384 * 1024 * 1024
    overlay = "scripts/prepare_sm120_overlay.sh" if arch == "sm120" else ""

    return HardwareProfile(
        profile_id=profile_id,
        gpu_name=gpu_name,
        gpu_arch=arch,
        compute_capability=cc,
        world_size=world_size,
        tp_size=tp_size,
        pp_size=pp_size,
        numa_nodes=numa_nodes,
        vram_per_gpu_mib=vram_mib,
        interconnect=interconnect,
        kernel_backend_dir=f"worker/zstep_worker/kernels_{arch}",
        overlay_script=overlay,
        decode_pipeline_1f1b=use_pp2,
        max_single_pxb_bytes=single_pxb,
        max_pp1_pxb_bytes=pp1_pxb,
        max_pp2_pxb_bytes=pp2_pxb,
    )
