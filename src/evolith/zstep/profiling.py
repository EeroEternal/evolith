from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _safe_load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _extract_moe_breakdown(candidate_dir: Path) -> str | None:
    moe_audit = _safe_load_json(
        candidate_dir / "docs/benchmarks/step_03_moe_mxfp4_sm120.json"
    )
    if not moe_audit:
        return None
    parts: list[str] = []
    router = moe_audit.get("router", {}).get("with_correction_bias", {})
    if isinstance(router, dict) and router:
        r_tags = [
            f"T{tok}={spec.get('latency_us')}us"
            for tok in ("6", "48", "4096")
            if isinstance((spec := router.get(tok)), dict) and spec.get("latency_us") is not None
        ]
        if r_tags:
            parts.append(f"Router({', '.join(r_tags)})")

    tp8 = moe_audit.get("moe", {}).get("tp8_local_rank", {}).get("results", {})
    if isinstance(tp8, dict) and tp8:
        m_tags = [
            f"T{tok}={spec.get('latency_us')}us@{spec.get('effective_gbps')}GB/s"
            for tok in ("6", "48", "192", "4096")
            if isinstance((spec := tp8.get(tok)), dict) and spec.get("latency_us") is not None
        ]
        if m_tags:
            parts.append(f"MXFP4_MoE_TP8({', '.join(m_tags)})")
    return "; ".join(parts) if parts else None


def _extract_attention_breakdown(candidate_dir: Path) -> str | None:
    attn_audit = _safe_load_json(
        candidate_dir / "docs/benchmarks/step_04_dsv41_attention_sm120.json"
    )
    if not attn_audit or not isinstance(attn_audit.get("cases"), list):
        return None
    picks: list[str] = []
    for case in attn_audit["cases"]:
        if not isinstance(case, dict):
            continue
        mode = case.get("mode")
        kv_len = case.get("kv_len")
        batch = case.get("batch")
        lat = case.get("latency_us")
        if mode == "decode" and kv_len == 8192 and batch in (1, 8, 32) and lat is not None:
            picks.append(f"decode_B{batch}@8k={lat}us")
        elif mode == "extend" and lat is not None:
            picks.append(f"extend_rows{case.get('rows')}@8k={lat}us")
    return ", ".join(picks) if picks else None


def _extract_mhc_engram_breakdown(candidate_dir: Path) -> str | None:
    step_5a = _safe_load_json(
        candidate_dir / "docs/benchmarks/step_05a_mhc_engram_sm120.json"
    )
    if not step_5a:
        return None
    parts: list[str] = []
    mhc_rows = step_5a.get("mhc", {}).get("results", [])
    if isinstance(mhc_rows, list) and mhc_rows:
        mhc_tags = [
            f"T{row.get('tokens')}={row.get('latency_us')}us"
            for row in mhc_rows
            if isinstance(row, dict) and row.get("tokens") in (6, 48, 192, 4096)
        ]
        if mhc_tags:
            parts.append(f"mHC({', '.join(mhc_tags)})")

    engram_layers = step_5a.get("engram", {}).get("results", [])
    if isinstance(engram_layers, list) and engram_layers:
        first = engram_layers[0]
        res = first.get("results", {}) if isinstance(first, dict) else {}
        if isinstance(res, dict):
            eg_tags = [
                f"T{tok}:pinned={spec.get('pinned_latency_us')}us/vram={spec.get('vram_latency_us')}us"
                f"({spec.get('pinned_over_vram_latency')}x)"
                for tok in ("6", "48", "192", "4096")
                if isinstance((spec := res.get(tok)), dict)
                and spec.get("pinned_latency_us") is not None
            ]
            if eg_tags:
                parts.append(f"Engram_L1({', '.join(eg_tags)})")
    return "; ".join(parts) if parts else None


def _extract_allreduce_budget(candidate_dir: Path) -> str | None:
    step_5b = _safe_load_json(
        candidate_dir / "docs/benchmarks/step_05b_pcie_allreduce_8gpu.json"
    )
    if not step_5b:
        return None
    shm = step_5b.get("host_mapped_shm", {}).get("results", {})
    nccl = step_5b.get("nccl", {}).get("results", {})
    if not isinstance(shm, dict) or not shm:
        return None
    comparisons: list[str] = []
    for tok in ("6", "48", "192", "4096"):
        s_row = shm.get(tok)
        n_row = nccl.get(tok) if isinstance(nccl, dict) else None
        if isinstance(s_row, dict) and s_row.get("80_call_budget_ms") is not None:
            shm_ms = float(s_row["80_call_budget_ms"])
            if isinstance(n_row, dict) and n_row.get("80_call_budget_ms") is not None:
                nccl_ms = float(n_row["80_call_budget_ms"])
                comparisons.append(f"T{tok}:shm={shm_ms:.2f}ms vs nccl={nccl_ms:.2f}ms")
            else:
                comparisons.append(f"T{tok}:shm={shm_ms:.2f}ms")
    return ", ".join(comparisons) if comparisons else None


def build_sm120_reflection_card(
    candidate_dir: Path,
    baseline_screen_metrics: Path | None = None,
    blocked_niches: dict[str, list[str]] | None = None,
) -> str:
    """
    Extracts real SM120 / RTX 6000D hardware & per-subsystem operator profiling
    evidence from zstep's archived benchmark audits and cached baseline screen
    measurements, giving the proposer a quantitative Roofline/Bottleneck reflection
    card rather than forcing blind source guesses.
    """
    lines: list[str] = [
        "[SM120 Hardware & Bottleneck Reflection Card | 8x NVIDIA RTX 6000D (TP=8)]"
    ]

    op_audit = _safe_load_json(
        candidate_dir / "docs/benchmarks/phase37_6000d_operator_audit.json"
    )
    if op_audit:
        prov = op_audit.get("provenance", {})
        sms = prov.get("multi_processor_count", 156)
        lines.append(
            f"  - Device: {prov.get('device_name', 'RTX 6000D')} "
            f"(CC {prov.get('compute_capability', '12.0')}, {sms} SMs, "
            f"PyTorch {prov.get('torch_version', 'cu130')})"
        )
        shapes = op_audit.get("block32_fp8_and_session", {}).get("shapes", {})
        op_summaries: list[str] = []
        for op_name, spec in list(shapes.items())[:4]:
            cg = spec.get("cg_us", {})
            b1 = cg.get("1", {}).get("b12x_us")
            b32 = cg.get("32", {}).get("b12x_us")
            b128 = cg.get("128", {}).get("b12x_us")
            if b1 is not None or b128 is not None:
                op_summaries.append(
                    f"{op_name}(CG us: B1={b1}, B32={b32}, B128={b128})"
                )
        if op_summaries:
            lines.append("  - Audited Linear/FP8 Kernels: " + "; ".join(op_summaries))

    comm_audit = _safe_load_json(
        candidate_dir / "docs/benchmarks/phase37_6000d_comm_audit.json"
    )
    if comm_audit:
        summary = (
            comm_audit.get("provenance", {}).get("hybrid_thresholds")
            or comm_audit.get("summary")
            or comm_audit.get("recommendations")
        )
        if isinstance(summary, dict):
            compact = ", ".join(f"{k}={v}" for k, v in list(summary.items())[:4])
            if compact:
                lines.append(f"  - PCIe/SHM AllReduce Thresholds: {compact}")

    moe_line = _extract_moe_breakdown(candidate_dir)
    if moe_line:
        lines.append(f"  - MoE & Router Microbench: {moe_line}")

    attn_line = _extract_attention_breakdown(candidate_dir)
    if attn_line:
        lines.append(f"  - MLA+SWA Attention Microbench: {attn_line}")

    mhc_line = _extract_mhc_engram_breakdown(candidate_dir)
    if mhc_line:
        lines.append(f"  - mHC & Engram Lookup Microbench: {mhc_line}")

    ar_line = _extract_allreduce_budget(candidate_dir)
    if ar_line:
        lines.append(f"  - 8xGPU 80-Call AllReduce Step Budget: {ar_line}")

    route_audit = _safe_load_json(
        candidate_dir / "docs/benchmarks/phase310_6000d_dsv41_post_route_a_audit.json"
    )
    if route_audit:
        prov = route_audit.get("provenance", {})
        banner = prov.get("ready_banner_exp_on")
        if banner:
            lines.append(f"  - Active SM120 Runtime Config: {banner}")

    if baseline_screen_metrics is not None:
        base_screen = _safe_load_json(baseline_screen_metrics)
        if base_screen and isinstance(base_screen.get("tiers"), list):
            tier_parts: list[str] = []
            for row in base_screen["tiers"]:
                overlap_str = ""
                steps = row.get("steps")
                overlapped = row.get("overlapped_decode_steps")
                if isinstance(steps, int) and steps > 0 and isinstance(overlapped, int):
                    overlap_str = f", overlap={overlapped}/{steps}({overlapped / steps:.0%})"
                tier_parts.append(
                    f"{row.get('label')}: out={float(row.get('output_throughput_tok_s', 0.0)):.1f} tok/s, "
                    f"TTFT={float(row.get('median_ttft_ms', 0.0)):.1f}ms, "
                    f"TPOT={float(row.get('mean_tpot_ms', 0.0)):.2f}ms{overlap_str}"
                )
            if tier_parts:
                lines.append("  - Current Baseline Screen Telemetry: " + " | ".join(tier_parts))

    if blocked_niches:
        lines.append(
            "  - [Premise 2-Strike Circuit Breaker — FROZEN NICHES (DO NOT PROPOSE)]:"
        )
        for niche_key, reasons in sorted(blocked_niches.items()):
            last_reason = reasons[-1] if reasons else "consecutive gate rejections"
            lines.append(
                f"    * BLOCKED `{niche_key}` ({len(reasons)} consecutive failures; "
                f"last: {last_reason}) -> Attack a DIFFERENT premise/niche!"
            )

    lines.extend(
        [
            "  - Architectural Reality Check (Avoid Dead-End Mutations):",
            "    1. Decode steps<=max_graph_bucket execute via captured CUDA Graph replay; "
            "packing tiny host H2D/D2H metadata copies or changing Python object allocations "
            "inside capture-only paths yields <0.2% end-to-end gain.",
            "    2. Note `main.rs` disables `device_token_overlap` on `deepseek_v41` only because "
            "of the 3-token Engram window; keeping decode token/Engram state device-resident "
            "with explicit cross-stream event synchronization is a major L2 lever.",
            "    3. Target real hot paths (>3% impact): device-resident decode + Engram window, "
            "SM120 shape-bucket/AllReduce dispatch thresholds (note NCCL beats SHM at small token "
            "counts while SHM beats NCCL at large prefill), or adaptive chunked-prefill token budgets.",
        ]
    )
    return "\n".join(lines)
