"""Remote container helper for `evolith` PPU/SAIL (`ZW810E * 8`) microbenchmarks & policy checks.

Executed inside the offline SAIL container on `10.8.141.36`. Keeps benchmark inputs
immutable, verifies every candidate against an independent FP32/FP64 reference (and
8-rank agreement for `icn_allreduce`), and measures paired alternating AB/BA rounds.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F


def _init_sail_device_and_stage(dev: torch.device) -> torch.Tensor:
    """Warm up SAIL 1.7 context/DMA/acBLAS and return a reusable 32 MiB pinned H2D buffer."""
    torch.cuda.set_device(dev)
    stage_u8 = torch.zeros(32 * 1024 * 1024, dtype=torch.uint8, pin_memory=True)
    warm_gpu = torch.zeros(4 * 1024 * 1024, dtype=torch.uint8, device=dev)
    warm_gpu.copy_(stage_u8[: 4 * 1024 * 1024], non_blocking=False)
    a = torch.ones((16, 16), dtype=torch.bfloat16, device=dev)
    _ = a @ a
    torch.cuda.synchronize(dev)
    return stage_u8


def _to_gpu_pinned(
    t_cpu: torch.Tensor,
    dev: torch.device,
    stage_u8: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Copy a pageable CPU tensor to `dev` strictly via `pin_memory=True` chunks."""
    src = t_cpu.to(dtype=dtype) if (dtype is not None and t_cpu.dtype != dtype) else t_cpu
    src = src.contiguous()
    dst = torch.zeros(src.shape, dtype=src.dtype, device=dev)
    src_bytes = src.view(torch.uint8).reshape(-1)
    dst_bytes = dst.view(torch.uint8).reshape(-1)
    cap = stage_u8.numel()
    total = src_bytes.numel()
    for off in range(0, total, cap):
        nb = min(cap, total - off)
        stage_u8[:nb].copy_(src_bytes[off : off + nb])
        dst_bytes[off : off + nb].copy_(stage_u8[:nb], non_blocking=False)
    torch.cuda.synchronize(dev)
    return dst


def _dequant_mxfp4_ref(packed_u8: torch.Tensor, scale_u8: torch.Tensor) -> torch.Tensor:
    lut = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
        device=packed_u8.device,
    )
    lo = lut[(packed_u8 & 0xF).long()]
    hi = lut[(packed_u8 >> 4).long()]
    vals = torch.stack([lo, hi], dim=-1).flatten(-2)
    scales = torch.where(
        scale_u8 == 0,
        torch.zeros_like(scale_u8, dtype=torch.float32),
        torch.exp2(scale_u8.to(torch.float32) - 127.0),
    )
    return (vals.unflatten(-1, (-1, 32)) * scales.unsqueeze(-1)).flatten(-2)


def _icn_worker(
    rank: int,
    world_size: int,
    init_file: str,
    baseline_cfg: dict,
    candidate_cfg: dict,
    rounds: int,
    iterations: int,
    seed: int,
    out_file: str,
) -> None:
    from zstep_worker.kernels_sail.ppu_collectives import PpuIcnCollectives

    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "0")
    os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
    os.environ.setdefault("PCCL_SOCKET_IFNAME", "lo")
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    stage_u8 = _init_sail_device_and_stage(dev)
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=120),
    )
    pg = dist.group.WORLD
    b_pccl_min = int(baseline_cfg.get("ar_pccl_min_bytes", 67108864))
    c_pccl_min = int(candidate_cfg.get("ar_pccl_min_bytes", 67108864))
    pccl_pg = None
    if min(b_pccl_min, c_pccl_min) < 67108864:
        pccl_pg = dist.new_group(backend="nccl", timeout=timedelta(seconds=120))
        _w_pccl = torch.zeros(1024, dtype=torch.bfloat16, device=dev)
        dist.all_reduce(_w_pccl, group=pccl_pg)
        torch.cuda.synchronize(dev)

    base_comm = PpuIcnCollectives(
        name=f"evolith_sail_base_{os.getppid()}",
        rank=rank,
        world_size=world_size,
        device=dev,
        exchange_group=pg,
        pccl_group=pccl_pg,
        blocks=int(baseline_cfg["ar_blocks"]),
        threads=int(baseline_cfg["ar_threads"]),
        oneshot_max_bytes=int(baseline_cfg["ar_oneshot_max_bytes"]),
        pccl_min_bytes=b_pccl_min,
        max_bf16_ar_elements=1024 * 5120,
    )
    cand_comm = PpuIcnCollectives(
        name=f"evolith_sail_cand_{os.getppid()}",
        rank=rank,
        world_size=world_size,
        device=dev,
        exchange_group=pg,
        pccl_group=pccl_pg,
        blocks=int(candidate_cfg["ar_blocks"]),
        threads=int(candidate_cfg["ar_threads"]),
        oneshot_max_bytes=int(candidate_cfg["ar_oneshot_max_bytes"]),
        pccl_min_bytes=c_pccl_min,
        max_bf16_ar_elements=1024 * 5120,
    )
    try:
        # Test across representative decode (`M=1`), batch decode (`M=16`), and prefill (`M=512`, `5.12 MB`)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed * 97 + rank + 1)
        x1_cpu = (torch.randn((1, 5120), generator=gen, dtype=torch.float32) * 0.5).to(
            torch.bfloat16
        )
        x8_cpu = (torch.randn((16, 5120), generator=gen, dtype=torch.float32) * 0.5).to(
            torch.bfloat16
        )
        x16_cpu = (torch.randn((512, 5120), generator=gen, dtype=torch.float32) * 0.5).to(
            torch.bfloat16
        )
        x1 = _to_gpu_pinned(x1_cpu, dev, stage_u8)
        x8 = _to_gpu_pinned(x8_cpu, dev, stage_u8)
        x16 = _to_gpu_pinned(x16_cpu, dev, stage_u8)
        x1_orig = x1.clone()
        x8_orig = x8.clone()
        x16_orig = x16.clone()

        # FP64 reference across all 8 ranks
        ref1_cpu = x1_cpu.float().double()
        ref8_cpu = x8_cpu.float().double()
        ref16_cpu = x16_cpu.float().double()
        dist.all_reduce(ref1_cpu, group=pg)
        dist.all_reduce(ref8_cpu, group=pg)
        dist.all_reduce(ref16_cpu, group=pg)
        ref1 = _to_gpu_pinned(ref1_cpu.to(torch.bfloat16), dev, stage_u8)
        ref8 = _to_gpu_pinned(ref8_cpu.to(torch.bfloat16), dev, stage_u8)
        ref16 = _to_gpu_pinned(ref16_cpu.to(torch.bfloat16), dev, stage_u8)

        y1_b = torch.zeros_like(x1)
        y8_b = torch.zeros_like(x8)
        y16_b = torch.zeros_like(x16)
        y1_c = torch.zeros_like(x1)
        y8_c = torch.zeros_like(x8)
        y16_c = torch.zeros_like(x16)

        for _ in range(4):
            base_comm.all_reduce_bf16(x1, output=y1_b)
            base_comm.all_reduce_bf16(x8, output=y8_b)
            base_comm.all_reduce_bf16(x16, output=y16_b)
            cand_comm.all_reduce_bf16(x1, output=y1_c)
            cand_comm.all_reduce_bf16(x8, output=y8_c)
            cand_comm.all_reduce_bf16(x16, output=y16_c)
        torch.cuda.synchronize(dev)

        inputs_unchanged = bool(
            torch.equal(x1, x1_orig) and torch.equal(x8, x8_orig) and torch.equal(x16, x16_orig)
        )
        err1 = float((y1_c.float() - ref1.float()).abs().max().item())
        err8 = float((y8_c.float() - ref8.float()).abs().max().item())
        err16 = float((y16_c.float() - ref16.float()).abs().max().item())
        ref_ok = err1 <= 0.0625 and err8 <= 0.0625 and err16 <= 0.0625

        # Verify all 8 ranks produced identical BF16 bit-patterns on BOTH ICNLink (M=16) and PCCL (M=512)
        y8_cpu = y8_c.view(torch.int16).to(torch.int32).cpu()
        g8 = [torch.empty_like(y8_cpu) for _ in range(world_size)]
        dist.all_gather(g8, y8_cpu, group=pg)
        y16_cpu = y16_c.view(torch.int16).to(torch.int32).cpu()
        gathered = [torch.empty_like(y16_cpu) for _ in range(world_size)]
        dist.all_gather(gathered, y16_cpu, group=pg)
        ranks_agree = all(torch.equal(g8[0], g) for g in g8[1:]) and all(
            torch.equal(gathered[0], g) for g in gathered[1:]
        )

        ok_tensor = torch.tensor(
            [1 if (inputs_unchanged and ref_ok and ranks_agree) else 0], dtype=torch.int32
        )
        dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN, group=pg)

        baseline_us: list[float] = []
        candidate_us: list[float] = []

        def _time_comm(
            comm: PpuIcnCollectives,
            out1: torch.Tensor,
            out8: torch.Tensor,
            out16: torch.Tensor,
        ) -> float:
            dist.barrier(group=pg)
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()
            for _ in range(iterations):
                comm.all_reduce_bf16(x1, output=out1)
                comm.all_reduce_bf16(x8, output=out8)
                comm.all_reduce_bf16(x16, output=out16)
            ev1.record()
            ev1.synchronize()
            return float(ev0.elapsed_time(ev1)) * 1000.0 / float(3 * iterations)

        for r_idx in range(rounds):
            if r_idx % 2 == 0:
                b_t = _time_comm(base_comm, y1_b, y8_b, y16_b)
                c_t = _time_comm(cand_comm, y1_c, y8_c, y16_c)
            else:
                c_t = _time_comm(cand_comm, y1_c, y8_c, y16_c)
                b_t = _time_comm(base_comm, y1_b, y8_b, y16_b)
            baseline_us.append(b_t)
            candidate_us.append(c_t)

        if rank == 0:
            Path(out_file).write_text(
                json.dumps(
                    {
                        "family": "icn_allreduce",
                        "seed": seed,
                        "rounds": rounds,
                        "iterations": iterations,
                        "world_size": world_size,
                        "baseline_config": baseline_cfg,
                        "config": candidate_cfg,
                        "inputs_unchanged": inputs_unchanged,
                        "reference_ok": bool(ok_tensor.item() == 1),
                        "ranks_agree": ranks_agree,
                        "baseline_us": baseline_us,
                        "candidate_us": candidate_us,
                    }
                ),
                encoding="utf-8",
            )
    finally:
        cand_comm.close()
        base_comm.close()
        dist.destroy_process_group()


def _run_micro(args: argparse.Namespace) -> dict:
    from zstep_worker.kernels_sail import dsv41_sail_ops as ops

    baseline_cfg = json.loads(args.baseline)
    candidate_cfg = json.loads(args.candidate)
    family = args.family
    seed = int(args.seed)
    rounds = int(args.rounds)
    iterations = int(args.iterations)

    if family == "icn_allreduce":
        with tempfile.TemporaryDirectory() as tmp:
            init_f = str(Path(tmp) / "gloo_init")
            out_f = str(Path(tmp) / "icn_out.json")
            mp.spawn(
                _icn_worker,
                args=(8, init_f, baseline_cfg, candidate_cfg, rounds, iterations, seed, out_f),
                nprocs=8,
                join=True,
            )
            return json.loads(Path(out_f).read_text(encoding="utf-8"))

    torch.cuda.set_device(0)
    dev = torch.device("cuda:0")
    stage_u8 = _init_sail_device_and_stage(dev)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed * 131 + 17)

    if family == "sparse_attn_tp8":
        arena = _to_gpu_pinned(
            torch.randn((65536, 1, 512), generator=gen, dtype=torch.float32) * 0.2,
            dev,
            stage_u8,
            dtype=torch.bfloat16,
        )
        sink = _to_gpu_pinned(
            torch.randn((8,), generator=gen, dtype=torch.float32), dev, stage_u8
        )
        # Benchmark representative decode (`M=8`) + prefill (`M=64`) sparse MQA (`topk=640`)
        q8 = _to_gpu_pinned(
            torch.randn((8, 8, 512), generator=gen, dtype=torch.float32) * 0.2,
            dev,
            stage_u8,
            dtype=torch.bfloat16,
        )
        q64 = _to_gpu_pinned(
            torch.randn((64, 8, 512), generator=gen, dtype=torch.float32) * 0.2,
            dev,
            stage_u8,
            dtype=torch.bfloat16,
        )
        idx8_cpu = torch.randint(0, 65536, (8, 640), generator=gen, dtype=torch.int32)
        idx8_cpu[:, 512:] = -1
        idx8 = _to_gpu_pinned(idx8_cpu, dev, stage_u8)
        idx64_cpu = torch.randint(0, 65536, (64, 640), generator=gen, dtype=torch.int32)
        idx64_cpu[:, 512:] = -1
        idx64 = _to_gpu_pinned(idx64_cpu, dev, stage_u8)

        q8_orig, q64_orig = q8.clone(), q64.clone()
        arena_orig = arena.clone()
        sm_scale = 512**-0.5

        def _ref_attn(q_in: torch.Tensor, idx_in: torch.Tensor) -> torch.Tensor:
            out_r = torch.zeros_like(q_in, dtype=torch.float32)
            for m in range(q_in.shape[0]):
                vm = idx_in[m] >= 0
                if bool(vm.any().item()):
                    kv_m = arena[idx_in[m][vm].long(), 0, :].float()
                    s = (q_in[m].float() @ kv_m.T) * sm_scale
                    p = torch.softmax(torch.cat([s, sink.unsqueeze(-1)], dim=-1), dim=-1)[
                        :, :-1
                    ]
                    out_r[m] = p @ kv_m
            return out_r

        ref8 = _ref_attn(q8, idx8)
        ref64 = _ref_attn(q64, idx64)

        p_acc = torch.zeros((64 * 64, 512), dtype=torch.float32, device=dev)
        p_m = torch.zeros((64 * 64,), dtype=torch.float32, device=dev)
        p_l = torch.zeros((64 * 64,), dtype=torch.float32, device=dev)
        q_pad = torch.zeros((64, 64, 512), dtype=torch.bfloat16, device=dev)
        o8 = torch.zeros_like(q8)
        o64 = torch.zeros_like(q64)

        b_bk = str(baseline_cfg["sparse_attn_backend"])
        c_bk = str(candidate_cfg["sparse_attn_backend"])
        for bk in (b_bk, c_bk):
            for _ in range(3):
                ops.sparse_attn_tp8_forward(
                    q8,
                    arena,
                    sink,
                    idx8,
                    out=o8,
                    partial_acc=p_acc[: 8 * 64],
                    partial_m=p_m[: 8 * 64],
                    partial_l=p_l[: 8 * 64],
                    q_pad64_buf=q_pad[:8],
                    sm_scale=sm_scale,
                    backend=bk,
                )
                ops.sparse_attn_tp8_forward(
                    q64,
                    arena,
                    sink,
                    idx64,
                    out=o64,
                    partial_acc=p_acc,
                    partial_m=p_m,
                    partial_l=p_l,
                    q_pad64_buf=q_pad,
                    sm_scale=sm_scale,
                    backend=bk,
                )
        torch.cuda.synchronize(dev)

        inputs_unchanged = bool(
            torch.equal(q8, q8_orig)
            and torch.equal(q64, q64_orig)
            and torch.equal(arena, arena_orig)
        )
        max_err8 = float((o8.float() - ref8).abs().max().item())
        max_err64 = float((o64.float() - ref64).abs().max().item())
        reference_ok = max_err8 < 2.0e-3 and max_err64 < 2.0e-3

        def _time_bk(bk: str) -> float:
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()
            for _ in range(iterations):
                ops.sparse_attn_tp8_forward(
                    q8,
                    arena,
                    sink,
                    idx8,
                    out=o8,
                    partial_acc=p_acc[: 8 * 64],
                    partial_m=p_m[: 8 * 64],
                    partial_l=p_l[: 8 * 64],
                    q_pad64_buf=q_pad[:8],
                    sm_scale=sm_scale,
                    backend=bk,
                )
                ops.sparse_attn_tp8_forward(
                    q64,
                    arena,
                    sink,
                    idx64,
                    out=o64,
                    partial_acc=p_acc,
                    partial_m=p_m,
                    partial_l=p_l,
                    q_pad64_buf=q_pad,
                    sm_scale=sm_scale,
                    backend=bk,
                )
            ev1.record()
            ev1.synchronize()
            return float(ev0.elapsed_time(ev1)) * 1000.0 / float(2 * iterations)

        baseline_us, candidate_us = [], []
        for r_idx in range(rounds):
            if r_idx % 2 == 0:
                bt, ct = _time_bk(b_bk), _time_bk(c_bk)
            else:
                ct, bt = _time_bk(c_bk), _time_bk(b_bk)
            baseline_us.append(bt)
            candidate_us.append(ct)

        return {
            "family": family,
            "seed": seed,
            "rounds": rounds,
            "iterations": iterations,
            "world_size": 1,
            "baseline_config": baseline_cfg,
            "config": candidate_cfg,
            "inputs_unchanged": inputs_unchanged,
            "reference_ok": reference_ok,
            "ranks_agree": True,
            "baseline_us": baseline_us,
            "candidate_us": candidate_us,
        }

    if family in ("mxfp4_moe_decode", "mxfp4_moe_prefill"):
        e_cnt, top_k, h_dim, i_tp = 64, 6, 5120, 288
        row_set = (8, 24) if family == "mxfp4_moe_decode" else (64, 128)
        max_m = max(row_set)
        xs = {
            m_r: _to_gpu_pinned(
                torch.randn((m_r, h_dim), generator=gen, dtype=torch.float32) * 0.5,
                dev,
                stage_u8,
                dtype=torch.bfloat16,
            )
            for m_r in row_set
        }
        xs_orig = {m_r: t.clone() for m_r, t in xs.items()}
        gate_w = _to_gpu_pinned(
            torch.randn((e_cnt, h_dim), generator=gen, dtype=torch.float32) * 0.02,
            dev,
            stage_u8,
        )
        gate_b = _to_gpu_pinned(
            torch.randn((e_cnt,), generator=gen, dtype=torch.float32) * 0.05,
            dev,
            stage_u8,
        )
        sh_w13 = _to_gpu_pinned(
            torch.randn((2 * i_tp, h_dim), generator=gen, dtype=torch.float32) * 0.02,
            dev,
            stage_u8,
            dtype=torch.bfloat16,
        )
        sh_w2 = _to_gpu_pinned(
            torch.randn((h_dim, i_tp), generator=gen, dtype=torch.float32) * 0.02,
            dev,
            stage_u8,
            dtype=torch.bfloat16,
        )
        exp_w13_p = _to_gpu_pinned(
            torch.randint(
                0, 256, (e_cnt, 2 * i_tp, h_dim // 2), generator=gen, dtype=torch.uint8
            ),
            dev,
            stage_u8,
        )
        exp_w13_s = _to_gpu_pinned(
            torch.randint(
                120, 126, (e_cnt, 2 * i_tp, h_dim // 32), generator=gen, dtype=torch.uint8
            ),
            dev,
            stage_u8,
        )
        exp_w2_p = _to_gpu_pinned(
            torch.randint(
                0, 256, (e_cnt, h_dim, i_tp // 2), generator=gen, dtype=torch.uint8
            ),
            dev,
            stage_u8,
        )
        exp_w2_s = _to_gpu_pinned(
            torch.randint(
                120, 126, (e_cnt, h_dim, i_tp // 32), generator=gen, dtype=torch.uint8
            ),
            dev,
            stage_u8,
        )

        w13_ref = _dequant_mxfp4_ref(exp_w13_p, exp_w13_s).to(torch.bfloat16)
        w2_ref = _dequant_mxfp4_ref(exp_w2_p, exp_w2_s).to(torch.bfloat16)

        r_buf = torch.zeros((max_m, e_cnt), dtype=torch.float32, device=dev)
        tw_buf = torch.zeros((max_m, top_k), dtype=torch.float32, device=dev)
        ti_buf = torch.zeros((max_m, top_k), dtype=torch.int32, device=dev)
        sgu_buf = torch.zeros((max_m, 2 * i_tp), dtype=torch.bfloat16, device=dev)
        so_buf = torch.zeros((max_m, h_dim), dtype=torch.bfloat16, device=dev)
        mi_buf = torch.zeros((max_m * top_k, i_tp), dtype=torch.bfloat16, device=dev)
        max_tiles = max_m * top_k // 16 + e_cnt + 16
        sp_buf = torch.full((max_tiles * 64,), -1, dtype=torch.int32, device=dev)
        ei_buf = torch.zeros((max_tiles,), dtype=torch.int32, device=dev)
        nv_buf = torch.zeros((1,), dtype=torch.int32, device=dev)
        po_buf = torch.zeros((max_m * top_k, h_dim), dtype=torch.bfloat16, device=dev)
        mo_buf = torch.zeros((max_m, h_dim), dtype=torch.bfloat16, device=dev)
        aq_buf = torch.zeros((max_m, h_dim), dtype=torch.bfloat16, device=dev)

        def _exec_one(xm: torch.Tensor, cfg: dict) -> torch.Tensor:
            return ops.dsv41_moe_tp8_forward(
                xm,
                gate_weight=gate_w,
                gate_bias=gate_b,
                shared_w13_bf16=sh_w13,
                shared_w2_bf16=sh_w2,
                experts_w13_packed=exp_w13_p,
                experts_w13_scale=exp_w13_s,
                experts_w2_packed=exp_w2_p,
                experts_w2_scale=exp_w2_s,
                top_k=top_k,
                route_scale=1.5,
                swiglu_limit=10.0,
                router_logits_buf=r_buf,
                topk_weights_buf=tw_buf,
                topk_ids_buf=ti_buf,
                shared_gu_buf=sgu_buf,
                shared_out_buf=so_buf,
                moe_inter_buf=mi_buf,
                sorted_pair_ids_buf=sp_buf,
                expert_ids_per_tile_buf=ei_buf,
                num_valid_tiles_buf=nv_buf,
                pair_out_buf=po_buf,
                moe_out_buf=mo_buf,
                act_quant_x_buf=aq_buf,
                gemv_max_rows=int(cfg["moe_gemv_max_rows"]),
                gemv_threads=int(cfg["moe_gemv_threads"]),
                block_m=int(cfg["moe_block_m"]),
                num_warps=int(cfg["moe_num_warps"]),
            )

        for cfg in (baseline_cfg, candidate_cfg):
            for _ in range(3):
                for m_r in row_set:
                    _exec_one(xs[m_r], cfg)
        y_cands = {m_r: _exec_one(xs[m_r], candidate_cfg).clone() for m_r in row_set}
        torch.cuda.synchronize(dev)

        # Independent FP32 reference with identical UE8M0 input activation quantization
        max_rel_err = 0.0
        for m_r in row_set:
            xm = xs[m_r]
            xq_ref = ops.inplace_act_quant_ue8m0(xm.clone())
            rl_ref = xm.float() @ gate_w.T
            sp_ref = torch.where(rl_ref > 20.0, rl_ref, torch.log1p(torch.exp(rl_ref)))
            sc = sp_ref.clamp_min(0.0).sqrt()
            # Deterministic tie-break `(score DESC, expert_id ASC)` matching router contract
            biased_sc = sc + gate_b - torch.arange(e_cnt, device=dev, dtype=torch.float32) * 1.0e-12
            idx_ref = biased_sc.topk(top_k, dim=-1)[1]
            wt_ref = sc.gather(1, idx_ref)
            wt_ref = (wt_ref / (wt_ref.sum(-1, keepdim=True) + 1.0e-20)) * 1.5
            y_ref = torch.zeros((m_r, h_dim), dtype=torch.float32, device=dev)
            for e_id in range(e_cnt):
                r_i, k_i = torch.where(idx_ref == e_id)
                if r_i.numel() == 0:
                    continue
                gu = xq_ref[r_i].float() @ w13_ref[e_id].float().T
                g, u = gu.split(i_tp, dim=-1)
                act = (
                    (F.silu(g.clamp(max=10.0)) * u.clamp(-10.0, 10.0))
                    * wt_ref[r_i, k_i].unsqueeze(-1)
                ).to(torch.bfloat16)
                ops.inplace_act_quant_ue8m0(act)
                # Match operator contract: each `(token, expert)` pair down-proj accumulates in FP32 -> BF16
                y_ref[r_i] += (act.float() @ w2_ref[e_id].float().T).to(torch.bfloat16).float()
            sh_gu_r = xq_ref.float() @ sh_w13.float().T
            sg, su = sh_gu_r.split(i_tp, dim=-1)
            sh_act_r = (F.silu(sg.clamp(max=10.0)) * su.clamp(-10.0, 10.0)).to(torch.bfloat16)
            ops.inplace_act_quant_ue8m0(sh_act_r)
            sh_out_r = (sh_act_r.float() @ sh_w2.float().T).to(torch.bfloat16).float()
            y_ref_bf16 = (y_ref + sh_out_r).to(torch.bfloat16).float()
            rel_err = float((y_cands[m_r].float() - y_ref_bf16).abs().max().item()) / max(
                float(y_ref_bf16.abs().max().item()), 1.0e-5
            )
            max_rel_err = max(max_rel_err, rel_err)

        inputs_unchanged = all(torch.equal(xs[m_r], xs_orig[m_r]) for m_r in row_set)
        reference_ok = max_rel_err < 0.015

        def _time_moe(cfg: dict) -> float:
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()
            for _ in range(iterations):
                for m_r in row_set:
                    _exec_one(xs[m_r], cfg)
            ev1.record()
            ev1.synchronize()
            return float(ev0.elapsed_time(ev1)) * 1000.0 / float(len(row_set) * iterations)

        baseline_us, candidate_us = [], []
        for r_idx in range(rounds):
            if r_idx % 2 == 0:
                bt, ct = _time_moe(baseline_cfg), _time_moe(candidate_cfg)
            else:
                ct, bt = _time_moe(candidate_cfg), _time_moe(baseline_cfg)
            baseline_us.append(bt)
            candidate_us.append(ct)

        return {
            "family": family,
            "seed": seed,
            "rounds": rounds,
            "iterations": iterations,
            "world_size": 1,
            "baseline_config": baseline_cfg,
            "config": candidate_cfg,
            "inputs_unchanged": inputs_unchanged,
            "reference_ok": reference_ok,
            "ranks_agree": True,
            "baseline_us": baseline_us,
            "candidate_us": candidate_us,
        }

    if family == "mhc_mixes":
        row_set = (8, 24, 256)
        max_m = max(row_set)
        norm_eps = 1.0e-20
        hc_eps = 1.0e-6
        sinkhorn_iters = 20
        xs = {
            m_r: _to_gpu_pinned(
                torch.randn((m_r, 4, 5120), generator=gen, dtype=torch.float32) * 0.5,
                dev,
                stage_u8,
                dtype=torch.bfloat16,
            )
            for m_r in row_set
        }
        xs_orig = {m_r: t.clone() for m_r, t in xs.items()}
        hc_fn_f32 = _to_gpu_pinned(
            torch.randn((24, 20480), generator=gen, dtype=torch.float32) * 0.02,
            dev,
            stage_u8,
        )
        hc_fn_bf16 = hc_fn_f32.to(torch.bfloat16).contiguous()
        hc_scale = _to_gpu_pinned(
            torch.randn((3,), generator=gen, dtype=torch.float32).abs() * 0.2 + 0.05,
            dev,
            stage_u8,
        )
        hc_base = _to_gpu_pinned(
            torch.randn((24,), generator=gen, dtype=torch.float32) * 0.1,
            dev,
            stage_u8,
        )

        pre_buf = torch.zeros((max_m, 4), dtype=torch.float32, device=dev)
        post_buf = torch.zeros((max_m, 4), dtype=torch.float32, device=dev)
        comb_buf = torch.zeros((max_m, 4, 4), dtype=torch.float32, device=dev)
        proj_buf = torch.zeros((max_m, 24), dtype=torch.float32, device=dev)
        part_buf = torch.zeros((max_m * 16 * 25,), dtype=torch.float32, device=dev)

        def _exec_mhc(xm: torch.Tensor, cfg: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            m_r = xm.shape[0]
            return ops.mhc_mixes_sinkhorn(
                xm,
                hc_fn_f32,
                hc_fn_bf16,
                hc_scale,
                hc_base,
                pre_out=pre_buf[:m_r],
                post_out=post_buf[:m_r],
                comb_out=comb_buf[:m_r],
                raw_proj_buf=proj_buf[:m_r],
                split_k_part_buf=part_buf,
                split_k=int(cfg["mhc_split_k"]),
                fused_max_rows=int(cfg["mhc_fused_max_rows"]),
                norm_eps=norm_eps,
                hc_eps=hc_eps,
                sinkhorn_iters=sinkhorn_iters,
            )

        for cfg in (baseline_cfg, candidate_cfg):
            for _ in range(3):
                for m_r in row_set:
                    _exec_mhc(xs[m_r], cfg)
        cand_outs = {}
        for m_r in row_set:
            pr, po, cb = _exec_mhc(xs[m_r], candidate_cfg)
            cand_outs[m_r] = (pr.clone(), po.clone(), cb.clone())
        torch.cuda.synchronize(dev)

        # Independent FP64 Sinkhorn reference
        max_abs_err = 0.0
        for m_r in row_set:
            x_f64 = xs[m_r].reshape(m_r, 20480).float().double()
            inv_rms = torch.rsqrt((x_f64 * x_f64).mean(dim=-1, keepdim=True) + norm_eps)
            raw_m = (x_f64 @ hc_fn_f32.double().T) * inv_rms
            sc_d = hc_scale.double()
            bs_d = hc_base.double()
            pre_r = torch.sigmoid(raw_m[:, 0:4] * sc_d[0] + bs_d[0:4]) + hc_eps
            post_r = 2.0 * torch.sigmoid(raw_m[:, 4:8] * sc_d[1] + bs_d[4:8])
            c_log = (raw_m[:, 8:24] * sc_d[2] + bs_d[8:24]).view(m_r, 4, 4)
            comb_r = torch.softmax(c_log, dim=-1) + hc_eps
            comb_r = comb_r / (comb_r.sum(dim=-2, keepdim=True) + hc_eps)
            for _ in range(sinkhorn_iters - 1):
                comb_r = comb_r / (comb_r.sum(dim=-1, keepdim=True) + hc_eps)
                comb_r = comb_r / (comb_r.sum(dim=-2, keepdim=True) + hc_eps)
            pr_c, po_c, cb_c = cand_outs[m_r]
            err = max(
                float((pr_c.double() - pre_r).abs().max().item()),
                float((po_c.double() - post_r).abs().max().item()),
                float((cb_c.double() - comb_r).abs().max().item()),
            )
            max_abs_err = max(max_abs_err, err)

        inputs_unchanged = all(torch.equal(xs[m_r], xs_orig[m_r]) for m_r in row_set)
        reference_ok = max_abs_err < 1.0e-4

        def _time_mhc(cfg: dict) -> float:
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()
            for _ in range(iterations):
                for m_r in row_set:
                    _exec_mhc(xs[m_r], cfg)
            ev1.record()
            ev1.synchronize()
            return float(ev0.elapsed_time(ev1)) * 1000.0 / float(len(row_set) * iterations)

        baseline_us, candidate_us = [], []
        for r_idx in range(rounds):
            if r_idx % 2 == 0:
                bt, ct = _time_mhc(baseline_cfg), _time_mhc(candidate_cfg)
            else:
                ct, bt = _time_mhc(candidate_cfg), _time_mhc(baseline_cfg)
            baseline_us.append(bt)
            candidate_us.append(ct)

        return {
            "family": family,
            "seed": seed,
            "rounds": rounds,
            "iterations": iterations,
            "world_size": 1,
            "baseline_config": baseline_cfg,
            "config": candidate_cfg,
            "inputs_unchanged": inputs_unchanged,
            "reference_ok": reference_ok,
            "ranks_agree": True,
            "baseline_us": baseline_us,
            "candidate_us": candidate_us,
        }

    raise ValueError(f"Unsupported SAIL microbenchmark family: {family!r}")


def _parse_applied_sail_from_log(log_text: str) -> dict | None:
    m = re.search(
        r"sail=(\w+)\(ar=(\d+)x(\d+),os=(\d+),pccl=(\d+),moe_gemv<=(\d+)@(\d+),bm=(\d+)w(\d+),attn=(\w+),mhc=(\d+)r(\d+)\)",
        log_text,
    )
    if not m:
        return None
    return {
        "backend": m.group(1),
        "ar_blocks": int(m.group(2)),
        "ar_threads": int(m.group(3)),
        "ar_oneshot_max_bytes": int(m.group(4)),
        "ar_pccl_min_bytes": int(m.group(5)),
        "moe_gemv_max_rows": int(m.group(6)),
        "moe_gemv_threads": int(m.group(7)),
        "moe_block_m": int(m.group(8)),
        "moe_num_warps": int(m.group(9)),
        "sparse_attn_backend": m.group(10),
        "mhc_split_k": int(m.group(11)),
        "mhc_fused_max_rows": int(m.group(12)),
    }


def _run_zstep_model(args: argparse.Namespace) -> dict:
    policy_cfg = json.loads(args.policy)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    policy_path = work_dir / "sail_policy.json"
    policy_path.write_text(json.dumps(policy_cfg, indent=2) + "\n", encoding="utf-8")

    env = os.environ.copy()
    env["ZSTEP_BACKEND"] = "sail"
    env["ZSTEP_SAIL_POLICY_PATH"] = str(policy_path)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    env["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
    env["TORCH_NCCL_ENABLE_MONITORING"] = "0"
    env["NCCL_SOCKET_IFNAME"] = "lo"
    env["PCCL_SOCKET_IFNAME"] = "lo"
    obs_prefix = work_dir / "obs"
    if args.observe:
        env["ZSTEP_SAIL_OBSERVE_PATH"] = str(obs_prefix)
    else:
        env.pop("ZSTEP_SAIL_OBSERVE_PATH", None)

    metrics_path = work_dir / "bench_metrics.json"
    py_bin = sys.executable
    cmd = [
        args.zstep_bin,
        "--world-size",
        "8",
        "--python",
        py_bin,
        "--worker-root",
        args.worker_root,
        "--model-dir",
        args.model_dir,
        "--prompt-ids",
        args.prompt_ids,
        "--max-new-tokens",
        str(args.decode_tokens),
        "--bench-tiers",
        args.bench_tier,
        "--warmup-prompts",
        str(args.warmup_prompts),
        "--repeats",
        "1",
        "--metrics-output",
        str(metrics_path),
        "--ready-timeout-secs",
        "600",
        "--step-timeout-secs",
        "180",
        "--heartbeat-timeout-ms",
        "30000",
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    combined_log = proc.stdout + "\n" + proc.stderr
    (work_dir / "zstep.log").write_text(combined_log, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"zstep full-model run exited {proc.returncode}: {combined_log[-1500:]}"
        )

    gen_match = re.search(r"GENERATED_IDS=\[([0-9,\s-]*)\]", proc.stdout)
    if not gen_match:
        raise ValueError("Missing GENERATED_IDS in zstep output")
    raw_ids = gen_match.group(1).strip()
    tokens = [int(x.strip()) for x in raw_ids.split(",") if x.strip()] if raw_ids else []

    applied = _parse_applied_sail_from_log(combined_log)
    bench = json.loads(metrics_path.read_text(encoding="utf-8"))
    tier0 = bench["tiers"][0]
    out_tok_s = float(tier0["output_throughput_tok_s"])
    total_tok_s = float(tier0["total_throughput_tok_s"])
    mean_tpot_ms = float(tier0["mean_tpot_ms"])
    # Prefer zstep's monotonic Rust wall_time_ms (`0.001 ms = 1 us` precision)
    if "wall_time_ms" in tier0 and float(tier0["wall_time_ms"]) > 0.0:
        tier_wall_us = float(tier0["wall_time_ms"]) * 1000.0
    else:
        tier_out_tokens = int(tier0["num_prompts"]) * int(tier0["output_len"])
        tier_wall_us = (tier_out_tokens / out_tok_s) * 1.0e6 if out_tok_s > 0 else 0.0

    out_payload: dict = {
        "applied_policy": applied,
        "tokens": tokens,
        "tier_wall_us": tier_wall_us,
        "output_throughput_tok_s": out_tok_s,
        "total_throughput_tok_s": total_tok_s,
        "mean_tpot_ms": mean_tpot_ms,
        "median_ttft_ms": float(tier0["median_ttft_ms"]),
    }

    if args.observe:
        agg: dict[str, dict] = {}
        rank_files = [Path(f"{obs_prefix}.rank{r}.json") for r in range(8)]
        if not all(p.is_file() for p in rank_files):
            raise ValueError("Missing per-rank ZSTEP_SAIL_OBSERVE_PATH JSON outputs")
        for r_idx, rf in enumerate(rank_files):
            rdata = json.loads(rf.read_text(encoding="utf-8"))
            for fam, fstats in rdata.get("families", {}).items():
                slot = agg.setdefault(
                    fam,
                    {
                        "name": fam,
                        "calls_per_rank": int(fstats["calls"]),
                        "rank_estimated_us": [],
                    },
                )
                slot["rank_estimated_us"].append(float(fstats["estimated_total_us"]))
        total_mean_us = 0.0
        families_list: list[dict] = []
        supported_names = {
            "sparse_attn_tp8",
            "icn_allreduce",
            "mxfp4_moe_decode",
            "mxfp4_moe_prefill",
            "mhc_mixes",
        }
        gap_reasons = {
            "attn_qkv_o_proj": (
                "Vendor acBLAS dense BF16 Q/KV/O LoRA projections have no parameterized policy family"
            ),
            "engram_compressor_indexer": (
                "Stateless Engram lookup and MLA Compressor/Indexer kernels have no parameterized policy family"
            ),
        }
        for fam, slot in sorted(agg.items()):
            mean_est = sum(slot["rank_estimated_us"]) / len(slot["rank_estimated_us"])
            total_mean_us += mean_est
            is_sup = fam in supported_names
            families_list.append(
                {
                    "name": fam,
                    "supported": is_sup,
                    "calls_per_rank": slot["calls_per_rank"],
                    "estimated_total_us": round(mean_est, 3),
                    "reason": (
                        None
                        if is_sup
                        else gap_reasons.get(
                            fam,
                            "Stateless Engram/Indexer/mHC kernels have no parameterized policy family yet",
                        )
                    ),
                }
            )
        for item in families_list:
            item["share"] = (
                round(item["estimated_total_us"] / total_mean_us, 6)
                if total_mean_us > 0.0
                else 0.0
            )
        out_payload["observation"] = {
            "total_estimated_us": round(total_mean_us, 3),
            "families": families_list,
        }

    return out_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_micro = sub.add_parser("micro")
    p_micro.add_argument("--family", required=True)
    p_micro.add_argument("--baseline", required=True)
    p_micro.add_argument("--candidate", required=True)
    p_micro.add_argument("--seed", type=int, required=True)
    p_micro.add_argument("--rounds", type=int, default=11)
    p_micro.add_argument("--iterations", type=int, default=30)

    p_model = sub.add_parser("model")
    p_model.add_argument("--zstep-bin", required=True)
    p_model.add_argument("--worker-root", required=True)
    p_model.add_argument("--model-dir", required=True)
    p_model.add_argument("--policy", required=True)
    p_model.add_argument("--work-dir", required=True)
    p_model.add_argument("--prompt-ids", default="671,6102,294,8760,344")
    p_model.add_argument("--decode-tokens", type=int, default=32)
    p_model.add_argument("--bench-tier", default="evolve:8:16:256:64")
    p_model.add_argument("--warmup-prompts", type=int, default=4)
    p_model.add_argument("--observe", action="store_true")

    args = parser.parse_args()
    if args.cmd == "micro":
        res = _run_micro(args)
    else:
        res = _run_zstep_model(args)
    print(json.dumps(res, allow_nan=False))


if __name__ == "__main__":
    main()
