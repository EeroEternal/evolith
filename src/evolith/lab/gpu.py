from __future__ import annotations

import importlib.metadata
import json
import platform
import random
import subprocess
import sys
from pathlib import Path

import torch

from evolith.lab.contracts import Evaluation, KernelConfig, Shape, content_hash
from evolith.lab.measurement import decide


def environment() -> dict:
    versions = {}
    for name in ("torch", "triton", "vllm", "transformers", "evolith"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    result = {"python": sys.version, "platform": platform.platform(), "packages": versions}
    result["cuda_available"] = torch.cuda.is_available()
    if result["cuda_available"]:
        result.update(
            gpu=torch.cuda.get_device_name(),
            capability=list(torch.cuda.get_device_capability()),
            cuda=torch.version.cuda,
        )
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,name,driver_version,temperature.gpu,clocks.sm,power.draw",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=10, check=False,
        )
        result["gpu_telemetry"] = p.stdout.strip().splitlines() if p.returncode == 0 else []
    return result


def model_info(path: Path) -> dict:
    cfg = json.loads((path / "config.json").read_text())
    text = cfg.get("text_config", cfg)
    if cfg.get("model_type") not in ("qwen3_5", "qwen3_5_moe"):
        raise ValueError("Only Qwen3.5 zero-centered RMSNorm is currently supported.")
    index = path / "model.safetensors.index.json"
    if index.exists():
        shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        missing = [s for s in shards if not (path / s).is_file()]
        if missing:
            raise ValueError(f"Missing model shards: {missing}")
        identity = {"config": cfg, "index": json.loads(index.read_text()),
                    "shards": [(s, (path / s).stat().st_size) for s in shards]}
    else:
        shards = sorted(path.glob("*.safetensors"))
        if not shards:
            raise ValueError("No safetensors weights found.")
        identity = {"config": cfg, "shards": [(p.name, p.stat().st_size) for p in shards]}
    return {
        "hidden": text["hidden_size"], "eps": text.get("rms_norm_eps", 1e-6),
        "model_type": cfg["model_type"], "fingerprint": content_hash(identity),
        "fingerprint_method": "config+index+shard_sizes (not a full weight-content digest)",
    }


def reference(x, residual, weight, eps):
    # Independent float64 oracle; explicitly reproduce the model's BF16 addition.
    value = x.double() if residual is None else (x + residual).double()
    y = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
    y = (y * (1 + weight.double())).to(x.dtype)
    return y if residual is None else (y, (x + residual))


def compiled_baseline(has_residual: bool):
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm

    fn = (GemmaRMSNorm._forward_static_with_residual if has_residual
          else GemmaRMSNorm._forward_static_no_residual)
    return torch.compile(fn, fullgraph=True)


def baseline_call(fn, x, r, w, eps):
    return fn(w, eps, x) if r is None else fn(w, eps, x, r)


def make_inputs(shape: Shape, seed: int, *, padded: bool = False, distribution="normal"):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    width = shape.hidden + (17 if padded else 0)
    x_base = torch.randn((shape.rows, width), generator=generator, device="cuda", dtype=torch.bfloat16)
    r_base = torch.randn(x_base.shape, generator=generator, device="cuda", dtype=x_base.dtype)
    w = torch.randn((shape.hidden,), generator=generator, device="cuda", dtype=x_base.dtype) * 0.2
    if distribution == "zero":
        x_base.zero_()
        r_base.zero_()
    elif distribution == "outlier":
        x_base[:, 0] = 1000
        r_base[:, -1] = -1000
    if padded:
        x_base[:, shape.hidden:] = float("nan")
        r_base[:, shape.hidden:] = 100
    return (x_base[:, :shape.hidden], r_base[:, :shape.hidden] if shape.residual else None, w,
            x_base, r_base)


def check_output(actual, expected) -> None:
    actual = actual if isinstance(actual, tuple) else (actual,)
    expected = expected if isinstance(expected, tuple) else (expected,)
    if len(actual) != len(expected):
        raise AssertionError("Output structure mismatch")
    for a, b in zip(actual, expected):
        if a.shape != b.shape or a.dtype != b.dtype:
            raise AssertionError("Output shape/dtype mismatch")
        if not bool(torch.isfinite(a).all()):
            raise AssertionError("NaN/Inf output")
        # BF16 permits reduction-rounding differences, not arbitrary finite outputs.
        torch.testing.assert_close(a, b, rtol=0.016, atol=0.002)


def verify(config: KernelConfig, shape: Shape, eps: float, seed: int) -> int:
    from evolith.lab.kernels import launch

    checks = 0
    for rows, hidden in {(shape.rows, shape.hidden), (3, shape.hidden), (7, 509)}:
        test = shape.model_copy(update={"rows": rows, "hidden": hidden})
        for distribution in ("normal", "zero", "outlier"):
            for padded in (False, True):
                x, r, w, xb, rb = make_inputs(
                    test, seed + checks, padded=padded, distribution=distribution
                )
                before = [t.view(torch.int16).clone() for t in (xb, rb, w)]
                expected = reference(x.clone(), None if r is None else r.clone(), w.clone(), eps)
                actual = launch(x, r, w, eps, config)
                torch.cuda.synchronize()
                check_output(actual, expected)
                for original, snapshot in zip((xb, rb, w), before):
                    if not torch.equal(original.view(torch.int16), snapshot):
                        raise AssertionError("Read-only input or padding was modified")
                # Repeat with changed values to detect accidental state/caching dependence.
                x.add_(0.125)
                expected = reference(x, r, w, eps)
                check_output(launch(x, r, w, eps, config), expected)
                checks += 2
    return checks


def time_pair(config, shape, eps, seed, rounds, iterations, min_gain):
    from evolith.lab.kernels import launch

    baseline = compiled_baseline(shape.residual)
    pool = [make_inputs(shape, seed + i)[:3] for i in range(8)]
    for x, r, w in pool:
        check_output(baseline_call(baseline, x, r, w, eps), reference(x, r, w, eps))
        launch(x, r, w, eps, config)
    torch.cuda.synchronize()
    # CUDA events measure device work; allocations and launches are included in
    # the actual path. No artificial cache flush: this is a warm working-set test.
    samples = [[], []]
    rng = random.Random(seed)
    for _ in range(rounds):
        order = [0, 1]
        rng.shuffle(order)
        for which in order:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            begin.record()
            for i in range(iterations):
                x, r, w = pool[i % len(pool)]
                if which == 0:
                    _ = baseline_call(baseline, x, r, w, eps)
                else:
                    _ = launch(x, r, w, eps, config)
            end.record()
            end.synchronize()
            samples[which].append(begin.elapsed_time(end) * 1000 / iterations)
    return decide(*samples, min_gain=min_gain, seed=seed)


def evaluate(config, shape, eps, seed, rounds, iterations, min_gain) -> Evaluation:
    result = Evaluation(config=config, shape=shape, correct=False)
    try:
        result.checks = verify(config, shape, eps, seed)
        result.correct = True
        result.timing = time_pair(config, shape, eps, seed, rounds, iterations, min_gain)
    except (RuntimeError, ValueError, AssertionError) as exc:
        result.reasons.append(f"{type(exc).__name__}: {str(exc)[:2000]}")
    return result
