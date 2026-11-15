"""Read-only Inductor launch observations, not an optimization or admission path."""

from __future__ import annotations

import hashlib
import json
import math
import os
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

from evolith.lab.store import write_json

MAX_EVENTS = 4096
MAX_LOG_BYTES = 16 << 20
MAX_TENSOR_BYTES = 8 << 20
MAX_COPY_BYTES = 1 << 30


def safe_metadata(value):
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): safe_metadata(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_metadata(v) for v in value]
    return {"unsupported_type": type(value).__name__}


def descriptor(autotuner) -> dict:
    fn = autotuner.fn
    source = getattr(fn, "src", "")
    if not isinstance(source, str):
        source = ""
    launchers = getattr(autotuner, "launchers", [])
    config = launchers[0].config if len(launchers) == 1 else None
    return {
        "name": autotuner.inductor_meta.get("kernel_name", getattr(fn, "__name__", "unknown")),
        "source_sha256": hashlib.sha256(source.encode()).hexdigest() if source else None,
        "kernel_cache_hash": getattr(autotuner, "kernel_hash", None),
        "configuration": None if config is None else {
            "kwargs": safe_metadata(config.kwargs), "num_warps": config.num_warps,
            "num_stages": config.num_stages,
        },
        "binary_cache_hash": getattr(launchers[0], "cache_hash", None) if len(launchers) == 1 else None,
        "autotune_cache_info": safe_metadata(getattr(autotuner, "autotune_cache_info", {})),
    }


def tensor_value(tensor, torch, remaining: int) -> tuple[dict, int]:
    size = tensor.numel() * tensor.element_size()
    metadata = {"shape": list(tensor.shape), "stride": list(tensor.stride()),
                "dtype": str(tensor.dtype), "bytes": size}
    if size > min(MAX_TENSOR_BYTES, remaining) or tensor.device.type == "meta":
        return metadata | {"complete": False, "reason": "copy_budget_or_meta"}, 0
    # Always copy, including CPU inputs. Canonicalization never touches model storage.
    cpu = tensor.detach().to(device="cpu", copy=True).contiguous().reshape(-1)
    raw = cpu.view(torch.uint8).numpy().tobytes()
    metadata["byte_sha256"] = hashlib.sha256(raw).hexdigest()
    if cpu.is_floating_point():
        numeric = cpu.to(torch.float64)
        numeric[numeric == 0] = 0  # Ignore signed-zero representation differences.
        finite = torch.isfinite(numeric)
        metadata["finite"] = bool(finite.all())
        metadata["value_sha256"] = hashlib.sha256(numeric.numpy().tobytes()).hexdigest()
        metadata["sample"] = [x if math.isfinite(x) else None for x in numeric[:16].tolist()]
    else:
        metadata["finite"] = True
        metadata["value_sha256"] = metadata["byte_sha256"]
        metadata["sample"] = cpu[:16].tolist()
    return metadata | {"complete": True}, size


class KernelTrace:
    def __init__(self, directory: Path, rank: int, torch):
        if not directory.is_absolute() or rank not in range(4):
            raise ValueError("Trace requires an absolute directory and one of four ranks.")
        self.directory, self.rank, self.torch = directory, rank, torch
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"rank-{rank}.jsonl"
        self.active = False
        self.finished = False
        self.events = self.log_bytes = self.copied_bytes = self.launch_index = 0
        self.source_bytes = 0
        self.skipped = 0
        self.seen = set()
        self.occurrences = {}
        self.replay_source = None
        self.replayed = False

    def emit(self, data: dict) -> None:
        line = json.dumps({"rank": self.rank, **data}, allow_nan=False, sort_keys=True) + "\n"
        size = len(line.encode())
        if self.events >= MAX_EVENTS or self.log_bytes + size > MAX_LOG_BYTES:
            self.skipped += 1
            return
        with self.path.open("a") as stream:
            stream.write(line)
        self.events += 1
        self.log_bytes += size

    def configuration(self, autotuner) -> dict:
        info = descriptor(autotuner)
        key = json.dumps(info, sort_keys=True)
        if key not in self.seen:
            self.seen.add(key)
            self.emit({"kind": "configuration", **info})
            source = getattr(autotuner.fn, "src", "")
            size = len(source.encode()) if isinstance(source, str) else 0
            if (isinstance(source, str) and source and size <= 256 << 10
                    and self.source_bytes + size <= 4 << 20):
                path = self.directory / "kernels" / f"{info['source_sha256']}.py"
                path.parent.mkdir(exist_ok=True)
                try:
                    with path.open("x") as stream:
                        stream.write(source)
                    self.source_bytes += size
                except FileExistsError:
                    pass
        return info

    def arguments(self, autotuner, args, kwargs, *, outputs: bool) -> dict:
        names = list(autotuner.triton_meta.get("signature", {}))
        bound = dict(zip(names, args))
        bound.update({k: v for k, v in kwargs.items() if k not in ("stream", "benchmark_run")})
        result = {}
        if len(args) > len(names):
            result["__unbound_arguments__"] = {"complete": False}
        mutated = set(autotuner.mutated_arg_names or ())
        for name, value in bound.items():
            # mutated_arg_names covers input mutation, not newly allocated outputs.
            if outputs and name not in mutated and not name.startswith("out_ptr"):
                continue
            if not outputs and name.startswith("out_ptr"):
                continue
            if isinstance(value, self.torch.Tensor):
                data, copied = tensor_value(value, self.torch, MAX_COPY_BYTES - self.copied_bytes)
                self.copied_bytes += copied
                result[name] = data
            elif not outputs:
                complete = type(value) in (int, float, bool, type(None))
                if isinstance(value, float) and not math.isfinite(value):
                    complete = False
                result[name] = {"complete": complete,
                                "scalar": safe_metadata(value)}
        return result

    def call(self, original, autotuner, args, kwargs):
        capture = self.active and not self.torch.cuda.is_current_stream_capturing()
        if self.active and not capture:
            self.skipped += 1
        before = self.arguments(autotuner, args, kwargs, outputs=False) if capture else None
        prepared = None
        if (capture and self.replay_source and not self.replayed
                and descriptor(autotuner)["source_sha256"] == self.replay_source):
            from evolith.lab.kernel_replay import prepare

            prepared = prepare(autotuner, args, self.torch)
        result = original(autotuner, *args, **kwargs)
        info = self.configuration(autotuner)
        if capture:
            identity = (info["name"], info["source_sha256"])
            occurrence = self.occurrences.get(identity, 0)
            self.occurrences[identity] = occurrence + 1
            self.emit({"kind": "values", "index": self.launch_index, "occurrence": occurrence,
                       **info, "inputs": before,
                       "outputs": self.arguments(autotuner, args, kwargs, outputs=True)})
            self.launch_index += 1
        if prepared is not None:
            from evolith.lab.kernel_replay import paired_replay

            paired_replay(autotuner, args, prepared, kwargs, self.torch, self.directory, self.rank)
            self.replayed = True
        return result

    @contextmanager
    def request(self, scheduler_output):
        arm = self.directory / "arm.json"
        selected = False
        if not self.finished and arm.exists():
            settings = json.loads(arm.read_text())
            expected = settings["prompt_token_ids"]
            self.replay_source = settings.get("replay_source")
            requests = scheduler_output.scheduled_new_reqs
            if requests:
                actual = [request.prompt_token_ids for request in requests]
                selected = (actual == [expected]
                            and scheduler_output.total_num_scheduled_tokens == len(expected))
                self.finished = True
                self.emit({"kind": "capture_request", "prompt_token_ids": actual,
                           "matches_expected": selected})
                if not selected:
                    raise ValueError("Trace request differs from declared fixed prefill tokens.")
        self.active = selected
        try:
            yield
        finally:
            self.active = False
            if selected:
                write_json(self.directory / f"rank-{self.rank}-complete.json", {
                    "rank": self.rank, "launches": self.launch_index, "copied_bytes": self.copied_bytes,
                    "events": self.events, "skipped": self.skipped,
                    "scope": "Inductor prefill launches only; not all model operators",
                })


def install(rank: int) -> KernelTrace:
    import torch
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

    directory = os.environ.get("EVOLITH_KERNEL_TRACE_DIR")
    if not directory or getattr(CachingAutotuner.run, "_evolith_trace", False):
        raise ValueError("Missing trace directory or duplicate trace installation.")
    trace = KernelTrace(Path(directory), rank, torch)
    original = CachingAutotuner.run

    @wraps(original)
    def observed(autotuner, *args, **kwargs):
        return trace.call(original, autotuner, args, kwargs)

    observed._evolith_trace = True
    CachingAutotuner.run = observed
    trace.emit({"kind": "installed", "pid": os.getpid(),
                "torch": torch.__version__, "device": str(torch.cuda.current_device())})
    return trace
