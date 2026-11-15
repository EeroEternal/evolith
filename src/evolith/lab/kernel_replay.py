"""Same-input replay of the measured initial RMSNorm, never an admission gate."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import time
from pathlib import Path

TARGET = "7b10b483e1476993e0513ec366676b0ff071082e759ab13265c39d36cae73a1c"


def replay_compiler(autotuner, directory: Path, rank: int):
    # copy.copy invokes CachingAutotuner.__getstate__, which rejects live launchers.
    compiler = type(autotuner).__new__(type(autotuner))
    compiler.__dict__.update(autotuner.__dict__)
    fn = getattr(autotuner, "fn", None)
    if fn is not None and getattr(fn, "_hash_lock", True) is None:
        if hashlib.sha256(fn.src.encode()).hexdigest() != TARGET:
            raise ValueError("Only the measured initial RMSNorm may be reconstructed.")
        path = directory / f"replay-source-{rank}.py"
        text = ("import triton\nimport triton.language as tl\n"
                "from torch._inductor.runtime.triton_helpers import libdevice\n\n"
                "@triton.jit\n" + fn.src + "\n")
        with path.open("x") as stream:
            stream.write(text)
        spec = importlib.util.spec_from_file_location(f"evolith_replay_{rank}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        compiler.fn = getattr(module, fn.src.split("def ", 1)[1].split("(", 1)[0])
    return compiler


def prepare(autotuner, args, torch):
    names = list(autotuner.triton_meta["signature"])[:len(args)]
    if len(names) != len(args) or autotuner.mutated_arg_names:
        raise ValueError("Replay requires fully bound, non-inplace initial RMSNorm.")
    copied = []
    total = 0
    for name, value in zip(names, args):
        if isinstance(value, torch.Tensor):
            total += value.numel() * value.element_size()
            if not value.is_contiguous() or total > 8 << 20:
                raise ValueError("Replay input layout or copy budget exceeded.")
            copied.append(torch.empty_like(value) if name.startswith("out_ptr") else value.detach().clone())
        elif type(value) is int:
            copied.append(value)
        else:
            raise ValueError("Unsupported replay argument.")
    if sum(name.startswith("out_ptr") for name in names) != 1:
        raise ValueError("Replay requires exactly one ordinary output.")
    return names, tuple(copied)


def paired_replay(autotuner, original_args, prepared, kwargs, torch, directory: Path, rank: int):
    import triton

    names, seed = prepared
    selected = autotuner.launchers[0]
    config = selected.config
    if (set(config.kwargs) != {"XBLOCK", "R0_BLOCK"} or config.kwargs["XBLOCK"] != 1
            or config.kwargs["R0_BLOCK"] not in (2048, 8192)):
        raise ValueError("Actual RMSNorm config does not match the measured replay contract.")
    output_index = next(i for i, name in enumerate(names) if name.startswith("out_ptr"))
    actual = original_args[output_index].detach().to("cpu", copy=True)
    inputs = {name: value.cpu() for name, value in zip(names, seed)
              if isinstance(value, torch.Tensor) and not name.startswith("out_ptr")}
    torch.save({"inputs": inputs, "actual": actual}, directory / f"replay-input-rank-{rank}.pt")
    compiler = replay_compiler(autotuner, directory, rank)
    outputs = {}
    for block in (2048, 8192):
        copied = tuple((torch.empty_like(value) if name.startswith("out_ptr") else value.clone())
                       if isinstance(value, torch.Tensor) else value for name, value in zip(names, seed))
        cfg = triton.Config({"XBLOCK": 1, "R0_BLOCK": block},
                            num_warps=config.num_warps, num_stages=config.num_stages)
        launcher = compiler._precompile_config(cfg).make_launcher()
        launcher(*copied, **kwargs)
        outputs[block] = copied[output_index].detach().to("cpu", copy=True)
    assert autotuner.launchers[0] is selected, "Replay must not replace the model launcher."
    selected_matches = torch.equal(actual, outputs[config.kwargs["R0_BLOCK"]])
    difference = (outputs[2048].to(torch.float64) - outputs[8192].to(torch.float64)).abs()
    report = {
        "rank": rank, "scope": "copied real inputs; original compile metadata; no model mutation",
        "selected_block": config.kwargs["R0_BLOCK"], "selected_replay_matches_actual": selected_matches,
        "paired_outputs_equal": torch.equal(outputs[2048], outputs[8192]),
        "different_elements": int((difference != 0).sum()), "max_abs_difference": float(difference.max()),
        "shape": list(actual.shape), "dtype": str(actual.dtype),
    }
    from evolith.lab.store import write_json

    torch.save({"inputs": {name: value.cpu() for name, value in zip(names, seed)
                           if isinstance(value, torch.Tensor) and not name.startswith("out_ptr")},
                "actual": actual, "outputs": outputs}, directory / f"replay-rank-{rank}.pt")
    write_json(directory / f"replay-rank-{rank}.json", report)
    if not selected_matches:
        raise ValueError("Independent replay does not reproduce the actual selected launcher.")
    return report


def main():
    import argparse
    import signal

    from evolith.lab.kernel_diagnosis import case, trace_identity
    from evolith.lab.restart import policy_environment
    from evolith.lab.store import new_run_id, write_json

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--budget-from", type=Path)
    args = parser.parse_args()
    started = time.time()
    budget = {
        "started_unix": started, "deadline_unix": started + 1740,
        "gpus": [0, 1, 2, 3], "approved_seconds": 1800,
    }
    if args.budget_from:
        budget = json.loads(args.budget_from.read_text())
        if (budget["gpus"] != [0, 1, 2, 3] or budget["approved_seconds"] != 1800
                or budget["deadline_unix"] - budget["started_unix"] != 1740):
            raise ValueError("Invalid inherited repair budget.")
    remaining = int(budget["deadline_unix"] - time.time() - 30)
    if remaining < 120:
        raise TimeoutError("Insufficient inherited repair budget.")
    write_json(args.output / "repair-budget.json", budget)

    def expired(*_):
        raise TimeoutError("Same-input probe budget exhausted.")

    def terminate(*_):
        raise SystemExit(143)

    signal.signal(signal.SIGALRM, expired)
    signal.signal(signal.SIGTERM, terminate)
    signal.alarm(min(480, remaining))
    try:
        with policy_environment("native", args.cache):
            report = case(Path("/models/qwen"), args.output, enabled=True, phase="same_input_replay",
                          session=new_run_id(), run_id=new_run_id(), replay_source=TARGET)
        write_json(args.output / "probe-result.json", {
            "run_id": report["run_id"], "trace_source_hash": trace_identity(),
        })
        print(json.dumps({"status": report["status"], "run_id": report["run_id"]}), flush=True)
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
