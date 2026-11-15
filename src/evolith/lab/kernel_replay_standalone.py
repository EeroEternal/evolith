"""Replay preserved real tensors without starting another model server."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from evolith.lab.kernel_replay import TARGET, paired_replay, prepare
from evolith.lab.store import write_json


def run(trial: Path, cache: Path, output: Path) -> dict:
    import torch
    import triton

    torch.cuda.set_device(0)
    torch.empty(1, device="cuda")
    torch.cuda.synchronize()
    output.mkdir(parents=True, exist_ok=False)
    rank_rows = []
    for rank in range(4):
        rows = [json.loads(line) for line in (trial / "trace" / f"rank-{rank}.jsonl").read_text().splitlines()]
        rank_rows.append(next(row for row in rows
                              if row["kind"] == "values" and row["source_sha256"] == TARGET))
    key = rank_rows[0]["kernel_cache_hash"]
    paths = list(cache.rglob(f"{key}.py"))
    if len(paths) != 1:
        raise ValueError("Cannot uniquely resolve original compiler metadata.")
    spec = importlib.util.spec_from_file_location("evolith_preserved_kernel", paths[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tuner = getattr(module, rank_rows[0]["name"])
    data = [torch.load(trial / "trace" / f"replay-input-rank-{rank}.pt",
                       weights_only=True, map_location="cpu") for rank in range(4)]
    if not all(all(torch.equal(value, peer["inputs"][name])
                   for name, value in data[0]["inputs"].items()) for peer in data[1:]):
        raise ValueError("Preserved initial RMSNorm inputs differ across ranks.")
    reports = []
    for rank, row in enumerate(rank_rows):
        directory = output / f"rank-{rank}"
        directory.mkdir()
        saved = data[rank]
        cfg = row["configuration"]
        selected = triton.Config(cfg["kwargs"], num_warps=cfg["num_warps"],
                                 num_stages=cfg["num_stages"])
        tuner.launchers = [tuner._precompile_config(selected).make_launcher()]
        names = list(tuner.triton_meta["signature"])
        bound = {name: value.cuda() for name, value in saved["inputs"].items()}
        bound.update({name: value["scalar"] for name, value in row["inputs"].items()
                      if "scalar" in value})
        output_name = next(name for name in names if name.startswith("out_ptr"))
        bound[output_name] = saved["actual"].cuda()
        args = tuple(bound[name] for name in names if name in bound)
        report = paired_replay(tuner, args, prepare(tuner, args, torch),
                               {"stream": torch.cuda.current_stream().cuda_stream},
                               torch, directory, rank)
        reports.append(report)
    result = {"status": "verified", "parent_trial_status": "failed_in_replay_plumbing",
              "scope": "same-GPU replay of preserved real tensors; not an engine correctness gate",
              "parent_trial": str(trial), "source_sha256": TARGET, "paired_replays": reports}
    write_json(output / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ("trial", "cache", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.trial, args.cache, args.output), indent=2))


if __name__ == "__main__":
    main()
