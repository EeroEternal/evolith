from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path

from evolith.lab.contracts import Manifest, Proposal
from evolith.lab.store import RunStore, new_run_id, write_json

PROMPTS = [
    "Return only the answer: What is 19 + 23?",
    "Write a Python function that returns the maximum of a nonempty list.",
    "Explain in two sentences why a GPU kernel must not read uninitialized memory.",
    "请用两句话解释大模型推理中的 KV Cache。",
]
HOLDOUT = [
    "Return only the answer: What is 7 multiplied by 8?",
    "Write a Python function that reverses a string without modifying its input.",
]


def summarize(outputs, elapsed):
    records = []
    for output in outputs:
        completion = output.outputs[0]
        logprobs = []
        for token, row in zip(completion.token_ids, completion.logprobs or []):
            if row is None or token not in row or not math.isfinite(row[token].logprob):
                raise RuntimeError("Nonfinite or missing generated-token log probability.")
            logprobs.append(row[token].logprob)
        if len(logprobs) != len(completion.token_ids):
            raise RuntimeError("Missing generated-token log probabilities.")
        records.append({
            "token_ids": list(completion.token_ids), "token_logprobs": logprobs,
            "text": completion.text, "finish_reason": completion.finish_reason,
        })
    return {"elapsed_sec": elapsed, "output_tokens": sum(len(r["token_ids"]) for r in records),
            "output_tok_s": sum(len(r["token_ids"]) for r in records) / elapsed,
            "records": records}


def compare_outputs(baseline: dict, candidate: dict) -> dict:
    if len(baseline["records"]) != len(candidate["records"]):
        return {"passed": False, "reason": "request count differs"}
    max_diff = 0.0
    for a, b in zip(baseline["records"], candidate["records"]):
        if not a["token_ids"] or a["token_ids"] != b["token_ids"]:
            return {"passed": False, "reason": "greedy token sequence differs or is empty"}
        if len(a["token_logprobs"]) != len(a["token_ids"]) or len(b["token_logprobs"]) != len(b["token_ids"]):
            return {"passed": False, "reason": "missing generated-token log probabilities"}
        for x, y in zip(a["token_logprobs"], b["token_logprobs"]):
            if not math.isfinite(x) or not math.isfinite(y):
                return {"passed": False, "reason": "nonfinite generated-token log probability"}
            max_diff = max(max_diff, abs(x - y))
    return {"passed": max_diff <= 0.05, "max_generated_token_logprob_diff": max_diff,
            "scope": "greedy tokens and returned logprobs; not a complete model-quality benchmark"}


def measure_model_trials(trials: dict):
    from evolith.lab.measurement import decide

    return decide(
        [value * 1e6 for value in trials["baseline_sec"]],
        [value * 1e6 for value in trials["candidate_sec"]],
        seed=2026,
    )


def verify_model(
    model_path: Path, manifest_path: Path, output: Path, *,
    tensor_parallel: int = 4, max_tokens: int = 32, proposals: Path | None = None,
    baseline_only: bool = False,
) -> dict:
    from vllm import LLM, SamplingParams

    from evolith.lab.gpu import environment, model_info

    manifest = Manifest.model_validate_json(manifest_path.read_text())
    info = model_info(model_path)
    if info["fingerprint"] != manifest.model_fingerprint:
        raise ValueError("Model fingerprint does not match the measured artifact.")
    if not manifest.winners and not baseline_only:
        raise ValueError("No independently confirmed kernels; refusing a fake optimized-model run.")
    run_id = new_run_id()
    directory = output.resolve() / run_id
    directory.mkdir(parents=True, exist_ok=False)
    report = {
        "run_id": run_id, "status": "running", "model": info,
        "kernel_run_id": manifest.run_id, "environment": environment(),
        "tensor_parallel": tensor_parallel, "max_tokens": max_tokens,
        "engine_mode": "eager model; native RMSNorm retains its existing compiled implementation",
    }
    write_json(directory / "report.json", report)
    report["request_mode"] = "six serialized requests; single-stream decode"
    llm = None
    sampling = SamplingParams(temperature=0, max_tokens=max_tokens, logprobs=1, ignore_eos=True)

    def generate(prompts):
        started = time.perf_counter()
        outputs = []
        for prompt in prompts:
            outputs.extend(llm.generate([prompt], sampling, use_tqdm=False))
        return summarize(outputs, time.perf_counter() - started)

    try:
        llm = LLM(
            model=str(model_path.resolve()), tensor_parallel_size=tensor_parallel,
            dtype="bfloat16", max_model_len=2048, max_num_seqs=8,
            max_num_batched_tokens=2048, gpu_memory_utilization=0.90,
            enforce_eager=True, compilation_config={"mode": 0},
            trust_remote_code=False, enable_prefix_caching=False,
            limit_mm_per_prompt={"image": 0, "video": 0}, seed=2026,
            disable_log_stats=True,
            worker_extension_cls="evolith.lab.adapter.EvolithWorkerExtension",
        )
        # Warm both prompt families on the frozen baseline before collecting samples.
        generate(PROMPTS + HOLDOUT)
        baseline = generate(PROMPTS + HOLDOUT)
        report["baseline"] = baseline
        if baseline_only:
            repeated = generate(PROMPTS + HOLDOUT)
            report["baseline_repeat"] = repeated
            report["correctness"] = compare_outputs(baseline, repeated)
            report["status"] = "baseline_measured" if report["correctness"]["passed"] else "rejected"
            report["deployment_approved"] = False
            report["adapter_exercised"] = False
        else:
            report["installation"] = llm.collective_rpc("evolith_activate", args=(manifest.model_dump(),))
            generate(PROMPTS + HOLDOUT)  # compilation/warmup excluded from timing
            candidate = generate(PROMPTS + HOLDOUT)
            report["candidate"] = candidate
            report["correctness"] = compare_outputs(baseline, candidate)
            report["dispatch"] = llm.collective_rpc("evolith_stats")
            report["adapter_exercised"] = any(
                any(key.startswith("accelerated:") and count > 0 for key, count in rank.items())
                for rank in report["dispatch"]
            )
            # Fixed prompts and AB/BA order, no prefix-cache advantage.
            trials = {"baseline_sec": [], "candidate_sec": []}
            if report["correctness"]["passed"] and report["adapter_exercised"]:
                for trial in range(5):
                    modes = ("baseline", "candidate") if trial % 2 == 0 else ("candidate", "baseline")
                    for mode in modes:
                        llm.collective_rpc("evolith_restore")
                        if mode == "candidate":
                            llm.collective_rpc("evolith_activate", args=(manifest.model_dump(),))
                        measured = generate(PROMPTS + HOLDOUT)
                        comparison = compare_outputs(baseline, measured)
                        if not comparison["passed"]:
                            raise RuntimeError(f"{mode} regression during trial {trial}: {comparison}")
                        trials[f"{mode}_sec"].append(measured["elapsed_sec"])
            report["trials"] = trials
            if trials["baseline_sec"]:
                timing = measure_model_trials(trials)
                report["e2e_measurement"] = timing.model_dump()
                report["observed_paired_e2e_speedup"] = timing.speedup
                report["ratio_of_median_latencies"] = (
                    statistics.median(trials["baseline_sec"]) / statistics.median(trials["candidate_sec"])
                )
            report["status"] = (
                "verified" if report["correctness"]["passed"] and report["adapter_exercised"]
                else "rejected"
            )
            report["deployment_approved"] = bool(
                report["status"] == "verified"
                and report.get("e2e_measurement", {}).get("accepted", False)
            )
        if proposals is not None:
            llm.collective_rpc("evolith_restore")
            prompt = (
                "Choose at most four Triton zero-centered RMSNorm configurations for hidden=5120 "
                "on RTX 5090. Respond ONLY with JSON matching "
                '{"candidates":[{"num_warps":4,"rows_per_program":1}]}. '
                "num_warps must be 4,8,16; rows_per_program must be 1,2,4,8. "
                "Use diverse configurations to explore latency at small batch sizes. "
                "Do not write code or explanations. Independent confirmation winners: "
                + json.dumps({k: v.model_dump() for k, v in manifest.winners.items()})
                + ". Selected observations (speedup, lower bound, max CV): "
                + json.dumps([
                    {"config": e.config.key, "rows": e.shape.rows, "residual": e.shape.residual,
                     "stage": e.stage, "speedup": round(e.timing.speedup, 3),
                     "lower": round(e.timing.lower_bound, 3),
                     "cv": round(max(e.timing.baseline_cv, e.timing.candidate_cv), 3)}
                    for e in manifest.evaluations if e.timing
                ][-6:])
            )
            from vllm.sampling_params import StructuredOutputsParams
            proposal_sampling = SamplingParams(
                temperature=0, max_tokens=1024,
                structured_outputs=StructuredOutputsParams(json=Proposal.model_json_schema()),
            )
            prompt = llm.get_tokenizer().apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            answer = llm.generate([prompt], proposal_sampling, use_tqdm=False)[0].outputs[0].text
            proposal = Proposal.model_validate_json(answer)
            write_json(proposals, proposal.model_dump())
            report["proposal_file"] = str(proposals)
            report["proposal_source"] = "local Qwen3.5-27B inference; validated parameter JSON only"
    except Exception as exc:
        report["status"] = "failed"
        report["deployment_approved"] = False
        report["error"] = f"{type(exc).__name__}: {str(exc)[:3000]}"
        raise
    finally:
        if llm is not None:
            try:
                llm.collective_rpc("evolith_restore")
            except Exception as exc:  # noqa: BLE001 - retain reports even if worker cleanup fails
                report["cleanup_error"] = f"{type(exc).__name__}: {str(exc)[:1000]}"
                report["status"] = "failed"
                report["deployment_approved"] = False
        write_json(directory / "report.json", report)
        RunStore(output / "runs.sqlite3").record(
            run_id, "model_baseline" if baseline_only else "model_verification", report
        )
    return {"status": report["status"], "report": str(directory / "report.json"),
            "deployment_approved": report["deployment_approved"],
            "observed_paired_e2e_speedup": report.get("observed_paired_e2e_speedup"),
            "proposal_file": report.get("proposal_file")}
