from __future__ import annotations

import json
import math
import statistics
import time

import httpx

from evolith.lab.contracts import content_hash
from evolith.lab.engine import MODEL_NAME
from evolith.lab.measurement import decide

# Fixed public fixtures, not model-generated goals. Two requests are holdouts.
CHAT_PROMPTS = [
    "Explain in about 100 words why GPU memory bandwidth matters for language model decoding.",
    "Write a Python function that checks whether a string has balanced parentheses. Explain its complexity.",
    "请用约一百字说明大模型推理中的 KV Cache 有什么作用，并举一个聊天场景的例子。",
    "Describe the steps needed to benchmark a GPU inference server fairly, in a numbered list.",
    "Write a Python function that merges two sorted lists without modifying either input.",
    "Explain the difference between time to first token and inter-token latency, with an example.",
]


def workload_fingerprint(max_tokens: int) -> str:
    return content_hash({"prompts": CHAT_PROMPTS, "max_tokens": max_tokens,
                         "context_limit": 2048, "concurrency": 1,
                         "ignore_eos": True, "thinking": False})


def stream_request(client, prompt: str, max_tokens: int, clock=time.perf_counter) -> dict:
    payload = {
        "model": MODEL_NAME, "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "max_tokens": max_tokens, "ignore_eos": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True, "stream_options": {"include_usage": True},
        "logprobs": True, "top_logprobs": 0,
        "return_token_ids": True, "return_tokens_as_token_ids": True,
    }
    started = clock()
    arrivals, tokens, probabilities = [], [], []
    usage = None
    done = False
    with client.stream("POST", "/v1/chat/completions", json=payload) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            if line == "data: [DONE]":
                done = True
                break
            data = json.loads(line[6:])
            if "error" in data:
                raise RuntimeError("Engine returned a streaming error.")
            if data.get("usage"):
                usage = data["usage"]
            for choice in data.get("choices", []):
                rows = (choice.get("logprobs") or {}).get("content") or []
                ids = choice.get("token_ids")
                if ids is None:
                    ids = []
                    for row in rows:
                        token = row["token"]
                        if not token.startswith("token_id:"):
                            raise ValueError("Missing unambiguous streamed token IDs.")
                        ids.append(int(token.removeprefix("token_id:")))
                if len(ids) != len(rows):
                    raise ValueError("Streamed token IDs/logprobs disagree.")
                now = clock()
                for token, row in zip(ids, rows):
                    probability = row["logprob"]
                    if (not isinstance(token, int) or isinstance(token, bool)
                            or not math.isfinite(probability)):
                        raise ValueError("Invalid generated-token evidence.")
                    tokens.append(token)
                    probabilities.append(probability)
                    arrivals.append(now)
    if (not done or len(tokens) != max_tokens or usage is None
            or usage["completion_tokens"] != len(tokens)):
        raise ValueError("Incomplete generation; latency cannot be compared.")
    ttft = arrivals[0] - started
    itl = (arrivals[-1] - arrivals[0]) / (len(tokens) - 1)
    if not all(math.isfinite(v) and v > 0 for v in (ttft, itl)):
        raise ValueError("Invalid latency measurements.")
    return {"token_ids": tokens, "token_logprobs": probabilities,
            "ttft_sec": ttft, "itl_sec": itl, "elapsed_sec": clock() - started}


def outputs_match(first: list[dict], second: list[dict]) -> bool:
    return len(first) == len(second) and bool(first) and all(
        a["token_ids"] == b["token_ids"] and bool(a["token_ids"])
        and len(a["token_logprobs"]) == len(b["token_logprobs"]) == len(a["token_ids"])
        and all(math.isfinite(x) and math.isfinite(y) and abs(x - y) <= 0.05
                for x, y in zip(a["token_logprobs"], b["token_logprobs"]))
        for a, b in zip(first, second)
    )


def compare_trials(baseline: dict, candidate: dict) -> dict:
    if baseline["workload_fingerprint"] != candidate["workload_fingerprint"]:
        raise ValueError("Different workloads cannot be compared.")
    correct = outputs_match(baseline["records"], candidate["records"])
    ttft = decide(baseline["ttft_us"], candidate["ttft_us"], seed=2031)
    itl = decide(baseline["itl_us"], candidate["itl_us"], seed=2031)
    stable = max(ttft.baseline_cv, ttft.candidate_cv, itl.baseline_cv, itl.candidate_cv) <= 0.10
    # A fixed Pareto gate: one latency improves >3%, the other does not regress.
    accepted = correct and stable and (
        (ttft.accepted and itl.lower_bound >= 1.0)
        or (itl.accepted and ttft.lower_bound >= 1.0)
    )
    return {"correct": correct, "accepted": accepted, "ttft": ttft.model_dump(),
            "itl": itl.model_dump(), "stable": stable,
            "rule": "99% lower bound >1.03 for one latency, >=1.0 for the other; CV<=10%; output match"}


def median_latencies(records: list[dict]) -> tuple[float, float]:
    return (statistics.median(r["ttft_sec"] for r in records) * 1e6,
            statistics.median(r["itl_sec"] for r in records) * 1e6)


def local_client(port: int) -> httpx.Client:
    return httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=300,
                        trust_env=False, follow_redirects=False)
