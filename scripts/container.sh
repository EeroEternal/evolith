#!/usr/bin/env bash
set -euo pipefail

# Run only on the GPU machine. No host networking, host IPC, privileged mode,
# writable source/model mount, or modification of an existing service.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="${EVOLITH_IMAGE:-vllm/vllm-openai:v0.19.0}"
model="${EVOLITH_MODEL:-/data/models/Qwen3.5-27B}"
devices="${EVOLITH_GPUS:-2}"
name="evolith-lab-$$"
mkdir -p "$root/results" "$root/cache"
cleanup() { docker stop --time 10 "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

command=(-m evolith.cli "$@")
if [[ "${1:-}" == test ]]; then
    shift
    command=(-m pytest -p no:cacheprovider "$@")
elif [[ "${1:-}" == python ]]; then
    shift
    command=("$@")
fi
timeout --signal=TERM --kill-after=30s "${EVOLITH_TIMEOUT:-1800}" \
    docker run --rm --init --pull=never --name "$name" --label org.evolith.lab=true \
    --network none --gpus "\"device=$devices\"" \
    --cap-drop ALL --security-opt no-new-privileges \
    --pids-limit 1024 --memory "${EVOLITH_MEMORY:-48g}" --cpus "${EVOLITH_CPUS:-12}" \
    --shm-size 4g --user "$(id -u):$(id -g)" --workdir /tmp \
    -v "$root:/workspace:ro" -v "$root/results:/results" -v "$root/cache:/cache" \
    -v "$model:/models/qwen:ro" \
    -e PYTHONPATH=/cache/testdeps:/workspace/src -e PYTHONDONTWRITEBYTECODE=1 \
    -e HOME=/tmp -e USER=evolith -e LOGNAME=evolith \
    -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
    -e TORCHINDUCTOR_COMPILE_THREADS=2 \
    -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
    -e TRITON_CACHE_DIR=/cache/triton -e HF_HOME=/cache/huggingface \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e VLLM_HOST_IP=127.0.0.1 -e NCCL_SOCKET_IFNAME=lo -e GLOO_SOCKET_IFNAME=lo \
    --entrypoint /usr/bin/python3 "$image" "${command[@]}"
