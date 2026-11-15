#!/usr/bin/env bash
set -euo pipefail

# Keep the model isolated; a CPU-only byte relay listens on host loopback.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
profile="$(realpath "${1:?Usage: bash scripts/serve.sh /absolute/path/to/engine.json}")"
name="${EVOLITH_SERVICE:-evolith-qwen}"
port="${EVOLITH_PORT:-18080}"
network="$name-internal"
relay="$name-loopback"
image="${EVOLITH_IMAGE:-vllm/vllm-openai:v0.19.0}"
model="${EVOLITH_MODEL:-/data/models/Qwen3.5-27B}"
[[ "$name" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || exit 2
[[ "$port" =~ ^[0-9]+$ ]] && ((10#$port >= 1024 && 10#$port <= 65535)) || exit 2
if docker container inspect "$name" >/dev/null 2>&1; then
    printf 'Refusing to replace existing container: %s\n' "$name" >&2
    exit 2
fi
if docker container inspect "$relay" >/dev/null 2>&1; then
    printf 'Refusing to replace existing container: %s\n' "$relay" >&2
    exit 2
fi
count="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["config"]["tensor_parallel"])' "$profile")"
[[ "$count" == 2 || "$count" == 4 ]] || exit 2
IFS=, read -r -a available <<< "${EVOLITH_GPUS:-0,1,2,3}"
((${#available[@]} >= count)) || exit 2
selected=("${available[@]:0:count}")
devices="$(IFS=,; printf '%s' "${selected[*]}")"
mkdir -p "$root/cache" "$root/results"
verification="$(mktemp -d "$root/results/prepublication-XXXXXXXX")"
created=0
launched=0
model_id=""
relay_id=""
cleanup() {
    if [[ "$created" == 1 && "$launched" == 0 ]]; then
        [[ -z "$relay_id" ]] || docker rm -f "$relay_id" >/dev/null 2>&1 || true
        [[ -z "$model_id" ]] || docker rm -f "$model_id" >/dev/null 2>&1 || true
        docker network rm "$network" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
docker network create --internal --label org.evolith.service=true "$network"
created=1
model_id="$(docker run --detach --init --pull=never --name "$name" --label org.evolith.service=true \
    --network "$network" --gpus "\"device=$devices\"" \
    --cap-drop ALL --security-opt no-new-privileges \
    --pids-limit 1024 --memory "${EVOLITH_MEMORY:-192g}" --cpus "${EVOLITH_CPUS:-24}" \
    --shm-size 4g --user "$(id -u):$(id -g)" --workdir /tmp \
    -v "$root:/workspace:ro" -v "$root/cache:/cache" \
    -v "$verification:/validation" \
    -v "$profile:/engine-profile.json:ro" -v "$model:/models/qwen:ro" \
    -e PYTHONPATH=/workspace/src -e PYTHONDONTWRITEBYTECODE=1 \
    -e EVOLITH_LOOPBACK_PUBLISH=1 \
    -e HOME=/tmp -e USER=evolith -e LOGNAME=evolith \
    -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
    -e TORCHINDUCTOR_COMPILE_THREADS=2 -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
    -e TRITON_CACHE_DIR=/cache/triton -e HF_HOME=/cache/huggingface \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e VLLM_HOST_IP=127.0.0.1 -e NCCL_SOCKET_IFNAME=lo -e GLOO_SOCKET_IFNAME=lo \
    --entrypoint /usr/bin/python3 "$image" -m evolith.cli serve-engine \
    --model-path /models/qwen --profile /engine-profile.json --port 8000 --container-loopback-publish)"
timeout --signal=TERM --kill-after=15s 690 docker exec "$model_id" /usr/bin/python3 \
    -m evolith.lab.prepublication --profile /engine-profile.json \
    --output /validation/report.json --port 8000
target="$(docker inspect --format "{{(index .NetworkSettings.Networks \"$network\").IPAddress}}" "$model_id")"
python3 -c 'import ipaddress,sys; a=ipaddress.ip_address(sys.argv[1]); assert a.version==4 and a.is_private and not a.is_loopback and not a.is_unspecified and not a.is_link_local' "$target"
relay_id="$(docker run --detach --init --pull=never --name "$relay" --label org.evolith.service=true \
    --network host --cap-drop ALL --security-opt no-new-privileges --read-only \
    --pids-limit 64 --memory 256m --cpus 1 --user "$(id -u):$(id -g)" \
    -v "$root/src:/workspace/src:ro" -v "$profile:/engine-profile.json:ro" \
    -e PYTHONPATH=/workspace/src -e PYTHONDONTWRITEBYTECODE=1 \
    -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
    --entrypoint /usr/bin/python3 "$image" -m evolith.lab.loopback_proxy --port "$port" --target "$target")"
launched=1
printf 'Verified %s at http://127.0.0.1:%s/v1; startup evidence: %s/report.json\n' \
    "$name" "$port" "$verification"
