#!/usr/bin/env bash
set -e

WORK_DIR=${WORK_DIR:-/opt/worker}
WORKER_REPO=${WORKER_REPO:-"https://github.com/kardinalg/comfy-worker.git"}
WORKER_BRANCH=${WORKER_BRANCH:-"main"}

MODEL_DIR=${MODEL_DIR:-/opt/ComfyUI/models}
CHECKPOINT_DIR="${MODEL_DIR}/checkpoints"

mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "${WORK_DIR}"

echo "[entrypoint] fetching worker code from ${WORKER_REPO} (${WORKER_BRANCH})..."

if [ ! -d "${WORK_DIR}/.git" ]; then
  git clone --branch "${WORKER_BRANCH}" "${WORKER_REPO}" "${WORK_DIR}"
else
  cd "${WORK_DIR}"
  git fetch origin "${WORKER_BRANCH}"
  git checkout "${WORKER_BRANCH}"
  git pull origin "${WORKER_BRANCH}"
fi

# (опційно) встановити залежності для воркера
if [ -f "${WORK_DIR}/requirements.txt" ]; then
  echo "[entrypoint] installing worker dependencies..."
  pip install --no-cache-dir -r "${WORK_DIR}/requirements.txt"
fi

download_sdxl() {
  echo "[entrypoint] checking SDXL models in ${CHECKPOINT_DIR}"

  if [ ! -f "${CHECKPOINT_DIR}/sd_xl_base_1.0.safetensors" ]; then
    echo "[entrypoint] downloading sd_xl_base_1.0.safetensors..."
    curl -L \
      -o "${CHECKPOINT_DIR}/sd_xl_base_1.0.safetensors" \
      "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors"
  else
    echo "[entrypoint] sd_xl_base_1.0.safetensors already exists"
  fi

  if [ ! -f "${CHECKPOINT_DIR}/sd_xl_refiner_1.0.safetensors" ]; then
    echo "[entrypoint] downloading sd_xl_refiner_1.0.safetensors..."
    curl -L \
      -o "${CHECKPOINT_DIR}/sd_xl_refiner_1.0.safetensors" \
      "https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors"
  else
    echo "[entrypoint] sd_xl_refiner_1.0.safetensors already exists"
  fi
}

download_sdxl

echo "[entrypoint] locating comfyui-api..."

# спочатку шукаємо в PATH (якщо базовий образ поклав його кудись типу /usr/local/bin)
if command -v comfyui-api >/dev/null 2>&1; then
  COMFY_API_BIN="$(command -v comfyui-api)"
elif [ -x "/comfyui-api" ]; then
  # деякі образи кладуть його прямо в /comfyui-api
  COMFY_API_BIN="/comfyui-api"
else
  echo "[entrypoint] ERROR: comfyui-api binary not found in PATH or at /comfyui-api" >&2
  ls -la /
  echo "[entrypoint] Current PATH: $PATH" >&2
  exit 1
fi

echo "[entrypoint] starting comfyui-api at ${COMFY_API_BIN}..."
"${COMFY_API_BIN}" &

echo "[entrypoint] waiting for comfyui-api..."
for i in {1..30}; do
  if curl -s http://127.0.0.1:3000/docs >/dev/null; then
    echo "[entrypoint] comfyui-api is up"
    break
  fi
  sleep 1
done

echo "[entrypoint] starting worker..."
cd "${WORK_DIR}"
python worker.py


