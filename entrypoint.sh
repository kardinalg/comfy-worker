#!/usr/bin/env bash
set -e

WORK_DIR=${WORK_DIR:-/opt/worker}
WORKER_REPO=${WORKER_REPO:-"https://github.com/kardinalg/comfy-worker.git"}
WORKER_BRANCH=${WORKER_BRANCH:-"main"}

MODEL_DIR=${MODEL_DIR:-/opt/ComfyUI/models}
CHECKPOINT_DIR="${MODEL_DIR}/checkpoints"


# якщо є persistent volume
if [ -d "/workspace" ]; then
  echo "[entrypoint] /workspace detected, using persistent volume"

  mkdir -p /workspace/ComfyUI/{models,output,workflows}

  # models
  if [ ! -L "/opt/ComfyUI/models" ]; then
    rm -rf /opt/ComfyUI/models
    ln -s /workspace/ComfyUI/models /opt/ComfyUI/models
  fi

  # output
  if [ ! -L "/opt/ComfyUI/output" ]; then
    rm -rf /opt/ComfyUI/output
    ln -s /workspace/ComfyUI/output /opt/ComfyUI/output
  fi

  # workflows
  if [ ! -L "/opt/ComfyUI/workflows" ]; then
    rm -rf /opt/ComfyUI/workflows
    ln -s /workspace/ComfyUI/workflows /opt/ComfyUI/workflows
  fi

else
  echo "[entrypoint] /workspace not found, using container filesystem"
fi




mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "${WORK_DIR}"

echo "[entrypoint] fetching worker code from ${WORKER_REPO} (${WORKER_BRANCH})..."

if [ ! -d "${WORK_DIR}/.git" ]; then
  git clone --branch "${WORKER_BRANCH}" "${WORKER_REPO}" "${WORK_DIR}"
else
  cd "${WORK_DIR}"
  git fetch origin "${WORKER_BRANCH}"
  git checkout -f "${WORKER_BRANCH}"
  git pull origin "${WORKER_BRANCH}"
fi

# (опційно) встановити залежності для воркера
if [ -f "${WORK_DIR}/requirements.txt" ]; then
  echo "[entrypoint] installing worker dependencies..."
  pip install --no-cache-dir -r "${WORK_DIR}/requirements.txt"
fi

WORKFLOWS_SRC="${WORK_DIR}/workflows"
WORKFLOWS_DST="/opt/comfy_workflows"

echo "[entrypoint] syncing workflows..."

if [ -d "${WORKFLOWS_SRC}" ]; then
  # якщо ще немає /opt/comfy_workflows – створюємо симлінк
  if [ ! -e "${WORKFLOWS_DST}" ]; then
    ln -s "${WORKFLOWS_SRC}" "${WORKFLOWS_DST}"
    echo "[entrypoint] linked ${WORKFLOWS_SRC} -> ${WORKFLOWS_DST}"
  else
    echo "[entrypoint] ${WORKFLOWS_DST} already exists, skipping link"
  fi
else
  echo "[entrypoint] WARNING: workflows dir ${WORKFLOWS_SRC} not found"
fi

echo "[entrypoint] fetching models..."
chmod +x /opt/worker/fetch_models.sh 2>/dev/null || true
/opt/worker/fetch_models.sh

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

if [ ! -d "/opt/ComfyUI/custom_nodes/ComfyUI-FluxTrainer/.git" ]; then
  pip uninstall -y opencv-python-headless
  pip install "opencv-python-headless<4.12"

  cd /opt/ComfyUI/custom_nodes
  git clone https://github.com/kijai/ComfyUI-FluxTrainer.git

  # Залежності
  cd ComfyUI-FluxTrainer
  pip install -r requirements.txt


  cd /opt/ComfyUI/custom_nodes
  git clone https://github.com/kijai/ComfyUI-KJNodes.git
  # іноді треба:
  cd ComfyUI-KJNodes
  pip install -r requirements.txt || true


  cd /opt/ComfyUI/custom_nodes
  git clone https://github.com/pythongosssss/ComfyUI-WD14-Tagger.git
  cd ComfyUI-WD14-Tagger
  pip install -r requirements.txt

  cd /opt/ComfyUI/custom_nodes
  git clone https://github.com/rgthree/rgthree-comfy.git

  cd /opt/ComfyUI/custom_nodes
  git clone https://github.com/yolain/ComfyUI-Easy-Use.git
  cd ComfyUI-Easy-Use
  pip install -r requirements.txt || true

  cd /opt/ComfyUI/custom_nodes
  git clone https://github.com/whitmell/ComfyUI-RvTools.git
fi

mkdir -p /opt/output

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


