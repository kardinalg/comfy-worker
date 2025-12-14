#!/usr/bin/env bash
set -e

# ========= CONFIG =========
MODEL_DIR="/opt/ComfyUI/models/checkpoints"
MODEL_ID="119226"                     # Copax Cute XL
MODEL_NAME="copax_cute_xl.safetensors"
CIVITAI_API_KEY="${CIVITAI_API_KEY:-}"

# ========= CHECKS =========
mkdir -p "$MODEL_DIR"

if [ -f "$MODEL_DIR/$MODEL_NAME" ]; then
  echo "[models] $MODEL_NAME already exists, skipping download"
  exit 0
fi

if [ -z "$CIVITAI_API_KEY" ]; then
  echo "[models][ERROR] CIVITAI_API_KEY is not set"
  exit 1
fi

# ========= DOWNLOAD =========
echo "[models] Downloading $MODEL_NAME from Civitai..."

curl -fL \
  -H "Authorization: Bearer $CIVITAI_API_KEY" \
  -o "$MODEL_DIR/$MODEL_NAME" \
  "https://civitai.com/api/download/models/$MODEL_ID"

# ========= VALIDATION =========
FILE_TYPE=$(file "$MODEL_DIR/$MODEL_NAME")

if echo "$FILE_TYPE" | grep -qi "html"; then
  echo "[models][ERROR] Downloaded file is HTML (invalid token or blocked)"
  rm -f "$MODEL_DIR/$MODEL_NAME"
  exit 1
fi

echo "[models] Download complete: $MODEL_NAME"


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