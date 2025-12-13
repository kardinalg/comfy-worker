#!/usr/bin/env bash
set -e

echo "[entrypoint] starting comfyui-api..."

# comfyui-api стартує ComfyUI як дочірній процес
./comfyui-api &

# чекаємо, поки API реально підніметься
echo "[entrypoint] waiting for comfyui-api..."
for i in {1..30}; do
  if curl -s http://127.0.0.1:3000/docs >/dev/null; then
    echo "[entrypoint] comfyui-api is up"
    break
  fi
  sleep 1
done

echo "[entrypoint] starting worker..."
cd /opt/worker
python worker.py
