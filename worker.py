#pip install requests
#python3 worker.py


import os
import time
import json
import uuid
import traceback
import requests
import base64
import subprocess
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from download_dependencies import (
    init_downloader,
    download_dependencies
)
from upload import init_uploader, upload_image, upload_file, upload_lora_chunked, upload_samples

# ------------------ Налаштування ------------------

API_BASE = os.environ["API_BASE"]
API_TOKEN = os.environ["API_TOKEN"]

GET_TASK_URL      = f"{API_BASE}/index.php?r=worker/getTask"
UPDATE_TASK_URL   = f"{API_BASE}/index.php?r=worker/updateTask"
UPLOAD_IMAGE_URL  = f"{API_BASE}/index.php?r=worker/uploadImage"
UPLOAD_FILE_URL   = f"{API_BASE}/index.php?r=worker/uploadFile"
UPLOAD_LORA_INIT  = f"{API_BASE}/index.php?r=lora/uploadLoraInit"
UPLOAD_LORA_CHUNK  = f"{API_BASE}/index.php?r=lora/uploadLoraChunk"
UPLOAD_LORA_FINAL  = f"{API_BASE}/index.php?r=lora/uploadLoraFinal"

COMFY_SERVER = "127.0.0.1:3000"            # ComfyUI на Salad-сервері
COMFY_HTTP   = f"http://{COMFY_SERVER}"

CHECK_INTERVAL = 5                         # сек. пауза між циклами

TMP_DIR = "/tmp/comfy_worker"
WORKFLOWS_DIR = "/opt/comfy_workflows"
os.makedirs(TMP_DIR, exist_ok=True)

TRAIN_DATA_DIR = "/opt/lora_train_data"
TRAIN_OUTPUT_DIR = "/opt/lora_train_output"
os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)
DOWNLOAD_FILE_URL = f"{API_BASE}/index.php?r=worker/getFile"

#COMFYUI_DIR = "/opt/ComfyUI"
#COMFYUI_LORA_DIR = os.path.join(COMFYUI_DIR, "models", "loras")
#COMFYUI_CHECKPOINTS_DIR = os.path.join(COMFYUI_DIR, "models", "checkpoints")

# ------------------ Сервісні функції ------------------

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def get_task():
    try:
        r = requests.post(GET_TASK_URL, data={"token": API_TOKEN}, timeout=15)
        r.raise_for_status()
        data = r.json()
        if not data.get("success"):
            return None
        return data.get("task")
    except Exception as e:
        log(f"Помилка запиту задачі: {e}")
        return None

def update_task(task_id, status, error=None, payload_update=None):
    payload = {
        "token": API_TOKEN,
        "id": task_id,
        "status": status,
    }
    if error:
        payload["error_message"] = error
    if payload_update is not None:
        payload["payload_update"] = json.dumps(payload_update, ensure_ascii=False)
    try:
        requests.post(UPDATE_TASK_URL, data=payload, timeout=15)
    except Exception as e:
        log(f"Не вдалося оновити статус задачі {task_id}: {e}")

def clear_dir(samples_dir="/opt/output/sample"):
    if not os.path.isdir(samples_dir):
        print(f"[INFO] samples dir not found: {samples_dir}")
        return

    for filename in os.listdir(samples_dir):
        file_path = os.path.join(samples_dir, filename)

        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
                print(f"[OK] deleted: {file_path}")
            except Exception as e:
                print(f"[ERROR] failed to delete {file_path}: {e}")


# ------------------ ComfyUI інтеграція ------------------
# API: /prompt, /history/{id}, /view?filename=...&subfolder=...&type=... :contentReference[oaicite:0]{index=0}


def build_workflow_from_payload(workflow_key: str, payload: dict) -> dict:
    path = os.path.join(WORKFLOWS_DIR, f"{workflow_key}.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Workflow template not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()

    # Проходимо по ВСІХ ключах payload
    for key, value in payload.items():
        placeholder = f"param_{key}"

        # None → null
        if value is None:
            replacement = "null"
        # числа
        elif isinstance(value, (int, float)):
            replacement = str(value)
        elif key == 'lora_nodes':
            replacement = str(value)
        else:
            s = str(value)
            dumped = json.dumps(s, ensure_ascii=False)
            replacement = dumped[1:-1]  # викидаємо зовнішні лапки

        # І тупо заміняємо в тексті ВСІ входження param_<key>
        txt = txt.replace(placeholder, replacement)

    # Тепер це вже повинен бути валідний JSON
    try:
        workflow = json.loads(txt)
    except json.JSONDecodeError as e:
        raise ValueError(f"Не вдалося розпарсити workflow після підстановки: {e}\nШматок: {txt}")

    return workflow

def queue_prompt_to_comfy(workflow: dict, client_id: str) -> str:
    """
    Відправляємо workflow в ComfyUI через /prompt.
    Повертає prompt_id.
    """
    url = f"{COMFY_HTTP}/prompt"
    payload = {
        "prompt": workflow,
        "client_id": client_id,
    }
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    prompt_id = data.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"ComfyUI не повернув prompt_id: {data}")
    return prompt_id

def run_comfy_training_workflow(workflow_key: str, payload: dict, timeout_sec: int = 7200) -> dict:
    client_id = str(uuid.uuid4())

    workflow = build_workflow_from_payload(workflow_key, payload)
    url = f"{COMFY_HTTP}/prompt"

    body = {
        "prompt": workflow,
        # Можеш задати свій id для трейсінгу:
        "id": str(uuid.uuid4()),
    }

    # timeout = (connect_timeout, read_timeout)
    r = requests.post(url, json=body, timeout=(5, timeout_sec))

    if r.status_code >= 400:
        raise RuntimeError(f"comfyui-api помилка {r.status_code}: {r.text}")

    data = r.json()
    # Для тренування тобі не обов'язково потрібні images,
    # ComfyUI workflow може просто зберегти LoRA на диск.
    return data



def run_workflow_via_comfy_api(workflow: dict, client_id: str) -> dict:
    url = f"{COMFY_HTTP}/prompt"
    payload = {
        "prompt": workflow,
        "client_id": client_id,
    }
    r = requests.post(url, json=payload, timeout=(5, 600))

    if r.status_code >= 400:
        # тут побачимо справжню причину 500
        raise RuntimeError(
            f"comfyui-api помилка {r.status_code}: {r.text}"
        )

    data = r.json()
    if "images" not in data:
        raise RuntimeError(f"Несподіваний формат відповіді comfyui-api: {data}")
    return data

def generate_with_comfy(workflow_key: str, payload: dict) -> str:
    """
    Повний цикл:
      1) побудувати workflow_json
      2) /prompt -> prompt_id
      3) чекати /history/prompt_id
      4) забрати перше зображення через /view
      5) повернути локальний шлях до PNG
    """
    client_id = str(uuid.uuid4())

    # 1) будуємо workflow з payload
    workflow = build_workflow_from_payload(workflow_key, payload)

    # 2) запускаємо workflow через comfyui-api
    result = run_workflow_via_comfy_api(workflow, client_id)
    task_id = result.get("id")
    log(f"comfyui-api task_id={task_id}")

    images = result.get("images") or []
    if not images:
        raise RuntimeError(f"comfyui-api не повернув images: {result}")

    # 3) беремо перше зображення
    first = images[0]

    # comfyui-api може повернути або чистий base64-рядок,
    # або dict з полями типу {"image": "...", "filename": "..."}
    if isinstance(first, dict):
        b64_data = first.get("image") or first.get("data")
        filename = first.get("filename") or f"{task_id}.png"
    else:
        b64_data = first
        filename = f"{task_id}.png"

    if not b64_data:
        raise RuntimeError(f"Немає base64 даних у images[0]: {first}")

    # 4) зберігаємо в TMP_DIR
    ext = os.path.splitext(filename)[1] or ".png"
    safe_id = (task_id or "comfy")[:8]
    tmp_name = f"comfy_{safe_id}{ext}"
    local_path = os.path.join(TMP_DIR, tmp_name)

    os.makedirs(TMP_DIR, exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(base64.b64decode(b64_data))

    log(f"Зображення збережено локально: {local_path}")
    return local_path


# ------------------ Головний цикл ------------------
def wait_for_file(path: str, timeout_sec: int = 300, min_size: int = 10_000_000):
    """Чекає появи файлу і щоб він був не пустий/не битий (min_size)."""
    t0 = time.time()
    while time.time() - t0 < timeout_sec:
        if os.path.exists(path):
            try:
                if os.path.getsize(path) >= min_size:
                    return True
            except OSError:
                pass
        time.sleep(2)
    return False

def handle_lora_train_task(task):
    tid = task["id"]
    workflow_key = task["workflow_key"]
    payload = task.get("payload") or {}

    lora_name = payload.get("lora_name")
    if not lora_name:
        raise RuntimeError("payload.lora_name обовʼязковий")

    # === де comfy зберігає модель локально ===
    # Рекомендую НЕ /opt/output, а volume, але лишаю як ти написав
    out_model_path = f"/opt/output/{tid}.safetensors_rank16_fp16.safetensors"

    # === параметри upload ===
    character_folder = payload.get("character_name") or payload.get("character_id") or lora_name

    # 1) Якщо файл вже є — НЕ тренуємо, одразу upload
    if os.path.exists(out_model_path) and os.path.getsize(out_model_path) > 10_000_000:
        log(f"[LoRA #{tid}] Файл вже існує: {out_model_path} — пропускаю тренування, роблю upload.")
        update_task(tid, "running", payload_update={"stage": "upload_existing_model"})
        upload_samples(tid)
        up = upload_lora_chunked(
            file_path=out_model_path,
            lora_name=lora_name,
        )
        update_task(tid, "done", None, {
            "note": "LoRA uploaded (training skipped, file existed)",
            "lora_model_local": out_model_path,
        })
        log(f"✅ LoRA-задача #{tid} завершена (skip train), upload: {up.get('path')}")
        return

    clear_dir()

    # 3) Запускаємо comfy training
    log(f"[LoRA #{tid}] Старт тренування через Comfy, workflow={workflow_key}")
    update_task(tid, "running", payload_update={"stage": "comfy_training_started"})

    result = run_comfy_training_workflow(workflow_key, payload, timeout_sec=7200)

    log(f"[LoRA #{tid}] Тренування завершене шукаю файл {out_model_path}")
    # 4) Чекаємо щоб файл реально зʼявився
    if not wait_for_file(out_model_path, timeout_sec=600, min_size=1_000_000):
        raise RuntimeError(f"[LoRA #{tid}] Comfy завершився, але файл не знайдено/замалий: {out_model_path}")

    log(f"[LoRA #{tid}] Файл знайдено")
    # 5) Upload
    update_task(tid, "running", payload_update={"stage": "upload_trained_model", "comfy_id": result.get("id")})
    
    upload_samples(tid)
    log(f"[LoRA #{tid}] Samples завантажено")
    up = upload_lora_chunked(
        file_path=out_model_path,
        lora_name=lora_name,
    )
    log(f"[LoRA #{tid}] Файл завантажено")

    payload_update = {
        "note": "LoRA training done via comfyui-api and uploaded",
        "comfy_id": result.get("id"),
        "stats": result.get("stats"),
        "lora_model_local": out_model_path,
        "lora_model_remote": up.get("path"),
        "remote_size": up.get("size"),
    }

    update_task(tid, "done", None, payload_update)
    log(f"✅ LoRA-задача #{tid} завершена, модель: {out_model_path} → {up.get('path')}")

def main():
    init_downloader(
        api_token=API_TOKEN,
        download_file_url=DOWNLOAD_FILE_URL,
        train_data_dir=TRAIN_DATA_DIR,
        log_fn=log,
    )
    init_uploader(API_TOKEN, UPLOAD_FILE_URL, UPLOAD_IMAGE_URL, log)
    log("Воркер запущено. Очікуємо задачі...")
    while True:
        task = get_task()
        if not task:
            time.sleep(CHECK_INTERVAL)
            continue

        tid = task["id"]
        ttype = task["type"]
        workflow_key = task["workflow_key"]
        payload = task["payload"] or {}
        task["payload"]["task_id"] = tid

        try:
            log(f"Отримано задачу #{tid} [{ttype}] workflow={workflow_key}")
            download_dependencies(task.get("dependency") or [])

            # приклад: type == 'lora_image' або 'frame_image' — все одно, ми просто шлемо в Comfy
            if ttype in ("lora_image", "frame_image", "other", "lora_test"):
                local_path = generate_with_comfy(workflow_key, payload)
                remote_path = upload_image(tid, local_path)
                if remote_path:
                    update_task(tid, "done", None, {"result_path": remote_path})
                    log(f"✅ Завершено задачу #{tid}, result={remote_path}")
                else:
                    update_task(tid, "failed", "Upload failed")
            elif ttype == "lora_train":
                # 🔥 новий тип задачі
                handle_lora_train_task(task)
            else:
                update_task(tid, "failed", f"Невідомий тип задачі: {ttype}")

        except NotImplementedError as e:
            # ти ще не реалізував build_workflow_from_payload
            log(f"❌ build_workflow_from_payload не реалізований: {e}")
            update_task(tid, "failed", "Workflow builder not implemented")
        except Exception as e:
            err = traceback.format_exc()
            log(f"❌ Помилка задачі #{tid}: {e}")
            update_task(tid, "failed", err)

        time.sleep(1)


if __name__ == "__main__":
    main()
