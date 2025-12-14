#pip install requests
#python3 worker.py


import os
import time
import json
import uuid
import traceback
import requests
import base64
from datetime import datetime

# ------------------ Налаштування ------------------

API_BASE = os.environ["API_BASE"]
API_TOKEN = os.environ["API_TOKEN"]

GET_TASK_URL      = f"{API_BASE}/index.php?r=worker/getTask"
UPDATE_TASK_URL   = f"{API_BASE}/index.php?r=worker/updateTask"
UPLOAD_IMAGE_URL  = f"{API_BASE}/index.php?r=worker/uploadImage"

COMFY_SERVER = "127.0.0.1:3000"            # ComfyUI на Salad-сервері
COMFY_HTTP   = f"http://{COMFY_SERVER}"

CHECK_INTERVAL = 5                         # сек. пауза між циклами

TMP_DIR = "/tmp/comfy_worker"
WORKFLOWS_DIR = "/opt/comfy_workflows"
os.makedirs(TMP_DIR, exist_ok=True)

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


def upload_image(task_id, path):
    files = {"file": open(path, "rb")}
    data = {"token": API_TOKEN, "task_id": task_id}
    try:
        r = requests.post(UPLOAD_IMAGE_URL, data=data, files=files, timeout=120)
        r.raise_for_status()
        resp = r.json()
        return resp.get("result_path")
    except Exception as e:
        log(f"Помилка аплоаду файлу {path}: {e}")
        return None
    finally:
        files["file"].close()


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
        raise ValueError(f"Не вдалося розпарсити workflow після підстановки: {e}\nШматок: {txt[:300]}")

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

def run_workflow_via_comfy_api(workflow: dict, client_id: str) -> dict:
    """
    Відправляємо workflow в comfyui-api (/prompt) і відразу
    отримуємо результат у форматі comfyui-api:
      {
        "id": "...",
        "prompt": {...},
        "images": [ ... base64 ... ]
      }
    """
    url = f"{COMFY_HTTP}/prompt"
    payload = {
        "prompt": workflow,
        "client_id": client_id,
    }
    r = requests.post(url, json=payload, timeout=(5, 600))
    r.raise_for_status()
    data = r.json()

    # Перевіримо, що це саме comfyui-api формат
    if "images" not in data:
        raise RuntimeError(f"Несподіваний формат відповіді comfyui-api: {data}")

    return data


def wait_for_result(prompt_id: str, timeout_sec: int = 600) -> dict:
    """
    Полінг /history/{prompt_id}, поки не буде результату.
    Повертає JSON history.
    """
    url = f"{COMFY_HTTP}/history/{prompt_id}"
    start = time.time()
    while True:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            data = r.json()
            # Структура: { prompt_id: { "outputs": {...} } }
            if prompt_id in data and "outputs" in data[prompt_id]:
                return data[prompt_id]
        # чек
        if time.time() - start > timeout_sec:
            raise TimeoutError(f"ComfyUI не завершив задачу за {timeout_sec} сек.")
        time.sleep(2)


def extract_first_image_info(history: dict) -> dict:
    """
    Витягуємо перший output image: filename, subfolder, type.
    """
    outputs = history.get("outputs", {})
    for node_id, node_out in outputs.items():
        images = node_out.get("images") or []
        if not images:
            continue
        img = images[0]
        return {
            "filename": img.get("filename"),
            "subfolder": img.get("subfolder") or "",
            "type": img.get("type") or "output",
        }
    raise RuntimeError("Не знайдено жодного зображення в history")


def download_image_from_comfy(info: dict, local_path: str):
    """
    /view?filename=...&subfolder=...&type=...
    """
    params = {
        "filename": info["filename"],
        "subfolder": info["subfolder"],
        "type": info["type"],  # input/output/temp
    }
    url = f"{COMFY_HTTP}/view"
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    with open(local_path, "wb") as f:
        f.write(r.content)


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

    # 2) відправляємо в ComfyUI
    # prompt_id = queue_prompt_to_comfy(workflow, client_id)
    # log(f"ComfyUI prompt_id={prompt_id}")

    # # 3) чекаємо завершення
    # history = wait_for_result(prompt_id)

    # # 4) беремо перше зображення
    # img_info = extract_first_image_info(history)
    # log(f"Отримано файл з ComfyUI: {img_info}")

    # # 5) качаємо в tmp
    # ext = os.path.splitext(img_info["filename"])[1] or ".png"
    # tmp_name = f"comfy_{prompt_id[:8]}{ext}"
    # local_path = os.path.join(TMP_DIR, tmp_name)
    # download_image_from_comfy(img_info, local_path)

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


def main():
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

        try:
            log(f"Отримано задачу #{tid} [{ttype}] workflow={workflow_key}")

            # приклад: type == 'lora_image' або 'frame_image' — все одно, ми просто шлемо в Comfy
            if ttype in ("lora_image", "frame_image", "other"):
                local_path = generate_with_comfy(workflow_key, payload)
                remote_path = upload_image(tid, local_path)
                if remote_path:
                    update_task(tid, "done", None, {"result_path": remote_path})
                    log(f"✅ Завершено задачу #{tid}, result={remote_path}")
                else:
                    update_task(tid, "failed", "Upload failed")
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
