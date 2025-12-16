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
import hashlib
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

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
os.makedirs(TRAIN_DATA_DIR, exist_ok=True)
os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)
DOWNLOAD_FILE_URL = f"{API_BASE}/index.php?r=worker/getFile"

# ------------------ Сервісні функції ------------------

def sha256_file(path, chunk=1024*1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

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

def upload_file(task_id, path):
    files = {"file": open(path, "rb")}
    data = {"token": API_TOKEN, "task_id": task_id}
    try:
        r = requests.post(UPLOAD_FILE_URL, data=data, files=files, timeout=120)
        r.raise_for_status()
        resp = r.json()
        return None
    except Exception as e:
        log(f"Помилка аплоаду файлу {path}: {e}")
        return None
    finally:
        files["file"].close()

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

def upload_samples(task_id, samples_dir="/opt/output/sample"):
    if not os.path.isdir(samples_dir):
        print(f"[INFO] samples dir not found: {samples_dir}")
        return

    for filename in os.listdir(samples_dir):
        file_path = os.path.join(samples_dir, filename)

        if not os.path.isfile(file_path):
            continue

        try:
            upload_file(task_id, file_path)
            print(f"[OK] uploaded: {file_path}")
        except Exception as e:
            print(f"[ERROR] failed to upload {file_path}: {e}")

def upload_lora_chunked(
    file_path: str,
    lora_name: str,
    chunk_size: int = 2 * 1024 * 1024,
    max_retries: int = 8,
):
    total_size = os.path.getsize(file_path)
    file_hash = sha256_file(file_path)

    headers = {"X-Auth-Token": API_TOKEN}

    # 1) init / resume
    r = requests.post(UPLOAD_LORA_INIT, headers=headers, data={
        "lora_name": lora_name,
        "total_size": str(total_size),
        "sha256": file_hash,
    }, timeout=30)
    r.raise_for_status()
    j = r.json()
    if j.get("status") != "ok":
        raise RuntimeError(j)

    uploaded = int(j["uploaded_bytes"])
    print(f"[upload] resume from {uploaded}/{total_size}")

    # 2) upload chunks append-only
    with open(file_path, "rb") as f:
        f.seek(uploaded)
        offset = uploaded

        while offset < total_size:
            data = f.read(chunk_size)
            if not data:
                break

            # retry loop
            attempt = 0
            while True:
                try:
                    rr = requests.post(
                        UPLOAD_LORA_CHUNK,
                        headers={**headers, "Content-Type": "application/octet-stream"},
                        params={"lora_name": lora_name, "offset": str(offset)},
                        data=data,
                        timeout=120,
                    )
                    # 409 = offset mismatch -> повторно init і продовжити з correct offset
                    if rr.status_code == 409:
                        jj = rr.json()
                        offset = int(jj.get("expected_offset", offset))
                        f.seek(offset)
                        print(f"[upload] offset mismatch, jump to {offset}")
                        data = f.read(chunk_size)
                        continue

                    rr.raise_for_status()
                    jj = rr.json()
                    if jj.get("status") != "ok":
                        raise RuntimeError(jj)

                    offset = int(jj["uploaded_bytes"])
                    print(f"[upload] {offset}/{total_size}")
                    break
                except Exception as e:
                    attempt += 1
                    if attempt > max_retries:
                        raise
                    sleep = min(2 ** attempt, 30)
                    print(f"[upload] retry {attempt}/{max_retries} after {sleep}s: {e}")
                    time.sleep(sleep)

    # 3) finalize
    rf = requests.post(UPLOAD_LORA_FINAL, headers=headers, data={
        "lora_name": lora_name,
        "total_size": str(total_size),
        "sha256": file_hash,
    }, timeout=60)
    rf.raise_for_status()
    jf = rf.json()
    if jf.get("status") != "ok":
        raise RuntimeError(jf)

    print(f"[upload] DONE: {jf.get('path')} size={jf.get('size')}")
    return jf

# ------------------ Lora train
def download_training_file(name: str) -> str:
    params = {
        "token": API_TOKEN,
        "name": name,
    }
    try:
        r = requests.post(DOWNLOAD_FILE_URL, params=params, timeout=600, stream=True)
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Не вдалося завантажити файл {name}: {e}")

    local_path = os.path.join(TRAIN_DATA_DIR, name)
    # на всяк випадок — створимо піддиректорії, якщо в імені є шлях
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    with open(local_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    log(f"Файл {name} збережено в {local_path}")
    return local_path

def _download_one(session: requests.Session, name: str) -> str:
    params = {"token": API_TOKEN, "name": name}

    local_path = os.path.join(TRAIN_DATA_DIR, name)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    with session.post(DOWNLOAD_FILE_URL, params=params, timeout=600, stream=True) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1MB
                if chunk:
                    f.write(chunk)

    return local_path


def download_training_files(file_prefix, names, max_workers=16, retries=5):
    """
    Паралельне скачування з обмеженням max_workers.
    Повертає (ok_paths, failed_names).
    """
    to_download = [file_prefix + n for n in names]
    ok_paths = []
    failed = []

    # ВАЖЛИВО: Session не thread-safe, тому робимо по сесії на потік через initializer-лайт.
    # Найпростіше — створювати session всередині задачі (трохи дорожче), або тримати thread-local.
    import threading
    tls = threading.local()

    def task(full_name: str) -> str:
        if not hasattr(tls, "session"):
            tls.session = requests.Session()

        last_err = None
        for attempt in range(retries):
            try:
                return _download_one(tls.session, full_name)
            except requests.HTTPError as e:
                status = getattr(e.response, "status_code", None)
                # retry на rate limit / тимчасові серверні
                if status in (429, 500, 502, 503, 504):
                    sleep_s = min(60, (2 ** attempt) + random.random())
                    time.sleep(sleep_s)
                    last_err = e
                    continue
                raise
            except (requests.ConnectionError, requests.Timeout) as e:
                sleep_s = min(60, (2 ** attempt) + random.random())
                time.sleep(sleep_s)
                last_err = e
                continue

        raise RuntimeError(f"Failed after {retries} retries: {full_name}. Last error: {last_err}")

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(task, n): n for n in to_download}
        for fut in as_completed(futures):
            n = futures[fut]
            try:
                p = fut.result()
                ok_paths.append(p)
                # якщо хочеш прогрес:
                # log(f"OK: {n}")
            except Exception as e:
                failed.append(n)
                log(f"FAIL: {n}: {e}")

    return ok_paths, failed


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
    out_model_path = f"/opt/output/{lora_name}.safetensors_rank16_fp16.safetensors"

    # === параметри upload ===
    character_folder = payload.get("character_name") or payload.get("character_id") or lora_name

    # 1) Якщо файл вже є — НЕ тренуємо, одразу upload
    if os.path.exists(out_model_path) and os.path.getsize(out_model_path) > 10_000_000:
        log(f"[LoRA #{tid}] Файл вже існує: {out_model_path} — пропускаю тренування, роблю upload.")
        update_task(tid, "running", payload_update={"stage": "upload_existing_model"})
        upload_samples(tid)
        upload_lora_chunked(
            file_path=out_model_path,
            lora_name=lora_name,
        )
        update_task(tid, "done", None, {
            "note": "LoRA uploaded (training skipped, file existed)",
            "lora_model_local": out_model_path,
        })
        log(f"✅ LoRA-задача #{tid} завершена (skip train), upload: {up.get('path')}")
        return

    # 2) Старий шлях з zip/файлами — лишаю як optional fallback
    file_names = payload.get("files") or []
    file_prefix = payload.get("files_prefix")
    if file_names:
        log(f"[LoRA #{tid}] (fallback) Скачуємо файли: {file_names}")
        #data_paths = 
        download_training_files(file_prefix, file_names)
        #payload["data_paths"] = data_paths  # якщо comfy/workflow це читає
    else:
        log(f"[LoRA #{tid}] payload.files порожній — припускаю, що dataset вже на диску/налаштований у workflow.")

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
