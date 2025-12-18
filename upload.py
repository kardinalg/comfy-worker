import os
import time
import hashlib
import requests

_API_TOKEN = None
_UPLOAD_FILE_URL = None
_UPLOAD_IMAGE_URL = None
_LOG = None

def init_uploader(api_token: str, upload_file_url: str, upload_image_url: str, log_fn):
    """
    Викликати один раз при старті воркера (в main.py).
    """
    global _API_TOKEN, _UPLOAD_FILE_URL, _UPLOAD_IMAGE_URL, _LOG
    _API_TOKEN = api_token
    _UPLOAD_FILE_URL = upload_file_url
    _UPLOAD_IMAGE_URL = upload_image_url
    _LOG = log_fn

def sha256_file(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def upload_file(task_id: int, path: str):
    files = {"file": open(path, "rb")}
    data = {"token": _API_TOKEN, "task_id": task_id}
    try:
        r = requests.post(_UPLOAD_FILE_URL, data=data, files=files, timeout=120)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        _LOG(f"Помилка аплоаду файлу {path}: {e}")
        return None
    finally:
        files["file"].close()


def upload_image(task_id: int, path: str):
    files = {"file": open(path, "rb")}
    data = {"token": _API_TOKEN, "task_id": task_id}
    try:
        r = requests.post(_UPLOAD_IMAGE_URL, data=data, files=files, timeout=120)
        r.raise_for_status()
        resp = r.json()
        return resp.get("result_path")
    except Exception as e:
        _LOG(f"Помилка аплоаду image {path}: {e}")
        return None
    finally:
        files["file"].close()


def upload_lora_chunked(
    upload_lora_init_url: str,
    upload_lora_chunk_url: str,
    upload_lora_final_url: str,
    file_path: str,
    lora_name: str,
    chunk_size: int = 2 * 1024 * 1024,
    max_retries: int = 8,
):
    total_size = os.path.getsize(file_path)
    file_hash = sha256_file(file_path)

    headers = {"X-Auth-Token": _API_TOKEN}

    r = requests.post(upload_lora_init_url, headers=headers, data={
        "lora_name": lora_name,
        "total_size": str(total_size),
        "sha256": file_hash,
    }, timeout=30)
    r.raise_for_status()
    j = r.json()
    if j.get("status") != "ok":
        raise RuntimeError(j)

    uploaded = int(j["uploaded_bytes"])
    _LOG(f"[upload] resume from {uploaded}/{total_size}")

    with open(file_path, "rb") as f:
        f.seek(uploaded)
        offset = uploaded

        while offset < total_size:
            data = f.read(chunk_size)
            if not data:
                break

            attempt = 0
            while True:
                try:
                    rr = requests.post(
                        upload_lora_chunk_url,
                        headers={**headers, "Content-Type": "application/octet-stream"},
                        params={"lora_name": lora_name, "offset": str(offset)},
                        data=data,
                        timeout=120,
                    )

                    if rr.status_code == 409:
                        jj = rr.json()
                        offset = int(jj.get("expected_offset", offset))
                        f.seek(offset)
                        _LOG(f"[upload] offset mismatch, jump to {offset}")
                        data = f.read(chunk_size)
                        continue

                    rr.raise_for_status()
                    jj = rr.json()
                    if jj.get("status") != "ok":
                        raise RuntimeError(jj)

                    offset = int(jj["uploaded_bytes"])
                    _LOG(f"[upload] {offset}/{total_size}")
                    break

                except Exception as e:
                    attempt += 1
                    if attempt > max_retries:
                        raise
                    sleep = min(2 ** attempt, 30)
                    _LOG(f"[upload] retry {attempt}/{max_retries} after {sleep}s: {e}")
                    time.sleep(sleep)

    rf = requests.post(upload_lora_final_url, headers=headers, data={
        "lora_name": lora_name,
        "total_size": str(total_size),
        "sha256": file_hash,
    }, timeout=60)
    rf.raise_for_status()
    jf = rf.json()
    if jf.get("status") != "ok":
        raise RuntimeError(jf)

    _LOG(f"[upload] DONE: {jf.get('path')} size={jf.get('size')}")
    return jf

def upload_samples(task_id, samples_dir="/opt/output/sample"):
    if not os.path.isdir(samples_dir):
        _LOG(f"[INFO] samples dir not found: {samples_dir}")
        return

    for filename in os.listdir(samples_dir):
        file_path = os.path.join(samples_dir, filename)

        if not os.path.isfile(file_path):
            continue

        try:
            upload_file(task_id, file_path)
            _LOG(f"[OK] uploaded: {file_path}")
        except Exception as e:
            _LOG(f"[ERROR] failed to upload {file_path}: {e}")
