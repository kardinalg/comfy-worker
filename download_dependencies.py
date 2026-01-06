import os
import time
import random
import threading
from pathlib import Path
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# ---- ComfyUI dirs (статичні) ----
COMFYUI_DIR = "/opt/ComfyUI"
COMFYUI_LORA_DIR = os.path.join(COMFYUI_DIR, "models", "loras")
COMFYUI_CHECKPOINTS_DIR = os.path.join(COMFYUI_DIR, "models", "checkpoints")
COMFYUI_VAE_DIR = os.path.join(COMFYUI_DIR, "models", "vae")
COMFYUI_TEXT_ENCODERS_DIR = os.path.join(COMFYUI_DIR, "models", "text_encoders")
COMFYUI_DIFFUSION_MODELS_DIR = os.path.join(COMFYUI_DIR, "models", "diffusion_models")
COMFYUI_UPSCALE = os.path.join(COMFYUI_DIR, "models", "upscale")
COMFYUI_INPUT_DIR = os.path.join(COMFYUI_DIR, "input")

# ---- runtime config (ініціалізується з main) ----
_API_TOKEN = None
_DOWNLOAD_FILE_URL = None
_TRAIN_DATA_DIR = None
_LOG = None

def init_downloader(api_token: str, download_file_url: str, train_data_dir: str, log_fn):
    """
    Викликати один раз при старті воркера (в main.py).
    """
    global _API_TOKEN, _DOWNLOAD_FILE_URL, _TRAIN_DATA_DIR, _LOG
    _API_TOKEN = api_token
    _DOWNLOAD_FILE_URL = download_file_url
    _TRAIN_DATA_DIR = train_data_dir
    _LOG = log_fn

    os.makedirs(COMFYUI_LORA_DIR, exist_ok=True)
    os.makedirs(COMFYUI_CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(_TRAIN_DATA_DIR, exist_ok=True)

# ------------------ helpers ------------------

def _require_init():
    if not _API_TOKEN or not _DOWNLOAD_FILE_URL or not _TRAIN_DATA_DIR or not _LOG:
        raise RuntimeError("Downloader not initialized. Call init_downloader(...) in main.py")


def _filename_from_url(url: str, fallback: str = "download.bin") -> str:
    name = Path(urlparse(url).path).name
    return name if name else fallback


def _get_target_dir(dep_type: str) -> str:
    t = (dep_type or "").lower()
    if t == "loras":
        return COMFYUI_LORA_DIR
    if t == "checkpoints":
        return COMFYUI_CHECKPOINTS_DIR
    if t == "vae":
        return COMFYUI_VAE_DIR
    if t == "text_encoders":
        return COMFYUI_TEXT_ENCODERS_DIR
    if t == "diffusion_models":
        return COMFYUI_DIFFUSION_MODELS_DIR
    if t == "upscale":
        return COMFYUI_UPSCALE
    if t == "input":
        return COMFYUI_INPUT_DIR
    raise ValueError(f"Unknown dependency type: {dep_type}")

def safe_basename(name: str) -> str:
    base = os.path.basename(name.replace("\\", "/"))
    if base in ("", ".", ".."):
        raise ValueError(f"Некоректне ім'я файлу: {name!r}")
    return base

def download_simple(url: str, target_dir: str, file_name: str, *, min_size: int = 1_024) -> str:
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(target_dir) / file_name

    # ✅ skip if already downloaded
    if out_path.exists():
        try:
            if out_path.stat().st_size >= min_size:
                return str(out_path)
        except OSError:
            pass  # якщо не можемо прочитати — спробуємо перекачати

    tmp_path = out_path.with_suffix(out_path.suffix + ".part")

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    # атомарно замінюємо
    os.replace(tmp_path, out_path)
    return str(out_path)


def download_civitai(url: str, target_dir: str, file_name: str, *, min_size: int = 1_024) -> str:
    api_key = os.environ.get("CIVITAI_API_KEY")
    if not api_key:
        raise RuntimeError("CIVITAI_API_KEY is not set")

    Path(target_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(target_dir) / file_name

    # ✅ skip if already downloaded
    if out_path.exists():
        try:
            if out_path.stat().st_size >= min_size:
                return str(out_path)
        except OSError:
            pass

    tmp_path = out_path.with_suffix(out_path.suffix + ".part")

    headers = {"Authorization": f"Bearer {api_key}"}

    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    os.replace(tmp_path, out_path)
    return str(out_path)


# ------------------ KG7 бекенд downloads (через ваш API /getFile) ------------------
def _download_one(session: requests.Session, name: str, dep_type: str) -> str:
    _require_init()

    params = {"token": _API_TOKEN, "name": name}
    local_path = os.path.join(dep_type, name)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    with session.post(_DOWNLOAD_FILE_URL, params=params, timeout=600, stream=True) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    return local_path


def download_files(file_prefix, names, dep_type, max_workers=16, retries=5):
    """
    Паралельне скачування. Повертає (ok_paths, failed_names).
    """
    _require_init()

    to_download = [file_prefix + n for n in names]
    ok_paths, failed = [], []

    tls = threading.local()

    def worker(full_name: str) -> str:
        if not hasattr(tls, "session"):
            tls.session = requests.Session()

        last_err = None
        for attempt in range(retries):
            try:
                return _download_one(tls.session, full_name, dep_type)
            except requests.HTTPError as e:
                status = getattr(e.response, "status_code", None)
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
        futures = {ex.submit(worker, n): n for n in to_download}
        for fut in as_completed(futures):
            n = futures[fut]
            try:
                p = fut.result()
                ok_paths.append(p)
            except Exception as e:
                failed.append(n)
                _LOG(f"FAIL: {n}: {e}")

    return ok_paths, failed


def download_lora_file(lora_name: str) -> str:
    """
    KG7-LoRA download через /getFile, зберігаємо в ComfyUI/models/loras
    """
    _require_init()

    filename = safe_basename(lora_name)
    local_path = os.path.join(COMFYUI_LORA_DIR, filename)

    if os.path.isfile(local_path):
        _LOG(f"LoRA {lora_name} вже існує: {local_path}")
        return local_path

    params = {"token": _API_TOKEN, "name": lora_name}

    try:
        r = requests.post(_DOWNLOAD_FILE_URL, params=params, timeout=600, stream=True)
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Не вдалося завантажити LoRA {lora_name}: {e}")

    with open(local_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    _LOG(f"LoRA {lora_name} збережено в {local_path}")
    return local_path


def download_lora_files(loras) -> str:
    _require_init()
    if isinstance(loras, list) and loras:
        for lora_name in loras:
            download_lora_file(lora_name)
    return "downloaded"


# ------------------ main dependency router ------------------

def download_dependencies(dependency: list):
    """
    dependency: list[dict] з url/url_type/type/files
    """
    _require_init()

    dependency = dependency or []
    for dep in dependency:
        if not isinstance(dep, dict):
            continue

        url = dep.get("url")
        url_type = (dep.get("url_type") or "simple").lower()
        dep_type = dep.get("type")  # loras / checkpoints
        files = dep.get("files") or []
        file_name = dep.get("file_name")

        if not url or not dep_type:
            continue

        _LOG(f"Залежність: file_name={file_name} type={dep_type} url_type={url_type} url={url}")

        target_dir = _get_target_dir(dep_type)

        if url_type == "simple":
            download_simple(url, target_dir, file_name)

        elif url_type == "civitai":
            download_civitai(url, target_dir, file_name)

        elif url_type == "kg7-lora":
            download_lora_file(url)

        elif url_type == "kg7-file":
            download_files(url, files, dep_type)

        else:
            raise ValueError(f"Unknown url_type: {url_type}")