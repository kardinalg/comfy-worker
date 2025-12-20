# wan_runner.py
import os
import time
import glob
import shutil
from typing import Optional, Iterable, Tuple


# ====== налаштування шляхів (підправ env у контейнері, якщо треба) ======
COMFY_ROOT = "/opt/ComfyUI"
COMFY_OUTPUT_DIR = os.environ.get("COMFY_OUTPUT_DIR") or os.path.join(COMFY_ROOT, "output")
VIDEO_OUT_DIR = os.path.join(COMFY_OUTPUT_DIR, "video")

TMP_DIR = os.environ.get("TMP_DIR") or "/tmp/comfy_worker"


# ====== helpers ======
def wait_for_new_file_by_patterns(
    patterns: Iterable[str],
    started_at: float,
    timeout_sec: int = 900,
    min_size: int = 1_000_000,
    poll_sec: float = 1.0,
    mtime_slack_sec: float = 2.0,
) -> Optional[str]:
    """
    Чекає, поки зʼявиться новий файл (mtime >= started_at - slack), який підходить під patterns (glob).
    Повертає шлях до найсвіжішого валідного файлу або None.
    """
    deadline = time.time() + timeout_sec
    os.makedirs(VIDEO_OUT_DIR, exist_ok=True)

    while time.time() < deadline:
        candidates = []
        for pat in patterns:
            candidates.extend(glob.glob(pat))

        candidates = [p for p in candidates if os.path.isfile(p)]
        candidates = [p for p in candidates if os.path.getmtime(p) >= started_at - mtime_slack_sec]

        if candidates:
            # найновіший перший
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)

            for p in candidates:
                try:
                    if os.path.getsize(p) >= min_size:
                        return p
                except OSError:
                    continue

        time.sleep(poll_sec)

    return None

def wait_for_newest_file(pattern: str, timeout_sec: int = 600, min_size: int = 100_000) -> str:
    deadline = time.time() + timeout_sec
    best = None
    while time.time() < deadline:
        files = glob.glob(pattern)
        files = [f for f in files if os.path.isfile(f)]
        if files:
            files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            for f in files:
                if os.path.getsize(f) >= min_size:
                    return f
            best = files[0]
        time.sleep(1)
    raise RuntimeError(f"File not found by pattern: {pattern}. Last seen: {best}")


def copy_to_tmp(src_path: str, out_name: str) -> str:
    os.makedirs(TMP_DIR, exist_ok=True)
    dst = os.path.join(TMP_DIR, out_name)
    shutil.copyfile(src_path, dst)
    return dst

def pick_wan_video_from_comfy_result(result: dict) -> str:
    comfy_id = result.get("id")
    if not comfy_id:
        raise RuntimeError(f"comfy result has no id: {result}")

    # comfyui-api створює {id}_video
    out_dir = os.path.join(COMFY_OUTPUT_DIR, f"{comfy_id}_video")
    pattern = os.path.join(out_dir, "*.mp4")
    video_path = wait_for_newest_file(pattern, timeout_sec=900, min_size=1_000_000)
    return video_path

def wait_for_wan_video_output(
    *,
    comfy_id: str,
    started_at: float,
    timeout_sec: int = 900,
    min_size: int = 1_000_000,
) -> str:
    """
    comfyui-api зберігає відео в:
      /opt/ComfyUI/output/{comfy_id}_video/*.mp4
    (це ти вже побачив руками).
    """
    out_dir = os.path.join(COMFY_OUTPUT_DIR, f"{comfy_id}_video")

    patterns = [
        os.path.join(out_dir, "*.mp4"),
        os.path.join(out_dir, "*.webm"),
        os.path.join(out_dir, "*.mov"),
        os.path.join(out_dir, "*.mkv"),
        os.path.join(out_dir, "*.gif"),
    ]

    p = wait_for_new_file_by_patterns(
        patterns=patterns,
        started_at=started_at,
        timeout_sec=timeout_sec,
        min_size=min_size,
        poll_sec=1.0,
    )
    if not p:
        raise RuntimeError(
            f"WAN: video output not found in {out_dir}. Expected patterns: {patterns}"
        )
    return p



# ====== main runner ======
def run_wan_and_wait_video(
    *,
    workflow_key: str,
    payload: dict,
    run_comfy_training_workflow,
    wait_timeout_sec: int = 900,
    comfy_timeout_sec: int = 3600,
) -> Tuple[dict, str, str]:
    """
    Запускає WAN workflow,
    потім чекає появи відео в ComfyUI/output/{comfy_id}_video/,
    копіює його в TMP_DIR.
    """
    started_at = time.time()

    result = run_comfy_training_workflow(workflow_key, payload, timeout_sec=comfy_timeout_sec)

    comfy_id = result.get("id")
    if not comfy_id:
        raise RuntimeError(f"WAN: comfy result has no id: {result}")

    comfy_video_path = wait_for_wan_video_output(
        comfy_id=comfy_id,
        started_at=started_at,
        timeout_sec=wait_timeout_sec,
        min_size=1_000_000,
    )

    ext = os.path.splitext(comfy_video_path)[1] or ".mp4"
    local_tmp_path = copy_to_tmp(comfy_video_path, f"wan_{comfy_id[:8]}{ext}")

    return result, comfy_video_path, local_tmp_path


def handle_wan_task(task: dict, run_comfy_training_workflow, update_task, log):
    """
    Handler в стилі твоїх задач.
    input (копіювання в ComfyUI/input) тут НЕ робимо — ти сказав, що вже зроблено.
    Очікуємо, що payload вже містить param_input_image і т.п.
    """
    tid = task["id"]
    workflow_key = task["workflow_key"]
    payload = task.get("payload") or {}

    log(f"[WAN #{tid}] Старт через Comfy, workflow={workflow_key}")
    update_task(tid, "running", payload_update={"stage": "comfy_wan_started"})

    result, comfy_video_path, local_path = run_wan_and_wait_video(
        workflow_key=workflow_key,
        payload=payload,
        run_comfy_training_workflow=run_comfy_training_workflow,
        wait_timeout_sec=900,
        comfy_timeout_sec=3600
    )

    payload_update = {
        "note": "WAN video generated via comfyui-api (file waited in output)",
        "comfy_id": result.get("id"),
        "stats": result.get("stats"),
        "video_comfy_output": comfy_video_path,
        "video_local": local_path,
    }

    update_task(tid, "done", None, payload_update)
    log(f"✅ WAN-задача #{tid} завершена: {local_path}")
    return local_path
