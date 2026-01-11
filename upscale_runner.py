import os
import time
import glob
import shutil
import subprocess
from typing import Iterable, Optional, Tuple, List


# ====== налаштування шляхів (підправ env у контейнері, якщо треба) ======
COMFY_ROOT = "/opt/ComfyUI"
COMFY_OUTPUT_DIR = os.environ.get("COMFY_OUTPUT_DIR") or os.path.join(COMFY_ROOT, "output")
VIDEO_OUT_DIR = os.path.join(COMFY_OUTPUT_DIR, "video")

TMP_DIR = os.environ.get("TMP_DIR") or "/tmp/comfy_worker"


# -----------------------
# helpers (re-use friendly)
# -----------------------

def ffprobe_ok(path: str) -> bool:
    p = subprocess.run(
        ["ffprobe", "-hide_banner", "-v", "error", "-show_format", "-show_streams", path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    return p.returncode == 0

def copy_to_tmp(src_path: str, out_name: str) -> str:
    os.makedirs(TMP_DIR, exist_ok=True)
    dst = os.path.join(TMP_DIR, out_name)
    shutil.copyfile(src_path, dst)
    return dst

def wait_for_stable_files(
    pattern: str,
    timeout_sec: int = 900,
    min_size: int = 100_000,
    settle_sec: float = 4.0,
    poll_sec: float = 1.0,
) -> List[str]:
    """
    Wait until at least one file exists that matches pattern and looks "stable"
    (size doesn't change over settle_sec and ffprobe passes for mp4 files).
    Returns list of matching files sorted by mtime asc (oldest->newest).
    """
    deadline = time.time() + timeout_sec
    last_seen = None

    while time.time() < deadline:
        files = [f for f in glob.glob(pattern) if os.path.isfile(f)]
        if files:
            files.sort(key=lambda p: os.path.getmtime(p))  # oldest -> newest
            last_seen = files[:]

            # Filter only sufficiently large
            big = []
            for f in files:
                try:
                    if os.path.getsize(f) >= min_size:
                        big.append(f)
                except OSError:
                    continue

            if big:
                # Ensure newest is stable
                newest = big[-1]
                try:
                    s1 = os.path.getsize(newest)
                    time.sleep(settle_sec)
                    s2 = os.path.getsize(newest)
                    if s1 == s2:
                        # If mp4, validate container
                        if newest.lower().endswith(".mp4"):
                            if ffprobe_ok(newest):
                                return big
                        else:
                            return big
                except OSError:
                    pass

        time.sleep(poll_sec)

    raise RuntimeError(f"Timed out waiting files by pattern: {pattern}. Last seen: {last_seen}")

def wait_for_video_outputs_in_comfy_id_dir(
    comfy_id: str,
    timeout_sec: int = 900,
    min_size: int = 200_000,
) -> Tuple[str, List[str]]:
    """
    comfyui-api stores outputs in:
      /opt/ComfyUI/output/{comfy_id}_video/
    Returns (out_dir, mp4_files_sorted_oldest_to_newest)
    """
    out_dir = os.path.join(COMFY_OUTPUT_DIR, f"{comfy_id}_video")
    pattern_mp4 = os.path.join(out_dir, "*.mp4")

    mp4s = wait_for_stable_files(
        pattern=pattern_mp4,
        timeout_sec=timeout_sec,
        min_size=min_size,
        settle_sec=4.0,
        poll_sec=1.0,
    )
    return out_dir, mp4s

def ffmpeg_concat_mp4s_copy(
    mp4_files: List[str],
    out_path: str,
    log,
) -> str:
    """
    Concatenate MP4 segments without re-encoding (fast).
    Assumes same codec/res/fps/pix_fmt (true in your case).
    """
    if not mp4_files:
        raise ValueError("No mp4 files to concat")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    list_txt = out_path + ".concat_list.txt"

    with open(list_txt, "w", encoding="utf-8") as f:
        for p in mp4_files:
            # ffmpeg concat demuxer requires "file 'path'"
            f.write(f"file '{p}'\n")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-fflags", "+genpts",
        "-f", "concat",
        "-safe", "0",
        "-i", list_txt,
        "-c", "copy",
        out_path,
    ]

    log(f"[ffmpeg] concat(copy): {len(mp4_files)} files -> {out_path}")
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) < 50_000:
        # Fallback: re-encode if copy concat fails due to timestamps/keyframes edge cases
        log(f"[ffmpeg] concat(copy) failed, fallback to re-encode. stderr:\n{p.stderr[-1500:]}")
        return ffmpeg_concat_mp4s_reencode(mp4_files, out_path, log=log)

    if not ffprobe_ok(out_path):
        log("[ffmpeg] concat(copy) produced invalid mp4, fallback to re-encode")
        return ffmpeg_concat_mp4s_reencode(mp4_files, out_path, log=log)

    return out_path

def ffmpeg_concat_mp4s_reencode(
    mp4_files: List[str],
    out_path: str,
    log,
    crf: int = 19,
) -> str:
    """
    Slow but robust: concatenate and re-encode to H.264.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    list_txt = out_path + ".concat_list.txt"

    with open(list_txt, "w", encoding="utf-8") as f:
        for p in mp4_files:
            f.write(f"file '{p}'\n")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_txt,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", str(crf),
        out_path,
    ]

    log(f"[ffmpeg] concat(reencode): {len(mp4_files)} files -> {out_path}")
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg re-encode concat failed. stderr:\n{p.stderr[-2000:]}")
    if not ffprobe_ok(out_path):
        raise RuntimeError("ffmpeg re-encode concat produced invalid mp4")
    return out_path


# -----------------------
# main runner for upscale task
# -----------------------

def run_upscale_and_wait_video(
    *,
    workflow_key: str,
    payload: dict,
    run_comfy_training_workflow,
    wait_timeout_sec: int = 1800,
    comfy_timeout_sec: int = 7200,
    log,
) -> Tuple[dict, str, str, List[str]]:
    """
    Runs an UPSCALE workflow via comfyui-api, then waits for MP4 outputs in
      /opt/ComfyUI/output/{comfy_id}_video/*.mp4
    If multiple MP4 segments exist, concatenates them into a single MP4.
    Copies final MP4 to TMP_DIR and returns paths.
    """
    started_at = time.time()

    result = run_comfy_training_workflow(workflow_key, payload, timeout_sec=comfy_timeout_sec)
    comfy_id = result.get("id")
    log(f"✅ Comfy задача завершена: {comfy_id}")
    if not comfy_id:
        raise RuntimeError(f"UPSCALE: comfy result has no id: {result}")

    out_dir, mp4_files = wait_for_video_outputs_in_comfy_id_dir(
        comfy_id=comfy_id,
        timeout_sec=wait_timeout_sec,
        min_size=200_000,
    )

    # Decide final mp4:
    if len(mp4_files) == 1:
        final_comfy_mp4 = mp4_files[0]
        log(f"UPSCALE: single mp4 output detected: {final_comfy_mp4}")
    else:
        # Multiple segments -> concat
        # Use a deterministic name inside comfy output dir
        final_comfy_mp4 = os.path.join(out_dir, f"{comfy_id}_merged.mp4")
        final_comfy_mp4 = ffmpeg_concat_mp4s_copy(mp4_files, final_comfy_mp4, log=log)
        log(f"UPSCALE: merged mp4: {final_comfy_mp4}")

    # Copy to TMP
    local_tmp_path = copy_to_tmp(final_comfy_mp4, f"upscale_{comfy_id[:8]}.mp4")
    return result, final_comfy_mp4, local_tmp_path, mp4_files


def handle_upscale_task(task: dict, run_comfy_training_workflow, update_task, log):
    """
    Handler in the same style as handle_wan_task, but for upscale+vfi workflows.
    - input copying into ComfyUI/input is NOT done here (as per your note).
    - waits for comfy outputs in {comfy_id}_video folder
    - if comfy produced multiple mp4 segments (common with rebatch), merges them
      into a single mp4 (concat copy, fallback re-encode).
    """
    tid = task["id"]
    workflow_key = task["workflow_key"]
    payload = task.get("payload") or {}

    log(f"[UPSCALE #{tid}] Старт через Comfy, workflow={workflow_key}")
    update_task(tid, "running", payload_update={"stage": "comfy_upscale_started"})

    try:
        result, comfy_mp4_path, local_path, segments = run_upscale_and_wait_video(
            workflow_key=workflow_key,
            payload=payload,
            run_comfy_training_workflow=run_comfy_training_workflow,
            wait_timeout_sec=1800,
            comfy_timeout_sec=7200,
            log=log,
        )

        payload_update = {
            "note": "Upscale video generated via comfyui-api (merged if segmented).",
            "comfy_id": result.get("id"),
            "stats": result.get("stats"),
            "video_comfy_output": comfy_mp4_path,
            "video_local": local_path,
            "video_segments": segments,
            "segments_count": len(segments),
        }

        update_task(tid, "done", None, payload_update)
        log(f"✅ UPSCALE-задача #{tid} завершена: {local_path}")
        return local_path

    except Exception as e:
        log(f"❌ UPSCALE-задача #{tid} помилка: {e}")
        update_task(tid, "error", str(e), payload_update={"stage": "comfy_upscale_failed"})
        raise
