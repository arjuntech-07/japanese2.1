# main.py
"""
Pure-RAM Manga / Manhwa / Manhua Translation Server & Modern Mobile Web UI
- Zero Disk Touch for Images: Uploads -> RAM -> GPU VRAM -> RAM -> Mobile Download -> Auto-Wipe
- OCR Engine: Qwen 3.5 9B Uncensored Q8_0 + BF16 mmproj (Thinking OFF)
- Translation Engine: Qwen 3.8 27B Uncensored OrcaRouter Q6_K + MTP Companion (Context-Aware)
- Fast Binary Streaming: Lightweight JSON polling + Dedicated Binary ZIP & Single-Page Endpoints
"""
import os
import sys
import io
import gc
import re
import time
import uuid
import json
import zipfile
import queue
import threading
import subprocess
import logging
import cv2
import torch
import numpy as np
import requests
from flask import Flask, request, jsonify, send_file, render_template_string

# Silence noisy third-party loggers while keeping our pipeline logs crisp
logging.getLogger("werkzeug").setLevel(logging.ERROR)

# ─────────────────────────────────────────────────────────────
# 0. EXPAND /dev/shm RAM DISK (PURE SYSTEM RAM — NO KAGGLE DISK)
# ─────────────────────────────────────────────────────────────
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.system("mount -o remount,size=14G /dev/shm 2>/dev/null || true")

RAM_BASE = "/dev/shm/manga_ram"
os.makedirs(RAM_BASE, exist_ok=True)
os.environ["TMPDIR"] = RAM_BASE

# ─────────────────────────────────────────────────────────────
# IMPORTS FROM PROJECT MODULES
# ─────────────────────────────────────────────────────────────
from crop_engine import run_yolo_ram_batch
from manga_ocr_utils import (
    ensure_ocr_model_downloaded,
    start_ocr_server,
    stop_ocr_server,
    encode_crop_to_base64_ram,
    strip_thinking,
    LLAMA_CHAT_URL as OCR_CHAT_URL,
    OCR_PROMPT
)
from translate_manga import (
    ensure_translation_models_downloaded,
    start_translation_server,
    stop_translation_server,
    translate_chunk
)
from cleaner import process_contour
from text_renderer import add_text

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
FLASK_PORT = 5000
YOLO_MODEL_PATH = os.environ.get(
    "YOLO_MODEL_PATH",
    "/kaggle/working/best.pt"
)
DISCORD_WEBHOOK_URL = os.environ.get(
    "DISCORD_WEBHOOK_URL",
    "https://discord.com/api/webhooks/1537848230539300905/9JN_TypsOffB8kblNdYv3e_e49_h898ph6LWSwOuccMogAogafNx5H_eIS68yKw55Y6E"
)

# ─────────────────────────────────────────────────────────────
# IN-MEMORY JOB QUEUE & RAM STORE (NEVER TOUCHES DISK)
# ─────────────────────────────────────────────────────────────
JOB_QUEUE = queue.Queue()
JOBS_RAM = {}  # { job_id: { status, phase, progress, logs, pages_in_ram, out_pages_bytes, zip_bytes, ... } }
JOBS_LOCK = threading.Lock()


def log_job(job_id: str, msg: str):
    """Prints live to Kaggle Notebook console AND updates Mobile UI status."""
    print(msg, flush=True)
    with JOBS_LOCK:
        if job_id in JOBS_RAM:
            JOBS_RAM[job_id]["last_log"] = msg.strip()
            JOBS_RAM[job_id]["logs"].append(msg.strip())
            if len(JOBS_RAM[job_id]["logs"]) > 40:
                JOBS_RAM[job_id]["logs"] = JOBS_RAM[job_id]["logs"][-40:]


def update_job_state(job_id: str, **kwargs):
    with JOBS_LOCK:
        if job_id in JOBS_RAM:
            JOBS_RAM[job_id].update(kwargs)


def free_job_ram(job_id: str):
    """Wipes all input/output images and ZIP buffers from Kaggle RAM."""
    with JOBS_LOCK:
        job = JOBS_RAM.pop(job_id, None)
        if job:
            job.clear()
    gc.collect()
    print(f"🧹 [RAM Purged] Job {job_id} completely wiped from Kaggle RAM.", flush=True)


# ─────────────────────────────────────────────────────────────
# 4-PHASE SEQUENTIAL PURE-RAM PIPELINE WORKER (UNTOUCHED PIPELINE)
# ─────────────────────────────────────────────────────────────
def process_manga_job_in_ram(job_id: str):
    with JOBS_LOCK:
        job = JOBS_RAM.get(job_id)
        if not job:
            return
        ram_pages = job["pages_in_ram"]  # List of (filename, numpy_bgr_img)
        rtl_mode = job["rtl"]
        enable_thinking = job["thinking"]
        chapter_name = job["chapter_name"]

    total_pages = len(ram_pages)
    start_ts = time.time()

    try:
        log_job(job_id, "\n" + "=" * 65)
        log_job(job_id, f"🚀 STARTING JOB [{chapter_name}] | {total_pages} Pages | {'Manga (RTL)' if rtl_mode else 'Manhwa (LTR)'}")
        log_job(job_id, "=" * 65)

        # =========================================================================
        # PHASE 1: YOLOv8 SPEECH BUBBLE DETECTION & CROPPING IN RAM
        # =========================================================================
        update_job_state(
            job_id,
            status="processing",
            phase="1/4: YOLOv8 Bubble Detection",
            progress=5
        )
        log_job(job_id, f"🔍 [Phase 1/4] Running YOLOv8 on {total_pages} pages in RAM...")

        yolo_path = YOLO_MODEL_PATH if os.path.exists(YOLO_MODEL_PATH) else "best.pt"
        page_map = run_yolo_ram_batch(ram_pages, model_path=yolo_path, rtl=rtl_mode)

        # Gather all cropped bubbles across all pages in sequential order
        all_bubbles_ordered = []
        for filename, _ in ram_pages:
            for b in page_map.get(filename, []):
                all_bubbles_ordered.append((filename, b))

        total_bubbles = len(all_bubbles_ordered)
        log_job(job_id, f"✅ [Phase 1/4] Detected {total_bubbles} speech bubbles across {total_pages} pages.")

        # =========================================================================
        # PHASE 2: VISION OCR EXTRACTION (QWEN 3.5 9B — THINKING OFF)
        # =========================================================================
        ocr_results = {}
        if total_bubbles > 0:
            update_job_state(
                job_id,
                phase=f"2/4: Qwen3.5-9B Vision OCR (0/{total_bubbles})",
                progress=15
            )
            log_job(job_id, f"\n👁️ [Phase 2/4] Loading Qwen 3.5 9B Vision OCR (Thinking OFF)...")

            if not start_ocr_server():
                raise RuntimeError("Failed to start Qwen 3.5 9B OCR llama-server.")

            for idx, (page_fname, b_info) in enumerate(all_bubbles_ordered):
                crop_id = b_info["crop_id"]
                b64_png = encode_crop_to_base64_ram(b_info["crop_img"])

                if not b64_png:
                    ocr_results[crop_id] = ""
                    continue

                payload = {
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_png}"}},
                            {"type": "text", "text": OCR_PROMPT}
                        ]
                    }],
                    "temperature": 0.0,
                    "top_p": 0.9,
                    "max_tokens": 256,
                    "chat_template_kwargs": {"enable_thinking": False}
                }

                try:
                    resp = requests.post(OCR_CHAT_URL, json=payload, timeout=90)
                    resp.raise_for_status()
                    raw_txt = resp.json()["choices"][0]["message"].get("content", "")
                    clean_txt = strip_thinking(raw_txt)
                    ocr_results[crop_id] = clean_txt
                except Exception as e:
                    log_job(job_id, f"  ⚠️ OCR Error on {crop_id}: {e}")
                    ocr_results[crop_id] = ""

                pct = 15 + int(((idx + 1) / total_bubbles) * 30)
                update_job_state(
                    job_id,
                    phase=f"2/4: Qwen3.5-9B Vision OCR ({idx + 1}/{total_bubbles})",
                    progress=pct
                )
                if (idx + 1) % 5 == 0 or (idx + 1) == total_bubbles:
                    snippet = ocr_results[crop_id].replace("\n", " ")[:22]
                    log_job(job_id, f"  📖 OCR [{idx + 1}/{total_bubbles}] {crop_id}: {snippet}")

            log_job(job_id, "🧹 Unloading Qwen 3.5 9B OCR from VRAM...")
            stop_ocr_server()

        # =========================================================================
        # PHASE 3: BULK CONTEXT-AWARE TRANSLATION (QWEN 3.8 27B + MTP)
        # =========================================================================
        translations_map = {}
        valid_crop_ids = []
        valid_texts = []

        for _, b_info in all_bubbles_ordered:
            cid = b_info["crop_id"]
            txt = ocr_results.get(cid, "").strip()
            if txt:
                valid_crop_ids.append(cid)
                valid_texts.append(txt)

        if valid_texts:
            update_job_state(
                job_id,
                phase=f"3/4: Loading Qwen3.8-27B + MTP Translator...",
                progress=48
            )
            log_job(job_id, f"\n🌐 [Phase 3/4] Starting Qwen 3.8 27B + MTP Speculative Translator ({len(valid_texts)} bubbles)...")

            if not start_translation_server():
                raise RuntimeError("Failed to start Qwen 3.8 27B translation llama-server.")

            CHUNK_SIZE = 10
            rolling_context = []
            total_chunks = (len(valid_texts) + CHUNK_SIZE - 1) // CHUNK_SIZE

            for c_idx, i in enumerate(range(0, len(valid_texts), CHUNK_SIZE)):
                chunk_texts = valid_texts[i: i + CHUNK_SIZE]
                chunk_cids = valid_crop_ids[i: i + CHUNK_SIZE]

                if rolling_context:
                    ctx_header = " | ".join(rolling_context[-4:])
                    chunk_with_ctx = list(chunk_texts)
                    chunk_with_ctx[0] = f"(Story context from previous bubbles: {ctx_header}) {chunk_with_ctx[0]}"
                else:
                    chunk_with_ctx = chunk_texts

                chunk_res = translate_chunk(chunk_with_ctx, enable_thinking=enable_thinking)

                for local_idx, cid in enumerate(chunk_cids):
                    orig_line = chunk_texts[local_idx].replace("\n", " ")
                    en_line = chunk_res.get(str(local_idx), "").strip()
                    translations_map[cid] = en_line

                    if en_line:
                        rolling_context.append(f"{orig_line} -> {en_line}")

                    log_job(job_id, f"   [ORIG] {orig_line}")
                    log_job(job_id, f"   [ENGL] {en_line if en_line else '[Skipped]'}")

                pct = 50 + int(((c_idx + 1) / total_chunks) * 30)
                update_job_state(
                    job_id,
                    phase=f"3/4: Qwen3.8-27B Translating Chunk ({c_idx + 1}/{total_chunks})",
                    progress=pct
                )

            log_job(job_id, "🧹 Unloading Qwen 3.8 27B + MTP from VRAM...")
            stop_translation_server()

        # =========================================================================
        # PHASE 4: TWO-PASS INPAINT CLEANING & TYPESETTING IN PURE RAM
        # =========================================================================
        update_job_state(
            job_id,
            phase=f"4/4: Cleaning & Typesetting (0/{total_pages} pages)",
            progress=82
        )
        log_job(job_id, f"\n🎨 [Phase 4/4] Cleaning Speech Bubbles & Rendering English Text in RAM...")

        output_pages = []
        zip_buffer = io.BytesIO()

        # Use ZIP_STORED or fast level=1 because PNGs are already compressed! (Saves CPU & sends faster)
        with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_STORED) as zf:
            for p_idx, (filename, full_img) in enumerate(ram_pages):
                bubbles = page_map.get(filename, [])

                # ── PASS 4A: Clean original text on full_img using ONLY the text mask ──
                for b_info in bubbles:
                    cid = b_info["crop_id"]
                    orig_txt = ocr_results.get(cid, "").strip()
                    english_text = translations_map.get(cid, "").strip()

                    if orig_txt in ("", "ー", "-", "—", "…", ".", "・") and len(english_text) <= 2:
                        b_info["skip_render"] = True
                        b_info["crop_img"] = None
                        continue

                    b_info["skip_render"] = False
                    x1, y1, x2, y2 = b_info["box"]
                    live_crop = full_img[y1:y2, x1:x2].copy()
                    if live_crop.size == 0:
                        continue

                    cleaned_crop, text_mask = process_contour(live_crop)

                    mask_bool = text_mask > 0
                    full_img[y1:y2, x1:x2][mask_bool] = cleaned_crop[mask_bool]
                    b_info["crop_img"] = None

                # ── PASS 4B: Render English text directly onto the cleaned full_img ──
                for b_info in bubbles:
                    if b_info.get("skip_render", False):
                        continue
                    cid = b_info["crop_id"]
                    english_text = translations_map.get(cid, "").strip()
                    if not english_text:
                        continue

                    x1, y1, x2, y2 = b_info["box"]
                    target_region = full_img[y1:y2, x1:x2].copy()
                    add_text(target_region, english_text)
                    full_img[y1:y2, x1:x2] = target_region

                out_fname = f"translated_{filename}"
                if not out_fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    out_fname += ".png"

                # Fast PNG encoding in RAM
                _, png_buf = cv2.imencode(".png", full_img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                png_bytes = png_buf.tobytes()

                zf.writestr(out_fname, png_bytes)

                # Store raw binary bytes in RAM (NOT giant Base64 strings inside JSON!)
                output_pages.append({
                    "filename": out_fname,
                    "bytes": png_bytes
                })

                pct = 82 + int(((p_idx + 1) / total_pages) * 18)
                update_job_state(
                    job_id,
                    phase=f"4/4: Cleaning & Typesetting ({p_idx + 1}/{total_pages} pages)",
                    progress=min(99, pct)
                )
                log_job(job_id, f"  ✨ Assembled in RAM [{p_idx + 1}/{total_pages}]: {out_fname}")

        zip_buffer.seek(0)
        elapsed = int(time.time() - start_ts)

        with JOBS_LOCK:
            if job_id in JOBS_RAM:
                JOBS_RAM[job_id]["pages_in_ram"] = []
                JOBS_RAM[job_id]["out_pages"] = output_pages
                JOBS_RAM[job_id]["zip_bytes"] = zip_buffer.getvalue()
                JOBS_RAM[job_id]["status"] = "done"
                JOBS_RAM[job_id]["elapsed"] = elapsed
                JOBS_RAM[job_id]["total_bubbles"] = total_bubbles
                JOBS_RAM[job_id]["phase"] = f"✅ Completed {total_pages} pages ({total_bubbles} bubbles) in {elapsed}s!"
                JOBS_RAM[job_id]["progress"] = 100

        gc.collect()
        log_job(job_id, f"\n🎉 JOB COMPLETE [{chapter_name}] in {elapsed}s! Ready for Mobile Download from RAM.")

    except Exception as e:
        log_job(job_id, f"❌ JOB FAILED [{chapter_name}]: {e}")
        update_job_state(
            job_id,
            status="error",
            phase=f"❌ Error: {e}"
        )
        stop_ocr_server()
        stop_translation_server()
        gc.collect()


def background_queue_daemon():
    """Continuously pulls queued jobs one by one so back-to-back folder uploads never collide."""
    while True:
        job_id = JOB_QUEUE.get()
        try:
            process_manga_job_in_ram(job_id)
        except Exception as e:
            print(f"⚠️ Worker exception on {job_id}: {e}", flush=True)
        finally:
            JOB_QUEUE.task_done()


# ─────────────────────────────────────────────────────────────
# MODERN GLASSMORPHISM MOBILE WEB UI (3 SEPARATE VIEWS + FAST STREAMING)
# ─────────────────────────────────────────────────────────────
HTML_UI = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>MangaAI Studio • Pure RAM Translator</title>
<style>
  :root {
    --bg-main: #07090e;
    --bg-card: rgba(17, 24, 39, 0.85);
    --bg-inner: rgba(11, 15, 25, 0.9);
    --border-subtle: rgba(255, 255, 255, 0.08);
    --border-glow: rgba(139, 92, 246, 0.45);
    --accent-primary: #8b5cf6;
    --accent-secondary: #06b6d4;
    --accent-success: #10b981;
    --accent-warn: #f59e0b;
    --accent-danger: #ef4444;
    --text-main: #f8fafc;
    --text-muted: #94a3b8;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; -webkit-tap-highlight-color: transparent; }
  body {
    background: radial-gradient(circle at 50% 0%, #1e1b4b 0%, #07090e 55%);
    background-attachment: fixed;
    color: var(--text-main);
    min-height: 100vh;
    padding: 14px 12px 40px;
    display: flex;
    flex-direction: column;
    align-items: center;
  }
  .container { width: 100%; max-width: 640px; }

  /* Header */
  .app-header { text-align: center; padding: 10px 0 16px; }
  .brand-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(139, 92, 246, 0.15); border: 1px solid var(--border-glow);
    color: #c4b5fd; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.6px;
    padding: 4px 12px; border-radius: 99px; margin-bottom: 8px; text-transform: uppercase;
  }
  .brand-dot { width: 7px; height: 7px; background: var(--accent-success); border-radius: 50%; box-shadow: 0 0 8px var(--accent-success); }
  h1 {
    font-size: 1.45rem; font-weight: 800; letter-spacing: -0.5px;
    background: linear-gradient(90deg, #ffffff, #a5b4fc, #67e8f9);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
  }
  .sub { font-size: 0.76rem; color: var(--text-muted); }

  /* Navigation Tabs (Separate Views!) */
  .nav-tabs {
    display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 6px;
    background: rgba(15, 23, 42, 0.9); padding: 6px; border-radius: 14px;
    border: 1px solid var(--border-subtle); margin-bottom: 14px;
    backdrop-filter: blur(12px); position: sticky; top: 8px; z-index: 50;
  }
  .tab-btn {
    background: transparent; border: none; color: var(--text-muted);
    padding: 10px 6px; border-radius: 10px; font-size: 0.78rem; font-weight: 700;
    cursor: pointer; transition: all 0.2s ease; display: flex; align-items: center; justify-content: center; gap: 5px;
  }
  .tab-btn.active {
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.28), rgba(6, 182, 212, 0.22));
    color: #fff; border: 1px solid var(--border-glow);
    box-shadow: 0 4px 14px rgba(139, 92, 246, 0.2);
  }
  .tab-count {
    background: rgba(255,255,255,0.12); color: #fff; padding: 1px 6px;
    border-radius: 99px; font-size: 0.68rem; font-weight: 800;
  }
  .tab-count.pulse { background: var(--accent-secondary); color: #000; }
  .tab-count.done-badge { background: var(--accent-success); color: #000; }

  .tab-panel { display: none; animation: fadeIn 0.2s ease; }
  .tab-panel.active { display: block; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }

  /* Cards */
  .card {
    background: var(--bg-card); border: 1px solid var(--border-subtle);
    border-radius: 16px; padding: 16px; margin-bottom: 14px;
    backdrop-filter: blur(16px); box-shadow: 0 10px 30px rgba(0,0,0,0.35);
  }
  .section-title {
    font-size: 0.82rem; font-weight: 700; color: #c4b5fd;
    text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 10px;
    display: flex; justify-content: space-between; align-items: center;
  }

  /* Upload Buttons */
  .upload-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 12px; }
  .file-drop-btn {
    background: linear-gradient(180deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.9));
    border: 1.5px dashed rgba(148, 163, 184, 0.35); border-radius: 14px;
    padding: 16px 10px; text-align: center; cursor: pointer; transition: all 0.2s;
    display: flex; flex-direction: column; align-items: center; gap: 6px;
  }
  .file-drop-btn:hover, .file-drop-btn:active {
    border-color: var(--accent-secondary); background: rgba(6, 182, 212, 0.08);
    transform: translateY(-1px);
  }
  .file-drop-icon { font-size: 1.5rem; }
  .file-drop-title { font-size: 0.84rem; font-weight: 700; color: #f1f5f9; }
  .file-drop-sub { font-size: 0.68rem; color: var(--text-muted); }
  input[type="file"] { display: none; }

  /* Dedicated Staged Uploaded Data Box */
  .staged-box {
    background: var(--bg-inner); border: 1px solid var(--border-subtle);
    border-radius: 12px; padding: 10px; margin-bottom: 14px;
  }
  .staged-header {
    display: flex; justify-content: space-between; align-items: center;
    font-size: 0.78rem; font-weight: 700; color: #e2e8f0; padding-bottom: 8px;
    border-bottom: 1px solid var(--border-subtle); margin-bottom: 8px;
  }
  .staged-list {
    max-height: 165px; overflow-y: auto; display: flex; flex-direction: column; gap: 5px;
    padding-right: 4px;
  }
  .staged-item {
    display: flex; justify-content: space-between; align-items: center;
    background: rgba(30, 41, 59, 0.55); padding: 6px 10px; border-radius: 8px;
    font-size: 0.75rem; color: #cbd5e1;
  }
  .staged-item-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 70%; }
  .staged-item-meta { font-size: 0.68rem; color: var(--accent-secondary); font-family: monospace; }
  .clear-files-btn {
    background: rgba(239, 68, 68, 0.15); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.3);
    padding: 3px 8px; border-radius: 6px; font-size: 0.7rem; font-weight: 600; cursor: pointer;
  }

  /* Inputs & Selects */
  label.field-lbl { font-size: 0.76rem; font-weight: 600; color: #94a3b8; display: block; margin-bottom: 5px; }
  input[type="text"], select {
    width: 100%; padding: 11px 12px; background: var(--bg-inner);
    border: 1px solid rgba(148, 163, 184, 0.25); border-radius: 10px;
    color: var(--text-main); font-size: 0.85rem; margin-bottom: 12px; outline: none;
  }
  input[type="text"]:focus, select:focus { border-color: var(--accent-primary); }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }

  /* Primary Action Button & Upload Progress */
  .btn-main {
    width: 100%; padding: 14px;
    background: linear-gradient(135deg, #8b5cf6 0%, #06b6d4 100%);
    color: #fff; border: none; border-radius: 12px; font-size: 0.92rem; font-weight: 800;
    cursor: pointer; box-shadow: 0 6px 20px rgba(139, 92, 246, 0.35); transition: opacity 0.2s;
  }
  .btn-main:disabled { background: #334155; color: #94a3b8; box-shadow: none; cursor: not-allowed; }
  .upload-prog-wrap { display: none; margin-top: 10px; }

  /* Job Cards */
  .job-card {
    background: var(--bg-inner); border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 14px; padding: 14px; margin-bottom: 12px; position: relative;
  }
  .job-card.done-card { border-color: rgba(16, 185, 129, 0.4); }
  .job-header { display: flex; justify-content: space-between; align-items: flex-start; gap: 8px; margin-bottom: 8px; }
  .job-title { font-weight: 800; font-size: 0.95rem; color: #fff; word-break: break-word; }
  .job-meta-pills { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 8px; }
  .meta-pill {
    font-size: 0.68rem; font-weight: 700; padding: 3px 8px; border-radius: 6px;
    background: rgba(255,255,255,0.06); color: #cbd5e1;
  }
  .status-pill {
    font-size: 0.7rem; font-weight: 800; padding: 3px 10px; border-radius: 99px; text-transform: uppercase;
  }
  .status-queued { background: rgba(245, 158, 11, 0.15); color: #fcd34d; border: 1px solid rgba(245, 158, 11, 0.3); }
  .status-processing { background: rgba(6, 182, 212, 0.15); color: #67e8f9; border: 1px solid rgba(6, 182, 212, 0.35); }
  .status-done { background: rgba(16, 185, 129, 0.18); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.4); }
  .status-error { background: rgba(239, 68, 68, 0.18); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.4); }

  /* Progress Bar */
  .prog-bg { width: 100%; height: 9px; background: #1e293b; border-radius: 99px; overflow: hidden; margin: 8px 0; }
  .prog-fill {
    height: 100%; background: linear-gradient(90deg, #8b5cf6, #06b6d4, #10b981);
    transition: width 0.35s ease; border-radius: 99px;
  }
  .log-line {
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #94a3b8;
    background: #05070b; border: 1px solid rgba(255,255,255,0.05);
    padding: 7px 9px; border-radius: 8px; margin-top: 6px; word-break: break-all;
  }

  /* Download & Action Buttons */
  .dl-grid { display: flex; flex-direction: column; gap: 8px; margin-top: 10px; }
  .btn-dl-zip {
    display: flex; align-items: center; justify-content: center; gap: 8px;
    width: 100%; padding: 13px; background: linear-gradient(135deg, #10b981, #059669);
    color: #fff; text-align: center; text-decoration: none; border-radius: 10px;
    font-weight: 800; font-size: 0.88rem; box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
  }
  .btn-dl-single {
    display: flex; align-items: center; justify-content: center; gap: 8px;
    width: 100%; padding: 12px; background: linear-gradient(135deg, #06b6d4, #2563eb);
    color: #fff; text-align: center; text-decoration: none; border-radius: 10px;
    font-weight: 800; font-size: 0.86rem;
  }
  .btn-preview-toggle {
    width: 100%; padding: 10px; background: rgba(139, 92, 246, 0.15);
    color: #c4b5fd; border: 1px solid rgba(139, 92, 246, 0.35); border-radius: 10px;
    font-size: 0.8rem; font-weight: 700; cursor: pointer;
  }
  .btn-wipe {
    width: 100%; padding: 9px; background: rgba(239, 68, 68, 0.1); color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.25); border-radius: 9px;
    font-size: 0.75rem; font-weight: 700; margin-top: 6px; cursor: pointer;
  }

  /* Reader / Individual Page List inside Completed Tab */
  .reader-box {
    display: none; margin-top: 10px; padding-top: 10px;
    border-top: 1px solid var(--border-subtle);
  }
  .reader-box.open { display: block; }
  .page-card {
    background: #070a12; border: 1px solid #1e293b; border-radius: 10px;
    padding: 8px; margin-bottom: 10px;
  }
  .page-card-head {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 6px; font-size: 0.74rem; color: #93c5fd; font-weight: 600;
  }
  .page-dl-mini {
    background: #1e293b; color: #38bdf8; text-decoration: none;
    padding: 3px 8px; border-radius: 6px; font-size: 0.7rem; font-weight: 700;
  }
  .page-card img { width: 100%; border-radius: 6px; display: block; }
  .empty-state {
    text-align: center; padding: 28px 12px; color: var(--text-muted); font-size: 0.84rem;
  }
</style>
</head>
<body>
<div class="container">
  <!-- Header -->
  <div class="app-header">
    <div class="brand-pill"><span class="brand-dot"></span> Dual T4 • 100% Pure RAM Engine</div>
    <h1>Manga & Manhwa AI Studio</h1>
    <div class="sub">YOLOv8 • Qwen3.5-9B Vision OCR • Qwen3.8-27B + MTP Translator</div>
  </div>

  <!-- 3 Separate View Tabs -->
  <div class="nav-tabs">
    <button class="tab-btn active" id="tabBtn-upload" onclick="switchTab('upload')">
      📤 Upload <span class="tab-count" id="stagedBadge">0</span>
    </button>
    <button class="tab-btn" id="tabBtn-queue" onclick="switchTab('queue')">
      ⚡ Queue <span class="tab-count" id="activeBadge">0</span>
    </button>
    <button class="tab-btn" id="tabBtn-done" onclick="switchTab('done')">
      📦 Ready <span class="tab-count done-badge" id="doneBadge">0</span>
    </button>
  </div>

  <!-- ================= TAB 1: UPLOAD & STAGED FILES VIEW ================= -->
  <div class="tab-panel active" id="panel-upload">
    <div class="card">
      <div class="section-title">
        <span>1. Select Raw Pages or Folder</span>
      </div>
      <div class="upload-grid">
        <label class="file-drop-btn">
          <span class="file-drop-icon">🖼️</span>
          <span class="file-drop-title">Select Images</span>
          <span class="file-drop-sub">1 or multiple pages</span>
          <input type="file" id="filesInput" accept="image/*" multiple>
        </label>
        <label class="file-drop-btn">
          <span class="file-drop-icon">📁</span>
          <span class="file-drop-title">Select Folder</span>
          <span class="file-drop-sub">Full 60+ page chapter</span>
          <input type="file" id="folderInput" webkitdirectory directory multiple>
        </label>
      </div>

      <!-- Dedicated Staged Data Inspector -->
      <div class="staged-box">
        <div class="staged-header">
          <span id="fileSummaryText">📂 No files staged yet</span>
          <button class="clear-files-btn" id="clearStagedBtn" style="display:none;" onclick="clearStagedFiles()">✖ Clear</button>
        </div>
        <div class="staged-list" id="stagedFilesList">
          <div style="text-align:center;color:#64748b;font-size:0.75rem;padding:10px;">
            Selected images or folder contents will appear here in reading order before sending to RAM.
          </div>
        </div>
      </div>

      <div class="section-title"><span>2. Translation Settings</span></div>
      <label class="field-lbl">Chapter / Batch Name (Used for Output ZIP & Queue)</label>
      <input type="text" id="chapterName" placeholder="Auto-filled from folder name or e.g. Chapter_01">

      <div class="row">
        <div>
          <label class="field-lbl">Reading Order</label>
          <select id="readingOrder">
            <option value="rtl" selected>🇯🇵 Manga (Right-to-Left)</option>
            <option value="ltr">🇰🇷🇨🇳 Manhwa (Left-to-Right)</option>
          </select>
        </div>
        <div>
          <label class="field-lbl">27B Translation Mode</label>
          <select id="thinkingMode">
            <option value="false" selected>⚡ Direct JSON (Fastest)</option>
            <option value="true">🧠 Deep Thinking</option>
          </select>
        </div>
      </div>

      <button id="submitBtn" class="btn-main" onclick="uploadBatch()">🚀 Send to Kaggle RAM & Start Translation</button>

      <!-- Live Upload Progress Bar -->
      <div class="upload-prog-wrap" id="uploadProgWrap">
        <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#67e8f9;margin-bottom:4px;">
          <span id="uploadProgLabel">Uploading to RAM...</span>
          <span id="uploadProgPct">0%</span>
        </div>
        <div class="prog-bg"><div class="prog-fill" id="uploadProgBar" style="width:0%;"></div></div>
      </div>
    </div>
  </div>

  <!-- ================= TAB 2: ACTIVE QUEUE VIEW ================= -->
  <div class="tab-panel" id="panel-queue">
    <div class="card">
      <div class="section-title">
        <span>⚡ Active GPU Processing & Queue</span>
        <span style="font-size:0.7rem;color:#94a3b8;text-transform:none;">Auto-refreshes every 2s</span>
      </div>
      <div id="activeJobsContainer" class="empty-state">No jobs currently processing in GPU/RAM.</div>
    </div>
  </div>

  <!-- ================= TAB 3: COMPLETED & DOWNLOADS VIEW ================= -->
  <div class="tab-panel" id="panel-done">
    <div class="card">
      <div class="section-title">
        <span>📦 Completed Chapters & Single Images</span>
        <span style="font-size:0.7rem;color:#10b981;text-transform:none;">Pure RAM Stream</span>
      </div>
      <div id="doneJobsContainer" class="empty-state">No completed translations yet. Finished jobs will appear here automatically!</div>
    </div>
  </div>
</div>

<script>
let selectedFiles = [];
let openPreviews = {}; // Remembers which job galleries the user opened so polling never closes them!
let lastDoneCount = 0;

function switchTab(tabName) {
  ['upload', 'queue', 'done'].forEach(t => {
    document.getElementById('tabBtn-' + t).classList.toggle('active', t === tabName);
    document.getElementById('panel-' + t).classList.toggle('active', t === tabName);
  });
}

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / 1048576).toFixed(1) + ' MB';
}

function clearStagedFiles() {
  selectedFiles = [];
  document.getElementById('filesInput').value = '';
  document.getElementById('folderInput').value = '';
  renderStagedFiles();
}

function handleSelectedFiles(fileList, isFolder) {
  const validExts = ['.jpg', '.jpeg', '.png', '.webp'];
  selectedFiles = Array.from(fileList).filter(f => {
    const lower = f.name.toLowerCase();
    return validExts.some(ext => lower.endsWith(ext));
  });
  selectedFiles.sort((a, b) => a.name.localeCompare(b.name, undefined, {numeric: true, sensitivity: 'base'}));

  const chapInput = document.getElementById('chapterName');
  if (selectedFiles.length > 0) {
    if (isFolder && selectedFiles[0].webkitRelativePath) {
      const folderName = selectedFiles[0].webkitRelativePath.split('/')[0];
      if (folderName) chapInput.value = folderName;
    } else if (!chapInput.value) {
      chapInput.value = selectedFiles.length === 1
        ? selectedFiles[0].name.replace(/\\.[^/.]+$/, "")
        : "Chapter_" + new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}).replace(':','-');
    }
  }
  renderStagedFiles();
}

function renderStagedFiles() {
  const sumEl = document.getElementById('fileSummaryText');
  const listEl = document.getElementById('stagedFilesList');
  const clearBtn = document.getElementById('clearStagedBtn');
  const badge = document.getElementById('stagedBadge');

  badge.innerText = selectedFiles.length;

  if (selectedFiles.length === 0) {
    sumEl.innerText = "📂 No files staged yet";
    clearBtn.style.display = "none";
    listEl.innerHTML = `<div style="text-align:center;color:#64748b;font-size:0.75rem;padding:10px;">Selected images or folder contents will appear here in reading order before sending to RAM.</div>`;
    return;
  }

  const totalBytes = selectedFiles.reduce((acc, f) => acc + f.size, 0);
  sumEl.innerText = `✅ ${selectedFiles.length} Page(s) Staged (${formatBytes(totalBytes)})`;
  clearBtn.style.display = "inline-block";

  listEl.innerHTML = selectedFiles.map((f, idx) => `
    <div class="staged-item">
      <span class="staged-item-name">#${idx + 1} • ${f.name}</span>
      <span class="staged-item-meta">${formatBytes(f.size)}</span>
    </div>
  `).join('');
}

document.getElementById('filesInput').addEventListener('change', e => handleSelectedFiles(e.target.files, false));
document.getElementById('folderInput').addEventListener('change', e => handleSelectedFiles(e.target.files, true));

function uploadBatch() {
  if (selectedFiles.length === 0) {
    alert("Please select images or a folder first!");
    return;
  }
  const btn = document.getElementById('submitBtn');
  const progWrap = document.getElementById('uploadProgWrap');
  const progBar = document.getElementById('uploadProgBar');
  const progPct = document.getElementById('uploadProgPct');
  const progLbl = document.getElementById('uploadProgLabel');

  btn.disabled = true;
  progWrap.style.display = 'block';
  progBar.style.width = '0%';
  progPct.innerText = '0%';
  progLbl.innerText = `Uploading ${selectedFiles.length} file(s) to Kaggle RAM...`;

  const formData = new FormData();
  formData.append('chapter_name', document.getElementById('chapterName').value || 'Manga_Chapter');
  formData.append('rtl', document.getElementById('readingOrder').value === 'rtl' ? 'true' : 'false');
  formData.append('thinking', document.getElementById('thinkingMode').value);

  for (const file of selectedFiles) {
    formData.append('images', file, file.name);
  }

  const xhr = new XMLHttpRequest();
  xhr.open('POST', '/api/upload_job', true);

  xhr.upload.onprogress = function(e) {
    if (e.lengthComputable) {
      const pct = Math.round((e.loaded / e.total) * 100);
      progBar.style.width = pct + '%';
      progPct.innerText = pct + '%';
      if (pct === 100) {
        progLbl.innerText = 'Decoding images into Kaggle RAM...';
      }
    }
  };

  xhr.onload = function() {
    btn.disabled = false;
    progWrap.style.display = 'none';
    if (xhr.status >= 200 && xhr.status < 300) {
      const data = JSON.parse(xhr.responseText);
      if (data.error) {
        alert("Error: " + data.error);
      } else {
        clearStagedFiles();
        document.getElementById('chapterName').value = '';
        refreshJobs();
        switchTab('queue'); // Automatically take user to Active Queue view!
      }
    } else {
      alert("Upload failed with status " + xhr.status);
    }
  };

  xhr.onerror = function() {
    btn.disabled = false;
    progWrap.style.display = 'none';
    alert("Network error while uploading batch.");
  };

  xhr.send(formData);
}

function togglePreview(jobId) {
  openPreviews[jobId] = !openPreviews[jobId];
  const el = document.getElementById('reader-' + jobId);
  if (el) el.classList.toggle('open', !!openPreviews[jobId]);
}

async function wipeJob(jobId) {
  delete openPreviews[jobId];
  await fetch('/api/wipe_job/' + jobId, { method: 'POST' });
  refreshJobs();
}

async function refreshJobs() {
  try {
    const res = await fetch('/api/jobs');
    const data = await res.json();
    const allJobs = data.jobs || [];

    const activeJobs = allJobs.filter(j => j.status === 'queued' || j.status === 'processing');
    const doneJobs = allJobs.filter(j => j.status === 'done' || j.status === 'error');

    const activeBadge = document.getElementById('activeBadge');
    const doneBadge = document.getElementById('doneBadge');
    activeBadge.innerText = activeJobs.length;
    activeBadge.classList.toggle('pulse', activeJobs.length > 0);
    doneBadge.innerText = doneJobs.length;

    // Auto-switch to Ready tab when a new job finishes while user is watching Queue!
    const strictlyDoneCount = doneJobs.filter(j => j.status === 'done').length;
    if (strictlyDoneCount > lastDoneCount && activeJobs.length === 0 && document.getElementById('panel-queue').classList.contains('active')) {
      switchTab('done');
    }
    lastDoneCount = strictlyDoneCount;

    // Render Active Queue Tab
    const activeContainer = document.getElementById('activeJobsContainer');
    if (activeJobs.length === 0) {
      activeContainer.innerHTML = `<div class="empty-state">No jobs currently processing. Upload a folder or image in the Upload tab!</div>`;
    } else {
      activeContainer.innerHTML = activeJobs.map(j => `
        <div class="job-card" style="text-align:left;">
          <div class="job-header">
            <span class="job-title">📁 ${j.chapter_name}</span>
            <span class="status-pill status-${j.status}">${j.status}</span>
          </div>
          <div class="job-meta-pills">
            <span class="meta-pill">📄 ${j.page_count} Page(s)</span>
            <span class="meta-pill">${j.rtl ? '🇯🇵 RTL Manga' : '🇰🇷 LTR Manhwa'}</span>
          </div>
          <div style="font-size:0.8rem;color:#e2e8f0;font-weight:600;">${j.phase}</div>
          <div class="prog-bg"><div class="prog-fill" style="width:${j.progress}%;"></div></div>
          <div class="log-line">${j.last_log || 'Waiting in queue...'}</div>
        </div>
      `).join('');
    }

    // Render Completed / Downloads Tab
    const doneContainer = document.getElementById('doneJobsContainer');
    if (doneJobs.length === 0) {
      doneContainer.innerHTML = `<div class="empty-state">No completed translations in RAM yet.</div>`;
    } else {
      doneContainer.innerHTML = doneJobs.map(j => {
        if (j.status === 'error') {
          return `
            <div class="job-card" style="text-align:left;">
              <div class="job-header">
                <span class="job-title">📁 ${j.chapter_name}</span>
                <span class="status-pill status-error">FAILED</span>
              </div>
              <div class="log-line">${j.phase}</div>
              <button class="btn-wipe" onclick="wipeJob('${j.job_id}')">✖ Remove Failed Job from RAM</button>
            </div>
          `;
        }

        const isOpen = !!openPreviews[j.job_id];
        const singleDownloadBtn = j.page_count === 1 ? `
          <a class="btn-dl-single" href="/api/page/${j.job_id}/0?dl=1" download="${j.filenames[0] || 'translated.png'}">
            🖼️ Download Translated Image Directly (PNG)
          </a>
        ` : '';

        // Only load <img> elements when the user opens the preview reader!
        const pagesHtml = isOpen ? (j.filenames || []).map((fname, idx) => `
          <div class="page-card">
            <div class="page-card-head">
              <span>#${idx + 1} • ${fname}</span>
              <a class="page-dl-mini" href="/api/page/${j.job_id}/${idx}?dl=1" download="${fname}">⬇ Save PNG</a>
            </div>
            <img src="/api/page/${j.job_id}/${idx}" loading="lazy">
          </div>
        `).join('') : '';

        return `
          <div class="job-card done-card" style="text-align:left;">
            <div class="job-header">
              <span class="job-title">✅ ${j.chapter_name}</span>
              <span class="status-pill status-done">READY</span>
            </div>
            <div class="job-meta-pills">
              <span class="meta-pill">📄 ${j.page_count} Page(s)</span>
              <span class="meta-pill">💬 ${j.total_bubbles || 0} Bubbles</span>
              <span class="meta-pill">⏱️ ${j.elapsed || 0}s</span>
            </div>
            <div class="dl-grid">
              ${singleDownloadBtn}
              <a class="btn-dl-zip" href="/api/download_zip/${j.job_id}" download="${j.chapter_name}_Translated.zip">
                📦 Download All ${j.page_count} Page(s) as .ZIP (Fast Binary Stream)
              </a>
              <button class="btn-preview-toggle" onclick="togglePreview('${j.job_id}')">
                ${isOpen ? '🔼 Hide Translated Pages' : `👁️ Preview / Download Individual Pages (${j.page_count})`}
              </button>
            </div>
            <div class="reader-box ${isOpen ? 'open' : ''}" id="reader-${j.job_id}">
              ${pagesHtml}
            </div>
            <button class="btn-wipe" onclick="wipeJob('${j.job_id}')">🧹 Purge Job from Kaggle RAM</button>
          </div>
        `;
      }).join('');
    }
  } catch (e) {}
}

setInterval(refreshJobs, 2000);
refreshJobs();
</script>
</body>
</html>
"""

# ─────────────────────────────────────────────────────────────
# FLASK API ROUTES (ULTRA-FAST BINARY STREAMING FROM RAM)
# ─────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # Allow up to 1 GB batch upload in RAM


@app.route("/")
def index():
    return render_template_string(HTML_UI)


@app.route("/api/upload_job", methods=["POST"])
def api_upload_job():
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images uploaded"}), 400

    chapter_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", request.form.get("chapter_name", "Manga_Batch")).strip("_") or "Manga_Batch"
    rtl = request.form.get("rtl", "true").lower() == "true"
    thinking = request.form.get("thinking", "false").lower() == "true"

    ram_pages = []
    for f in files:
        raw_bytes = f.read()
        if not raw_bytes:
            continue
        np_arr = np.frombuffer(raw_bytes, np.uint8)
        bgr_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if bgr_img is not None:
            clean_fname = os.path.basename(f.filename)
            ram_pages.append((clean_fname, bgr_img))

    if not ram_pages:
        return jsonify({"error": "Could not decode any valid images"}), 400

    # Natural alphanumeric sort (01.jpg, 02.jpg, ... 10.jpg, 100.jpg)
    ram_pages.sort(key=lambda x: [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', x[0])])

    job_id = str(uuid.uuid4())[:8]
    queue_pos = JOB_QUEUE.qsize() + 1

    with JOBS_LOCK:
        JOBS_RAM[job_id] = {
            "job_id": job_id,
            "chapter_name": chapter_name,
            "page_count": len(ram_pages),
            "rtl": rtl,
            "thinking": thinking,
            "status": "queued",
            "phase": f"⏳ Queued (#{queue_pos}) — Waiting for active job...",
            "progress": 2,
            "last_log": f"Loaded {len(ram_pages)} pages strictly into Kaggle System RAM.",
            "logs": [],
            "pages_in_ram": ram_pages,
            "out_pages": [],
            "zip_bytes": None,
            "elapsed": 0,
            "total_bubbles": 0,
            "created_at": time.time()
        }

    JOB_QUEUE.put(job_id)
    print(f"📥 [Mobile Upload -> RAM] Queued '{chapter_name}' ({len(ram_pages)} pages) as Job #{job_id}", flush=True)
    return jsonify({"job_id": job_id, "page_count": len(ram_pages)})


@app.route("/api/jobs", methods=["GET"])
def api_list_jobs():
    """
    LIGHTWEIGHT METADATA ONLY (~2 KB):
    Never sends 200 MB of Base64 images inside the status poll!
    """
    with JOBS_LOCK:
        jobs_list = []
        for jid, j in sorted(JOBS_RAM.items(), key=lambda item: item[1]["created_at"], reverse=True):
            filenames = [p["filename"] for p in j.get("out_pages", [])] if j["status"] == "done" else []
            jobs_list.append({
                "job_id": jid,
                "chapter_name": j["chapter_name"],
                "page_count": j["page_count"],
                "rtl": j.get("rtl", True),
                "status": j["status"],
                "phase": j["phase"],
                "progress": j["progress"],
                "last_log": j["last_log"],
                "elapsed": j.get("elapsed", 0),
                "total_bubbles": j.get("total_bubbles", 0),
                "filenames": filenames
            })
    return jsonify({"jobs": jobs_list})


@app.route("/api/page/<job_id>/<int:page_idx>", methods=["GET"])
def api_get_single_page(job_id, page_idx):
    """Streams a single translated PNG directly from RAM (for fast preview or 1-image download)."""
    with JOBS_LOCK:
        job = JOBS_RAM.get(job_id)
        if not job or job.get("status") != "done":
            return "Page not found", 404
        out_pages = job.get("out_pages", [])
        if page_idx < 0 or page_idx >= len(out_pages):
            return "Invalid page index", 404
        page_info = out_pages[page_idx]
        png_bytes = page_info["bytes"]
        fname = page_info["filename"]

    as_dl = request.args.get("dl", "0") == "1"
    return send_file(
        io.BytesIO(png_bytes),
        mimetype="image/png",
        as_attachment=as_dl,
        download_name=fname
    )


@app.route("/api/download_zip/<job_id>", methods=["GET"])
def api_download_zip(job_id):
    """Streams the entire translated chapter ZIP directly from RAM in binary mode."""
    with JOBS_LOCK:
        job = JOBS_RAM.get(job_id)
        if not job or not job.get("zip_bytes"):
            return "Job not found or already cleared from RAM", 404
        zip_bytes = job["zip_bytes"]
        dl_name = f"{job['chapter_name']}_Translated.zip"

    return send_file(
        io.BytesIO(zip_bytes),
        mimetype="application/zip",
        as_attachment=True,
        download_name=dl_name
    )


@app.route("/api/wipe_job/<job_id>", methods=["POST"])
def api_wipe_job(job_id):
    free_job_ram(job_id)
    return jsonify({"ok": True})


# ─────────────────────────────────────────────────────────────
# MAIN STARTUP: PRE-DOWNLOAD MODELS + CLOUDFLARE TUNNEL + LIVE LOGS
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 65, flush=True)
    print(" 🇯🇵🇰🇷🇨🇳 MANGA / MANHWA PURE-RAM TRANSLATION SERVER (DUAL T4)", flush=True)
    print("=" * 65, flush=True)

    # 1. Pre-download models once into /tmp/models so mobile jobs run immediately
    print("\n📦 Step 1: Verifying / Downloading Models in /tmp/models (Once per session)...", flush=True)
    ensure_ocr_model_downloaded()
    ensure_translation_models_downloaded()

    # 2. Start Background Job Queue Worker Thread
    worker_thread = threading.Thread(target=background_queue_daemon, daemon=True)
    worker_thread.start()

    # 3. Start Flask Server in Background Thread
    flask_thread = threading.Thread(
        target=lambda: app.run(host="127.0.0.1", port=FLASK_PORT, debug=False, threaded=True, use_reloader=False),
        daemon=True
    )
    flask_thread.start()

    # 4. Download & Launch Cloudflare Tunnel to Expose Mobile UI
    cf_bin = "/tmp/cloudflared"
    if not os.path.exists(cf_bin):
        print("🌐 Downloading Cloudflare Tunnel binary to /tmp...", flush=True)
        subprocess.run(
            f"wget -q -O {cf_bin} https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x {cf_bin}",
            shell=True,
            check=True
        )

    cf_log_path = os.path.join(RAM_BASE, "cloudflared_ui.log")
    cf_log_file = open(cf_log_path, "w", buffering=1)
    cf_process = subprocess.Popen(
        [cf_bin, "tunnel", "--url", f"http://127.0.0.1:{FLASK_PORT}"],
        stdout=cf_log_file,
        stderr=subprocess.STDOUT,
        text=True
    )

    cloudflare_url = None
    for _ in range(30):
        if os.path.exists(cf_log_path):
            with open(cf_log_path, "r") as f:
                m = re.search(r"(https://[a-zA-Z0-9-]+\.trycloudflare\.com)", f.read())
                if m:
                    cloudflare_url = m.group(1)
                    break
        time.sleep(1)

    if cloudflare_url:
        print("\n" + "★" * 65, flush=True)
        print(f"📱 MOBILE WEB UI IS LIVE:  {cloudflare_url}", flush=True)
        print("★" * 65 + "\n", flush=True)

        if DISCORD_WEBHOOK_URL:
            try:
                msg = (
                    f"🚀 **Manga/Manhwa Pure-RAM Mobile Translator is LIVE!**\n"
                    f"👁️ **OCR:** `Qwen3.5-9B-Uncensored (Q8_0 + BF16 mmproj)`\n"
                    f"🌐 **Translator:** `Qwen3.8-27B-Uncensored-OrcaRouter (Q6_K + MTP)`\n"
                    f"📱 **Open Mobile UI:** {cloudflare_url}"
                )
                requests.post(DISCORD_WEBHOOK_URL, json={"content": msg}, timeout=5)
                print("📨 Sent Mobile UI URL to Discord Webhook!", flush=True)
            except Exception:
                pass
    else:
        print("⚠️ Could not extract Cloudflare URL. Check /dev/shm/manga_ram/cloudflared_ui.log", flush=True)

    print("📡 Live Pipeline Logs are active below. Waiting for Mobile Uploads...\n", flush=True)
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down server and cleaning RAM...", flush=True)
        stop_ocr_server()
        stop_translation_server()
        cf_process.terminate()


if __name__ == "__main__":
    main()
