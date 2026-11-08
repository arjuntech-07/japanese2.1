# main.py
"""
Pure-RAM Manga / Manhwa / Manhua Translation Server & Mobile Web UI
- Zero Disk Touch for Images: Uploads -> RAM -> GPU VRAM -> RAM -> Mobile Download -> Auto-Wipe
- OCR Engine: Qwen 3.5 9B Uncensored Q8_0 + BF16 mmproj (Thinking OFF)
- Translation Engine: Qwen 3.8 27B Uncensored OrcaRouter Q6_K + MTP Companion (Context-Aware)
- Multi-Job Queue: Upload multiple folders/chapters back-to-back from mobile with live progress
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
import base64
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
    "/kaggle/input/models/rohith7boi/yolov8/pytorch/default/1/best.pt"
)
DISCORD_WEBHOOK_URL = os.environ.get(
    "DISCORD_WEBHOOK_URL",
    "https://discord.com/api/webhooks/1537848230539300905/9JN_TypsOffB8kblNdYv3e_e49_h898ph6LWSwOuccMogAogafNx5H_eIS68yKw55Y6E"
)

# ─────────────────────────────────────────────────────────────
# IN-MEMORY JOB QUEUE & RAM STORE (NEVER TOUCHES DISK)
# ─────────────────────────────────────────────────────────────
JOB_QUEUE = queue.Queue()
JOBS_RAM = {}  # { job_id: { status, phase, progress, logs, pages_in_ram, out_pages_b64, zip_bytes, ... } }
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
# 4-PHASE SEQUENTIAL PURE-RAM PIPELINE WORKER
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
            rolling_context = []  # Holds last 4 translated pairs for cross-chunk pronoun/story memory!
            total_chunks = (len(valid_texts) + CHUNK_SIZE - 1) // CHUNK_SIZE

            for c_idx, i in enumerate(range(0, len(valid_texts), CHUNK_SIZE)):
                chunk_texts = valid_texts[i: i + CHUNK_SIZE]
                chunk_cids = valid_crop_ids[i: i + CHUNK_SIZE]

                # Prepend rolling context hint to first line if available so 27B keeps story continuity
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
        # PHASE 4: INPAINT CLEANING & TYPESETTING IN PURE RAM + ZIP PACKAGING
        # =========================================================================
        update_job_state(
            job_id,
            phase=f"4/4: Cleaning & Typesetting (0/{total_pages} pages)",
            progress=82
        )
        log_job(job_id, f"\n🎨 [Phase 4/4] Cleaning Speech Bubbles & Rendering English Text in RAM...")

        output_pages = []  # List of {"filename": ..., "b64": ...}
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p_idx, (filename, full_img) in enumerate(ram_pages):
                bubbles = page_map.get(filename, [])

                for b_info in bubbles:
                    cid = b_info["crop_id"]
                    english_text = translations_map.get(cid, "")
                    x1, y1, x2, y2 = b_info["box"]
                    bubble_img = b_info["crop_img"]

                    if bubble_img is None or bubble_img.size == 0:
                        continue

                    # Always erase original Japanese/Korean/Chinese text inside bubble
                    cleaned_bubble, _ = process_contour(bubble_img)

                    # Render translated English dialogue if non-empty
                    if english_text and english_text.strip():
                        add_text(cleaned_bubble, english_text)

                    target_w = x2 - x1
                    target_h = y2 - y1
                    if cleaned_bubble.shape[0] != target_h or cleaned_bubble.shape[1] != target_w:
                        cleaned_bubble = cv2.resize(
                            cleaned_bubble,
                            (target_w, target_h),
                            interpolation=cv2.INTER_AREA
                        )

                    full_img[y1:y2, x1:x2] = cleaned_bubble

                    # Free individual crop from RAM immediately
                    b_info["crop_img"] = None

                # Encode finished page to PNG in RAM
                out_fname = f"translated_{filename}"
                if not out_fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    out_fname += ".png"

                _, png_buf = cv2.imencode(".png", full_img)
                png_bytes = png_buf.tobytes()

                # Write directly into in-memory ZIP archive
                zf.writestr(out_fname, png_bytes)

                # Also store Base64 for instant Mobile UI preview gallery
                b64_page = base64.b64encode(png_bytes).decode("utf-8")
                output_pages.append({
                    "filename": out_fname,
                    "b64": b64_page
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

        # Wipe raw input numpy arrays from RAM immediately; keep only output until user downloads
        with JOBS_LOCK:
            if job_id in JOBS_RAM:
                JOBS_RAM[job_id]["pages_in_ram"] = []
                JOBS_RAM[job_id]["out_pages"] = output_pages
                JOBS_RAM[job_id]["zip_bytes"] = zip_buffer.getvalue()
                JOBS_RAM[job_id]["status"] = "done"
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
# MOBILE WEB UI (HTML5 + FOLDER/MULTI-FILE UPLOAD + LIVE QUEUE)
# ─────────────────────────────────────────────────────────────
HTML_UI = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>Manga/Manhwa RAM Translator</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  body { background: #0b0f19; color: #e2e8f0; padding: 14px; display: flex; flex-direction: column; align-items: center; }
  .container { width: 100%; max-width: 600px; }
  h1 { font-size: 1.25rem; font-weight: 800; color: #f8fafc; text-align: center; margin-bottom: 4px; }
  .badge { background: #1e293b; color: #38bdf8; font-size: 0.7rem; padding: 3px 8px; border-radius: 99px; border: 1px solid #38bdf844; vertical-align: middle; }
  .sub { font-size: 0.76rem; color: #94a3b8; text-align: center; margin-bottom: 14px; }
  .card { background: #131c2e; border: 1px solid #263554; border-radius: 12px; padding: 14px; margin-bottom: 12px; }
  label { font-size: 0.8rem; font-weight: 600; color: #93c5fd; display: block; margin-bottom: 6px; }
  .upload-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 8px; }
  .file-btn { background: #1e293b; border: 1px dashed #475569; border-radius: 8px; padding: 12px 8px; text-align: center; cursor: pointer; font-size: 0.8rem; color: #cbd5e1; font-weight: 600; }
  .file-btn:hover { border-color: #38bdf8; color: #fff; }
  input[type="file"] { display: none; }
  input[type="text"], select { width: 100%; padding: 9px; background: #0b0f19; border: 1px solid #334155; border-radius: 8px; color: #f8fafc; font-size: 0.85rem; margin-bottom: 8px; }
  .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  .col { flex: 1; min-width: 130px; }
  .btn-main { width: 100%; padding: 14px; background: linear-gradient(135deg, #f97316, #ea580c); color: #fff; border: none; border-radius: 10px; font-size: 0.95rem; font-weight: 700; cursor: pointer; }
  .btn-main:disabled { background: #334155; color: #64748b; }
  .file-summary { font-size: 0.8rem; color: #10b981; text-align: center; margin: 6px 0; font-weight: 600; }
  .job-card { background: #0f172a; border: 1px solid #334155; border-radius: 10px; padding: 12px; margin-bottom: 10px; }
  .job-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
  .job-title { font-weight: 700; font-size: 0.9rem; color: #f8fafc; }
  .job-badge { font-size: 0.72rem; padding: 2px 8px; border-radius: 99px; background: #1e293b; color: #fbbf24; }
  .prog-bg { width: 100%; height: 8px; background: #1e293b; border-radius: 99px; overflow: hidden; margin: 6px 0; }
  .prog-fill { height: 100%; background: #38bdf8; transition: width 0.3s; }
  .log-line { font-family: monospace; font-size: 0.74rem; color: #94a3b8; background: #090d16; padding: 6px 8px; border-radius: 6px; margin-top: 6px; word-break: break-all; }
  .btn-dl { display: block; width: 100%; padding: 11px; background: #10b981; color: #fff; text-align: center; text-decoration: none; border-radius: 8px; font-weight: 700; font-size: 0.85rem; margin-top: 8px; }
  .btn-wipe { width: 100%; padding: 8px; background: #ef444422; color: #f87171; border: 1px solid #ef444455; border-radius: 8px; font-size: 0.75rem; font-weight: 600; margin-top: 6px; cursor: pointer; }
  .gallery { display: flex; flex-direction: column; gap: 10px; margin-top: 10px; max-height: 520px; overflow-y: auto; }
  .gallery img { width: 100%; border-radius: 6px; border: 1px solid #334155; }
</style>
</head>
<body>
<div class="container">
  <h1>Manga & Manhwa Translator <span class="badge">100% Pure RAM</span></h1>
  <div class="sub">YOLOv8 • Qwen3.5-9B Vision OCR • Qwen3.8-27B + MTP Translator</div>

  <div class="card">
    <label>1. Select Manga/Manhwa Images or Whole Folder (Sent Straight to Kaggle RAM)</label>
    <div class="upload-grid">
      <label class="file-btn">
        🖼️ Select Images
        <input type="file" id="filesInput" accept="image/*"multiple>
      </label>
      <label class="file-btn">
        📁 Select Folder
        <input type="file" id="folderInput" webkitdirectory directory multiple>
      </label>
    </div>
    <div id="fileSummary" class="file-summary">No files selected yet</div>

    <label>2. Chapter / Batch Name (Used for Output ZIP & Queue)</label>
    <input type="text" id="chapterName" placeholder="Auto-filled from folder name or e.g. Chapter_01">

    <div class="row">
      <div class="col">
        <label>Reading Order</label>
        <select id="readingOrder">
          <option value="rtl" selected>🇯🇵 Manga (Right-to-Left)</option>
          <option value="ltr">🇰🇷🇨🇳 Manhwa/Manhua (Left-to-Right)</option>
        </select>
      </div>
      <div class="col">
        <label>27B Translation Mode</label>
        <select id="thinkingMode">
          <option value="false" selected>⚡ Direct JSON (Fastest, No Think)</option>
          <option value="true">🧠 Deep Thinking (Strip &lt;think&gt;)</option>
        </select>
      </div>
    </div>

    <button id="submitBtn" class="btn-main" onclick="uploadBatch()">🚀 Upload to RAM & Join Translation Queue</button>
  </div>

  <div class="card">
    <label>📋 Active & Completed Jobs in RAM (Queue)</label>
    <div id="jobsContainer" style="font-size:0.8rem; color:#64748b; text-align:center;">No active jobs in RAM.</div>
  </div>
</div>

<script>
let selectedFiles = [];

function handleSelectedFiles(fileList, isFolder) {
  const validExts = ['.jpg', '.jpeg', '.png', '.webp'];
  selectedFiles = Array.from(fileList).filter(f => {
    const lower = f.name.toLowerCase();
    return validExts.some(ext => lower.endsWith(ext));
  });
  selectedFiles.sort((a, b) => a.name.localeCompare(b.name, undefined, {numeric: true, sensitivity: 'base'}));

  const sumEl = document.getElementById('fileSummary');
  const chapInput = document.getElementById('chapterName');

  if (selectedFiles.length > 0) {
    sumEl.innerText = `✅ Selected ${selectedFiles.length} image(s) ready for RAM upload`;
    if (isFolder && selectedFiles[0].webkitRelativePath) {
      const folderName = selectedFiles[0].webkitRelativePath.split('/')[0];
      if (folderName && !chapInput.value) chapInput.value = folderName;
    } else if (!chapInput.value) {
      chapInput.value = "Manga_Batch_" + new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}).replace(':','-');
    }
  } else {
    sumEl.innerText = "No valid image files found";
  }
}

document.getElementById('filesInput').addEventListener('change', e => handleSelectedFiles(e.target.files, false));
document.getElementById('folderInput').addEventListener('change', e => handleSelectedFiles(e.target.files, true));

async function uploadBatch() {
  if (selectedFiles.length === 0) {
    alert("Please select images or a folder first!");
    return;
  }
  const btn = document.getElementById('submitBtn');
  btn.disabled = true;
  btn.innerText = `⏳ Uploading ${selectedFiles.length} images into Kaggle RAM...`;

  const formData = new FormData();
  formData.append('chapter_name', document.getElementById('chapterName').value || 'Manga_Chapter');
  formData.append('rtl', document.getElementById('readingOrder').value === 'rtl' ? 'true' : 'false');
  formData.append('thinking', document.getElementById('thinkingMode').value);

  for (const file of selectedFiles) {
    formData.append('images', file, file.name);
  }

  try {
    const res = await fetch('/api/upload_job', { method: 'POST', body: formData });
    const data = await res.json();
    if (data.error) {
      alert("Error: " + data.error);
    } else {
      selectedFiles = [];
      document.getElementById('filesInput').value = '';
      document.getElementById('folderInput').value = '';
      document.getElementById('chapterName').value = '';
      document.getElementById('fileSummary').innerText = "✅ Uploaded! You can immediately select & queue another folder.";
      refreshJobs();
    }
  } catch (e) {
    alert("Upload failed: " + e);
  } finally {
    btn.disabled = false;
    btn.innerText = "🚀 Upload to RAM & Join Translation Queue";
  }
}

async function wipeJob(jobId) {
  await fetch('/api/wipe_job/' + jobId, { method: 'POST' });
  refreshJobs();
}

async function refreshJobs() {
  try {
    const res = await fetch('/api/jobs');
    const data = await res.json();
    const container = document.getElementById('jobsContainer');
    if (!data.jobs || data.jobs.length === 0) {
      container.innerHTML = "No active jobs in RAM.";
      return;
    }

    container.innerHTML = data.jobs.map(j => {
      let actionsHtml = '';
      if (j.status === 'done') {
        const galleryImgs = (j.out_pages || []).map(p =>
          `<div><div style="font-size:0.72rem;color:#93c5fd;margin-bottom:2px;">${p.filename}</div><img src="data:image/png;base64,${p.b64}" loading="lazy"></div>`
        ).join('');
        actionsHtml = `
          <a class="btn-dl" href="/api/download_zip/${j.job_id}" download="${j.chapter_name}_Translated.zip">
            📦 Download Full Translated Chapter (.ZIP) (${j.page_count} Pages)
          </a>
          <details style="margin-top:8px;">
            <summary style="cursor:pointer;color:#38bdf8;font-size:0.8rem;font-weight:600;">👁️ Tap to View Translated Pages (${j.page_count})</summary>
            <div class="gallery">${galleryImgs}</div>
          </details>
          <button class="btn-wipe" onclick="wipeJob('${j.job_id}')">🧹 Clear Job from Kaggle RAM</button>
        `;
      } else if (j.status === 'error') {
        actionsHtml = `<button class="btn-wipe" onclick="wipeJob('${j.job_id}')">✖ Remove Failed Job from RAM</button>`;
      }

      return `
        <div class="job-card" style="text-align:left;">
          <div class="job-header">
            <span class="job-title">📁 ${j.chapter_name} (${j.page_count} pages)</span>
            <span class="job-badge">${j.status.toUpperCase()}</span>
          </div>
          <div style="font-size:0.78rem;color:#cbd5e1;">${j.phase}</div>
          <div class="prog-bg"><div class="prog-fill" style="width:${j.progress}%;"></div></div>
          <div class="log-line">${j.last_log || 'Waiting in queue...'}</div>
          ${actionsHtml}
        </div>
      `;
    }).join('');
  } catch (e) {}
}

setInterval(refreshJobs, 2500);
refreshJobs();
</script>
</body>
</html>
"""

# ─────────────────────────────────────────────────────────────
# FLASK API ROUTES (PURE RAM)
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
            "created_at": time.time()
        }

    JOB_QUEUE.put(job_id)
    print(f"📥 [Mobile Upload -> RAM] Queued '{chapter_name}' ({len(ram_pages)} pages) as Job #{job_id}", flush=True)
    return jsonify({"job_id": job_id, "page_count": len(ram_pages)})


@app.route("/api/jobs", methods=["GET"])
def api_list_jobs():
    with JOBS_LOCK:
        jobs_list = []
        for jid, j in sorted(JOBS_RAM.items(), key=lambda item: item[1]["created_at"], reverse=True):
            jobs_list.append({
                "job_id": jid,
                "chapter_name": j["chapter_name"],
                "page_count": j["page_count"],
                "status": j["status"],
                "phase": j["phase"],
                "progress": j["progress"],
                "last_log": j["last_log"],
                "out_pages": j["out_pages"] if j["status"] == "done" else []
            })
    return jsonify({"jobs": jobs_list})


@app.route("/api/download_zip/<job_id>", methods=["GET"])
def api_download_zip(job_id):
    with JOBS_LOCK:
        job = JOBS_RAM.get(job_id)
        if not job or not job.get("zip_bytes"):
            return "Job not found or already cleared from RAM", 404
        zip_bytes = job["zip_bytes"]
        dl_name = f"{job['chapter_name']}_Translated.zip"

    # Stream directly from RAM to Mobile
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