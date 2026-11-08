import os
import sys
import time
import base64
import subprocess
import requests
import cv2
import numpy as np
from huggingface_hub import hf_hub_download

# ==========================================================
# ⚙️ CONFIGURATION: QWEN 3.5 9B VISION OCR (PURE RAM INPUT)
# ==========================================================
MODEL_DIR = "/tmp/models"
OCR_REPO_ID = "HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive"
OCR_MODEL_FILE = "Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf"
OCR_MMPROJ_FILE = "mmproj-Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-BF16.gguf"

OCR_MODEL_PATH = os.path.join(MODEL_DIR, OCR_MODEL_FILE)
OCR_MMPROJ_PATH = os.path.join(MODEL_DIR, OCR_MMPROJ_FILE)

SERVER_PORT = 8080
LLAMA_CHAT_URL = f"http://127.0.0.1:{SERVER_PORT}/v1/chat/completions"
LLAMA_HEALTH_URL = f"http://127.0.0.1:{SERVER_PORT}/health"

# Logs stored in /dev/shm (Pure RAM) so Kaggle disk is never touched
RAM_LOG_DIR = "/dev/shm/manga_ram"
os.makedirs(RAM_LOG_DIR, exist_ok=True)
OCR_LOG_PATH = os.path.join(RAM_LOG_DIR, "llama_ocr.log")

_ocr_process = None
_ocr_log_file = None

# 🧠 STRICT OCR PROMPT
OCR_PROMPT = (
    "You are an expert OCR engine for Manga, Manhwa, and Manhua speech bubbles.\n"
    "Task: Extract the exact text from this cropped speech bubble image.\n"
    "CRITICAL RULES:\n"
    "1. READING ORDER: Detect if text is Vertical or Horizontal. Read accordingly.\n"
    "2. VERTICAL TEXT: If text is vertical, combine columns Right-to-Left.\n"
    "3. NOISE: Ignore background art, screen tones, or borders. Focus ONLY on dialogue text.\n"
    "4. STYLIZED FONTS: Accurately infer characters even if handwritten or brush-style.\n"
    "5. OUTPUT: Output ONLY the raw extracted text. No markdown, no quotes, no explanations.\n"
    "6. LANGUAGE: Keep the original language (Japanese/Chinese/Korean). Do NOT translate."
)


def ensure_llama_server_binary():
    """Ensures the high-performance llama.cpp CUDA binary is installed."""
    llama_bin = os.path.expanduser("~/.llama-app/llama")
    if os.path.exists("./llama-server"):
        return ["./llama-server"]
    if os.path.exists(llama_bin):
        return [llama_bin, "serve"]

    print("📦 Installing llama.cpp CUDA engine...", flush=True)
    subprocess.run("curl -LsSf https://llama.app/install.sh | sh", shell=True, check=False)
    if os.path.exists(llama_bin):
        return [llama_bin, "serve"]
    return ["llama-server"]


def ensure_ocr_model_downloaded():
    """Downloads Qwen3.5-9B-Uncensored Q8_0 and its BF16 mmproj into /tmp/models if missing."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    hf_token = os.environ.get("HF_TOKEN", None)

    # 1. Download Main 9B Q8_0 GGUF (~9.5 GB)
    if os.path.exists(OCR_MODEL_PATH) and os.path.getsize(OCR_MODEL_PATH) > 8 * 1e9:
        print(f"✅ OCR Model ready: {OCR_MODEL_FILE}", flush=True)
    else:
        print(f"📥 Downloading OCR Model ({OCR_MODEL_FILE}) to {MODEL_DIR}...", flush=True)
        hf_hub_download(
            repo_id=OCR_REPO_ID,
            filename=OCR_MODEL_FILE,
            local_dir=MODEL_DIR,
            token=hf_token
        )

    # 2. Download Vision Projector mmproj (~800+ MB)
    if os.path.exists(OCR_MMPROJ_PATH) and os.path.getsize(OCR_MMPROJ_PATH) > 500 * 1e6:
        print(f"✅ OCR Vision Projector ready: {OCR_MMPROJ_FILE}", flush=True)
    else:
        print(f"📥 Downloading OCR Vision Projector ({OCR_MMPROJ_FILE}) to {MODEL_DIR}...", flush=True)
        hf_hub_download(
            repo_id=OCR_REPO_ID,
            filename=OCR_MMPROJ_FILE,
            local_dir=MODEL_DIR,
            token=hf_token
        )


def start_ocr_server():
    """Starts llama-server with Qwen3.5-9B + mmproj across Dual GPUs."""
    global _ocr_process, _ocr_log_file

    # Clean up any existing llama-server holding VRAM
    stop_ocr_server()
    ensure_ocr_model_downloaded()
    server_exec = ensure_llama_server_binary()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    _ocr_log_file = open(OCR_LOG_PATH, "w", buffering=1)

    cmd = server_exec + [
        "--model", OCR_MODEL_PATH,
        "--mmproj", OCR_MMPROJ_PATH,
        "--host", "127.0.0.1",
        "--port", str(SERVER_PORT),
        "--n-gpu-layers", "99",
        "--ctx-size", "8192",
        "--no-context-shift",
        "--flash-attn", "on",
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0",
        "--tensor-split", "15,15",
        "--reasoning-format", "none",
        "--threads", "4",
        "--parallel", "1",
        "--cont-batching"
    ]

    print(f"🚀 Starting Qwen3.5-9B Vision OCR Server on port {SERVER_PORT}...", flush=True)
    _ocr_process = subprocess.Popen(
        cmd,
        stdout=_ocr_log_file,
        stderr=subprocess.STDOUT,
        text=True
    )

    start_time = time.time()
    while time.time() - start_time < 240:
        if _ocr_process.poll() is not None:
            print("❌ OCR llama-server terminated unexpectedly! Last log lines:", flush=True)
            with open(OCR_LOG_PATH, "r") as f:
                print("".join(f.readlines()[-25:]), flush=True)
            return False
        try:
            res = requests.get(LLAMA_HEALTH_URL, timeout=1)
            if res.status_code == 200:
                print("✅ Qwen3.5-9B Vision OCR Server is ONLINE in VRAM!", flush=True)
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1.5)

    print("❌ Timed out waiting for OCR server.", flush=True)
    stop_ocr_server()
    return False


def stop_ocr_server():
    """Forcefully stops the OCR llama-server so 100% of VRAM is freed for Qwen3.8-27B."""
    global _ocr_process, _ocr_log_file
    if _ocr_process is not None:
        try:
            _ocr_process.terminate()
            _ocr_process.wait(timeout=5)
        except Exception:
            pass
        _ocr_process = None

    if _ocr_log_file is not None:
        try:
            _ocr_log_file.close()
        except Exception:
            pass
        _ocr_log_file = None

    subprocess.run("pkill -9 -f llama-server 2>/dev/null || true", shell=True)
    time.sleep(2)


def encode_crop_to_base64_ram(crop_input) -> str:
    """
    Converts an in-memory numpy BGR image (or fallback path) to a Base64 PNG string in pure RAM.
    Includes Smart 2x/3x Upscaling in RAM ONLY for small bubbles (<300px) so OCR sees crisp text
    without altering the original bubble dimensions used for inpainting/rendering!
    """
    if isinstance(crop_input, str):
        img = cv2.imread(crop_input)
    else:
        img = crop_input

    if img is None or img.size == 0:
        return ""

    h, w = img.shape[:2]
    if w < 300 or h < 300:
        scale = 3 if max(w, h) < 150 else 2
        img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)

    success, buffer = cv2.imencode(".png", img)
    if not success:
        return ""
    return base64.b64encode(buffer).decode("utf-8")


def strip_thinking(text: str) -> str:
    """Removes any <think>...</think> blocks if emitted."""
    import re
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text.replace("```", "").strip()


def run_vl_ocr_batch(crop_items):
    """
    Runs Vision OCR on a list/dict of crops in pure RAM.
    Accepts either:
      - A dict mapping {crop_id: numpy_bgr_image} (Pure RAM mode)
      - Or a list of file paths (Legacy mode)
    Returns: dict mapping {crop_id: extracted_text}
    """
    if not crop_items:
        return {}

    if not start_ocr_server():
        return {}

    # Normalize input into {crop_id: image_or_path}
    if isinstance(crop_items, list):
        items_dict = {os.path.basename(p): p for p in crop_items}
    else:
        items_dict = crop_items

    print(f"📖 Sending {len(items_dict)} speech bubbles from RAM to Qwen3.5-9B Vision...", flush=True)
    results = {}

    for i, (crop_id, crop_data) in enumerate(items_dict.items()):
        b64_png = encode_crop_to_base64_ram(crop_data)
        if not b64_png:
            results[crop_id] = ""
            continue

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_png}"}
                        },
                        {"type": "text", "text": OCR_PROMPT}
                    ]
                }
            ],
            "temperature": 0.0,
            "top_p": 0.9,
            "max_tokens": 256,
            "chat_template_kwargs": {"enable_thinking": False}
        }

        try:
            resp = requests.post(LLAMA_CHAT_URL, json=payload, timeout=90)
            resp.raise_for_status()
            raw_content = resp.json()["choices"][0]["message"].get("content", "")
            clean_text = strip_thinking(raw_content)
            results[crop_id] = clean_text

            if (i + 1) % 5 == 0 or (i + 1) == len(items_dict):
                preview = clean_text.replace("\n", " ")[:20]
                print(f"  👁️ OCR [{i + 1}/{len(items_dict)}] | {preview}...", flush=True)

        except Exception as e:
            print(f"  ⚠️ OCR Error on {crop_id}: {e}", flush=True)
            results[crop_id] = ""

    print("✅ Vision OCR Batch Complete. Unloading Qwen3.5-9B from VRAM...", flush=True)
    stop_ocr_server()
    return results
    