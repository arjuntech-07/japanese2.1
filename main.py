# main.py (Japanese Hybrid Project)
"""
Japanese Manga Translation Pipeline
Hybrid: Chinese Engineering + Japanese Models
"""
import os
import cv2
import numpy as np
from PIL import Image
import transformers
import logging
import gc
import torch
import re

# Suppress Hugging Face model config dumps
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()

# Optional: silence other noisy loggers
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("manga_ocr").setLevel(logging.WARNING)

# ─────────────────────────────────────────────
# IMPORTS (Updated for Hybrid Project)
# ─────────────────────────────────────────────
from crop_engine import run_yolo_batch  # ← Chinese (with RTL fix)
from manga_ocr_utils import run_vl_ocr_batch  # ← Upgraded to Qwen-VL
from text_renderer import add_text  # ← Chinese (with thick outline)
from translate_manga import translate_chunk  # ← NEW: Chunked Ollama API
from cleaner import process_contour  # ← Chinese (fixes threshold bug)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
INPUT_FOLDER = "/kaggle/input/datasets/rohith7boi/test1-0"
CROP_DIR = "crops1"
OUTPUT_FOLDER = "/kaggle/working/translated_manga"
MODEL_PATH = "/kaggle/input/models/rohith7boi/yolov8/pytorch/default/1/best.pt"


# ─────────────────────────────────────────────
# PHASE 3: Page Assembly (Translate + Clean + Render)
# ─────────────────────────────────────────────
def process_page_assembly(original_image_path, bubble_data_list, ocr_results):
    """
    Reconstructs the page: Bulk Translates FIRST, then Replaces Text.
    """
    # 1. Read Full Image (OpenCV for consistent BGR handling)
    full_img = cv2.imread(original_image_path)
    if full_img is None:
        print(f"  ❌ Failed to load: {original_image_path}")
        return

    filename = os.path.basename(original_image_path)
    print(f"\n  📄 Processing: {filename}")

    # ==========================================
    # STEP 1: CHUNKED BULK TRANSLATION (GPU 1 Active)
    # ==========================================
    page_translations = {}  # Store translations here
    valid_bubbles = []
    page_texts = []
    
    # 1. Gather all valid text for the page first
    for bubble in bubble_data_list:
        crop_filename = os.path.basename(bubble["crop_path"])
        original_text = ocr_results.get(crop_filename, "").strip()
        
        if original_text:
            valid_bubbles.append(bubble)
            page_texts.append(original_text)

    # 2. Process in safe chunks of 10 bubbles
    # This gives the model context, but prevents JSON truncation on large pages
    CHUNK_SIZE = 10 
    
    if page_texts:
        print(f"   🌐 Translating {len(page_texts)} bubbles in chunks of {CHUNK_SIZE}...")
        
        for i in range(0, len(page_texts), CHUNK_SIZE):
            chunk_texts = page_texts[i : i + CHUNK_SIZE]
            chunk_bubbles = valid_bubbles[i : i + CHUNK_SIZE]
            
            # Get translations for this specific chunk
            chunk_results = translate_chunk(chunk_texts)
            
            # Map results back to filenames
            for local_idx, bubble in enumerate(chunk_bubbles):
                crop_filename = os.path.basename(bubble["crop_path"])
                english = chunk_results.get(str(local_idx), "")
                
                # Final sanity check: If it returned text but it's just cursive/corrupt unicode text, wipe it
                if english and (bool(re.search(r'[\u1d400-\u1d7ff]', english)) or "<think>" in english):
                    english = ""
                    
                page_translations[crop_filename] = english
                
                # Terminal logging
                clean_orig = chunk_texts[local_idx].replace('\n', ' ')
                clean_trans = english.replace('\n', ' ') if english else "[Skipped/Empty]"
                print(f"      [JP] : {clean_orig}")
                print(f"      [EN] : {clean_trans}")
                
    print("      " + "-" * 40)

    # ==========================================
    # STEP 2: RENDER & INPAINT (GPU 0 Active)
    # ==========================================
    for bubble in bubble_data_list:
        crop_filename = os.path.basename(bubble["crop_path"])

        # FIX: Default to empty string instead of None so we don't skip the cleaning phase
        english_text = page_translations.get(crop_filename, "")

        x1, y1, x2, y2 = bubble["box"]
        bubble_img = cv2.imread(bubble["crop_path"])

        if bubble_img is None:
            continue

        # FIX: ALWAYS clean the bubble so old Japanese text is erased!
        cleaned_bubble, text_contour = process_contour(bubble_img)

        # FIX: Only attempt to render text if there is actually valid text to draw
        if english_text and english_text.strip():
            add_text(cleaned_bubble, english_text)

        # Resize and paste back to original box size
        target_width = x2 - x1
        target_height = y2 - y1

        if cleaned_bubble.shape[0] != target_height or cleaned_bubble.shape[1] != target_width:
            cleaned_bubble = cv2.resize(
                cleaned_bubble,
                (target_width, target_height),
                interpolation=cv2.INTER_AREA  # prevents blur
            )

        full_img[y1:y2, x1:x2] = cleaned_bubble

    # ==========================================
    # STEP 3: SAVE FINAL IMAGE
    # ==========================================
    rel_path = os.path.relpath(original_image_path, INPUT_FOLDER)

    # Keep Japanese project's "_processed" suffix style
    rel_path_parts = rel_path.split(os.sep)
    if len(rel_path_parts) > 1:
        rel_path_parts[-2] = rel_path_parts[-2] + "_processed"
        save_subfolder = os.path.join(OUTPUT_FOLDER, *rel_path_parts[:-1])
    else:
        save_subfolder = OUTPUT_FOLDER

    os.makedirs(save_subfolder, exist_ok=True)

    save_filename = f"translated_{os.path.basename(original_image_path)}"
    save_path = os.path.join(save_subfolder, save_filename)

    cv2.imwrite(save_path, full_img)
    print(f"   ✨ Saved to: {save_path}")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print(" 🇯🇵 Japanese Manga Translation Pipeline (Hybrid)")
    print("=" * 60)

    # 1. Setup Folders (Never try to create Kaggle input directories)
    for folder in [CROP_DIR, OUTPUT_FOLDER]:
        os.makedirs(folder, exist_ok=True)

    # 2. Phase 1: Run Cropping (Batch process all pages)
    page_map = run_yolo_batch(INPUT_FOLDER, CROP_DIR, MODEL_PATH)
    if not page_map:
        print("❌ No bubbles detected. Exiting.")
        return

    # 3. Phase 2: Run OCR (Batch process all crops)
    all_crops = [b["crop_path"] for bubbles in page_map.values() for b in bubbles]
    if not all_crops:
        print("❌ No crops found. Exiting.")
        return

    ocr_results = run_vl_ocr_batch(all_crops)  # <--- QWEN-VL FUNCTION

    # 4. Phase 3: Assembly & Inpainting (Translate + Clean + Render)
    print("\n" + "=" * 60)
    print("=== PHASE 3: TRANSLATING & REPLACING ===")
    print("=" * 60)

    # ✅ NEW: WARM-UP THE 27B TRANSLATION MODEL
    # This forces Ollama to load the 20GB model into VRAM before we start processing pages
    print("🔥 Warming up Qwen 3.6 27B (Loading into VRAM, please wait ~60-120s)...")
    try:
        import requests
        from translate_manga import MODEL_NAME, OLLAMA_URL
        
        warmup_payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "ping"}],
            "stream": False,
            "keep_alive": "24h",
            "options": {"num_ctx": 8192} # Pre-allocates the KV Cache
        }
        # Give it 10 minutes to load the weights from disk
        requests.post(OLLAMA_URL, json=warmup_payload, timeout=600)
        print("✅ Translation Model loaded into VRAM and ready!\n")
    except Exception as e:
        print(f"⚠️ Warmup ping failed, but we will try anyway: {e}\n")

    # Use sorted() to ensure pages process in alphabetical/numerical order (01, 02, 03...)
    for original_page, bubble_data in sorted(page_map.items()):
        process_page_assembly(original_page, bubble_data, ocr_results)

        # --- CRITICAL FIX: FLUSH GPU MEMORY ---
        gc.collect()  # Force Python to delete unused variables
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Force PyTorch/YOLO to empty VRAM

    print("\n" + "=" * 60)
    print(f"🎉 Success! Check the '{OUTPUT_FOLDER}' folder.")
    print("=" * 60)


if __name__ == "__main__":
    main()