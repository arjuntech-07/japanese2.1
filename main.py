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
from translate_manga import translate_manga  # ← Japanese (Sugoi)
from cleaner import process_contour  # ← Chinese (fixes threshold bug)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
# Change these lines at the top of main.py
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
    print(f"\n   📄 Processing: {filename}")

    # ==========================================
    # STEP 1: BULK TRANSLATION (GPU 1 Active)
    # ==========================================
    page_translations = {}  # Store translations here

    for bubble in bubble_data_list:
        crop_filename = os.path.basename(bubble["crop_path"])
        original_text = ocr_results.get(crop_filename, "")

        # Skip if no text found
        if not original_text or not original_text.strip():
            continue

        # Run translation
        english = translate_manga(original_text, target_lang="en")

        if english and english.strip():
            # Clean up reasoning tags from Qwen 3
            if "</think>" in english:
                english = english.split("</think>")[-1].strip()

            page_translations[crop_filename] = english

            # Print to terminal
            clean_orig = original_text.replace('\n', ' ')
            clean_trans = english.replace('\n', ' ')
            print(f"      [JP] : {clean_orig}")
            print(f"      [EN] : {clean_trans}")

    print("      " + "-" * 40)

    # ==========================================
    # STEP 2: RENDER & INPAINT (GPU 0 Active)
    # ==========================================
    # Now loop through the bubbles again, ONLY for drawing
    for bubble in bubble_data_list:
        crop_filename = os.path.basename(bubble["crop_path"])

        # Grab the translation completed in Step 1
        english_text = page_translations.get(crop_filename)

        if not english_text:
            continue

        x1, y1, x2, y2 = bubble["box"]
        bubble_img = cv2.imread(bubble["crop_path"])

        if bubble_img is None:
            continue

        # Clean (IOPaint)
        cleaned_bubble, text_contour = process_contour(bubble_img)

        # Render Text
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
    # Preserve subfolder structure
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
    print("🇯🇵 Japanese Manga Translation Pipeline (Hybrid)")
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

    ocr_results = run_vl_ocr_batch(all_crops)  # <--- NEW QWEN-VL FUNCTION

    # 4. Phase 3: Assembly & Inpainting (Translate + Clean + Render)
    print("\n" + "=" * 60)
    print("=== PHASE 3: TRANSLATING & REPLACING ===")
    print("=" * 60)

    for original_page, bubble_data in page_map.items():
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