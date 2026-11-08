import os
import gc
import glob
import cv2
import torch
import numpy as np
from ultralytics import YOLO


def sort_boxes_reading_order(boxes: np.ndarray, page_height: int, rtl: bool = True) -> list:
    """
    Sorts bounding boxes in true comic reading order:
      1. Top-to-Bottom by horizontal panel row bands
      2. Right-to-Left (Manga, rtl=True) or Left-to-Right (Manhwa/Manhua, rtl=False)
         whenever two bubbles are on roughly the same vertical level.
    """
    if len(boxes) == 0:
        return []

    # Dynamic vertical threshold (~7% of page height) to group bubbles in the same row band
    row_threshold = max(50, int(page_height * 0.07))

    # First sort strictly by vertical center (y_center)
    box_list = list(boxes)
    box_list.sort(key=lambda b: (b[1] + b[3]) / 2.0)

    rows = []
    for box in box_list:
        y_center = (box[1] + box[3]) / 2.0
        placed = False
        for row in rows:
            # Compare against the average y_center of the current row band
            row_y_avg = sum((b[1] + b[3]) / 2.0 for b in row) / len(row)
            if abs(y_center - row_y_avg) <= row_threshold:
                row.append(box)
                placed = True
                break
        if not placed:
            rows.append([box])

    # Sort each row horizontally: Right-to-Left (-x_center) for Manga, Left-to-Right (+x_center) for Manhwa
    ordered_boxes = []
    for row in rows:
        if rtl:
            row.sort(key=lambda b: -((b[0] + b[2]) / 2.0))
        else:
            row.sort(key=lambda b: ((b[0] + b[2]) / 2.0))
        ordered_boxes.extend(row)

    return ordered_boxes


def run_yolo_ram_batch(ram_pages: list, model_path: str = "best.pt", rtl: bool = True) -> dict:
    """
    PURE RAM MODE:
    Accepts `ram_pages`: list of tuples [(filename, bgr_numpy_array), ...] held strictly in RAM.
    Detects speech bubbles with YOLOv8, crops them in RAM, and immediately frees GPU VRAM.

    Returns:
      page_map = {
          filename: [
              {
                  "crop_id": "page01_bub0",
                  "crop_img": numpy_bgr_crop,   # Held strictly in RAM!
                  "box": (x1, y1, x2, y2)
              }, ...
          ]
      }
    """
    print(f"\n🔄 Phase 1: Detecting speech bubbles in RAM ({'Manga RTL' if rtl else 'Manhwa LTR'})...", flush=True)

    if not ram_pages:
        print("❌ No images provided in RAM.", flush=True)
        return {}

    print(f"📦 Loading YOLOv8 ({model_path})...", flush=True)
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"❌ Error loading YOLO model: {e}", flush=True)
        return {}

    PADDING = 10
    page_map = {}
    device = "0" if torch.cuda.is_available() else "cpu"

    print(f"🚀 Scanning {len(ram_pages)} pages in RAM...", flush=True)

    for filename, full_img in ram_pages:
        name_no_ext = os.path.splitext(filename)[0]
        page_map[filename] = []

        if full_img is None or full_img.size == 0:
            continue

        height, width = full_img.shape[:2]

        try:
            # Run inference directly on the in-memory numpy BGR image
            results = model(full_img, verbose=False, device=device, conf=0.22, iou=0.45)

            for result in results:
                if result.boxes is None or len(result.boxes) == 0:
                    continue

                boxes = result.boxes.xyxy.cpu().numpy()
                sorted_boxes = sort_boxes_reading_order(boxes, height, rtl=rtl)

                for i, box in enumerate(sorted_boxes):
                    x1, y1, x2, y2 = map(int, box)

                    # Apply safe padding within image bounds
                    x1 = max(0, x1 - PADDING)
                    y1 = max(0, y1 - PADDING)
                    x2 = min(width, x2 + PADDING)
                    y2 = min(height, y2 + PADDING)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    # Slice copy in pure RAM (never touches disk)
                    cropped_bubble = full_img[y1:y2, x1:x2].copy()
                    crop_id = f"{name_no_ext}_bub{i}"

                    page_map[filename].append({
                        "crop_id": crop_id,
                        "crop_img": cropped_bubble,
                        "box": (x1, y1, x2, y2)
                    })

        except Exception as e:
            print(f"⚠️ Error detecting bubbles on {filename}: {e}", flush=True)

    # CRITICAL: Completely unload YOLO from GPU VRAM before llama-server starts!
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    total_bubbles = sum(len(bubbles) for bubbles in page_map.values())
    print(f"✅ Phase 1 Complete in RAM. Detected {total_bubbles} speech bubbles (YOLO VRAM freed).", flush=True)
    return page_map


# Legacy disk wrapper (kept for backward compatibility if called with folder paths)
def run_yolo_batch(input_folder: str, output_folder: str = None, model_path: str = "best.pt", rtl: bool = True) -> dict:
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(input_folder, '**', ext), recursive=True))
    files.sort()

    ram_pages = []
    for fpath in files:
        img = cv2.imread(fpath)
        if img is not None:
            ram_pages.append((os.path.basename(fpath), img))

    return run_yolo_ram_batch(ram_pages, model_path=model_path, rtl=rtl)
    