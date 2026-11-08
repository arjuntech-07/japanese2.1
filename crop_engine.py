
import os
import gc
import glob
import cv2
import torch
import numpy as np
from ultralytics import YOLO


def sort_boxes_reading_order(boxes: np.ndarray, page_height: int, rtl: bool = True) -> list:
    """
    Sort bounding boxes in comic reading order:
      1. Top-to-Bottom by horizontal panel row bands.
      2. Right-to-Left for Manga (rtl=True).
      3. Left-to-Right for Manhwa/Manhua (rtl=False).
    """
    if len(boxes) == 0:
        return []

    row_threshold = max(50, int(page_height * 0.07))

    box_list = list(boxes)
    box_list.sort(key=lambda b: (b[1] + b[3]) / 2.0)

    rows = []

    for box in box_list:
        y_center = (box[1] + box[3]) / 2.0
        placed = False

        for row in rows:
            row_y_avg = sum(
                (b[1] + b[3]) / 2.0 for b in row
            ) / len(row)

            if abs(y_center - row_y_avg) <= row_threshold:
                row.append(box)
                placed = True
                break

        if not placed:
            rows.append([box])

    ordered_boxes = []

    for row in rows:
        if rtl:
            row.sort(key=lambda b: -((b[0] + b[2]) / 2.0))
        else:
            row.sort(key=lambda b: ((b[0] + b[2]) / 2.0))

        ordered_boxes.extend(row)

    return ordered_boxes


def run_yolo_ram_batch(
    ram_pages: list,
    model_path: str = "best.pt",
    rtl: bool = True
) -> dict:
    """
    PURE RAM MODE:

    Accepts:
        ram_pages: [(filename, bgr_numpy_array), ...]

    Detects speech bubbles with YOLOv8, crops them in RAM,
    and frees GPU VRAM after detection.

    Returns:
        {
            filename: [
                {
                    "crop_id": "page01_bub0",
                    "crop_img": numpy_bgr_crop,
                    "box": (x1, y1, x2, y2)
                }
            ]
        }
    """

    print(
        f"\n🔄 Phase 1: Detecting speech bubbles in RAM "
        f"({'Manga RTL' if rtl else 'Manhwa LTR'})...",
        flush=True
    )

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
    CONFIDENCE = 0.40
    IOU_THRESHOLD = 0.45

    page_map = {}

    device = "0" if torch.cuda.is_available() else "cpu"

    print(
        f"🚀 Scanning {len(ram_pages)} pages in RAM...",
        flush=True
    )

    try:
        for filename, full_img in ram_pages:
            name_no_ext = os.path.splitext(filename)[0]
            page_map[filename] = []

            if full_img is None or full_img.size == 0:
                continue

            height, width = full_img.shape[:2]

            try:
                # FIXED: Increased YOLO confidence from 0.22 to 0.40.
                # Helps reduce false detections on facial features.
                results = model(
                    full_img,
                    verbose=False,
                    device=device,
                    conf=CONFIDENCE,
                    iou=IOU_THRESHOLD
                )

                for result in results:
                    if result.boxes is None or len(result.boxes) == 0:
                        continue

                    boxes = result.boxes.xyxy.cpu().numpy()

                    # Sort detections into Manga RTL / Manhwa LTR order.
                    sorted_boxes = sort_boxes_reading_order(
                        boxes,
                        height,
                        rtl=rtl
                    )

                    for i, box in enumerate(sorted_boxes):
                        x1, y1, x2, y2 = map(int, box)

                        # FIXED: Skip tiny false-positive detections.
                        # Minimum width: 24 pixels.
                        # Minimum height: 24 pixels.
                        if (x2 - x1) < 24 or (y2 - y1) < 24:
                            continue

                        # Apply safe padding within image boundaries.
                        x1 = max(0, x1 - PADDING)
                        y1 = max(0, y1 - PADDING)
                        x2 = min(width, x2 + PADDING)
                        y2 = min(height, y2 + PADDING)

                        if x2 <= x1 or y2 <= y1:
                            continue

                        # Crop directly into RAM.
                        cropped_bubble = full_img[
                            y1:y2,
                            x1:x2
                        ].copy()

                        crop_id = f"{name_no_ext}_bub{i}"

                        page_map[filename].append({
                            "crop_id": crop_id,
                            "crop_img": cropped_bubble,
                            "box": (x1, y1, x2, y2)
                        })

            except Exception as e:
                print(
                    f"⚠️ Error detecting bubbles on {filename}: {e}",
                    flush=True
                )

    finally:
        # Completely unload YOLO and release GPU memory.
        del model
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_bubbles = sum(
        len(bubbles) for bubbles in page_map.values()
    )

    print(
        f"✅ Phase 1 Complete in RAM. "
        f"Detected {total_bubbles} speech bubbles "
        f"(YOLO VRAM freed).",
        flush=True
    )

    return page_map


# Legacy disk wrapper (kept for backward compatibility)
def run_yolo_batch(
    input_folder: str,
    output_folder: str = None,
    model_path: str = "best.pt",
    rtl: bool = True
) -> dict:

    extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.webp"
    ]

    files = []

    for ext in extensions:
        files.extend(
            glob.glob(
                os.path.join(input_folder, "**", ext),
                recursive=True
            )
        )

    files.sort()

    ram_pages = []

    for fpath in files:
        img = cv2.imread(fpath)

        if img is not None:
            ram_pages.append(
                (os.path.basename(fpath), img)
            )

    return run_yolo_ram_batch(
        ram_pages,
        model_path=model_path,
        rtl=rtl
    )
