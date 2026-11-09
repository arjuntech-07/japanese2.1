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
    UPGRADED: Uses median bubble height instead of total page_height so 10,000px+
    tall Korean/Chinese webtoon strips never group vertically distant bubbles into the same row.
    """
    if len(boxes) == 0:
        return []

    box_list = list(boxes)

    # Adaptive row threshold based on actual bubble heights on the page (clamped between 40px and 130px)
    heights = [max(1.0, float(b[3] - b[1])) for b in box_list]
    median_h = float(np.median(heights)) if heights else 80.0
    row_threshold = int(min(max(40.0, median_h * 0.60), 130.0))

    # Sort primarily by vertical center (Top-to-Bottom)
    box_list.sort(key=lambda b: (b[1] + b[3]) / 2.0)

    rows = []

    for box in box_list:
        y_center = (box[1] + box[3]) / 2.0
        placed = False

        for row in rows:
            row_y_avg = sum((b[1] + b[3]) / 2.0 for b in row) / len(row)

            if abs(y_center - row_y_avg) <= row_threshold:
                row.append(box)
                placed = True
                break

        if not placed:
            rows.append([box])

    ordered_boxes = []

    for row in rows:
        if rtl:
            # Manga: Right-to-Left
            row.sort(key=lambda b: -((b[0] + b[2]) / 2.0))
        else:
            # Manhwa / Manhua: Left-to-Right
            row.sort(key=lambda b: ((b[0] + b[2]) / 2.0))

        ordered_boxes.extend(row)

    return ordered_boxes


def _merge_overlapping_boxes(boxes: list, iou_thresh: float = 0.35) -> np.ndarray:
    """
    Merges duplicate or seam-split bounding boxes from overlapping vertical webtoon slices.
    """
    if not boxes:
        return np.empty((0, 4), dtype=np.float32)

    arr = np.array(boxes, dtype=np.float32)
    merged = True

    while merged and len(arr) > 1:
        merged = False
        new_boxes = []
        used = [False] * len(arr)

        for i in range(len(arr)):
            if used[i]:
                continue
            bx1, by1, bx2, by2 = arr[i]
            area_i = max(1.0, (bx2 - bx1) * (by2 - by1))

            for j in range(i + 1, len(arr)):
                if used[j]:
                    continue
                gx1, gy1, gx2, gy2 = arr[j]
                area_j = max(1.0, (gx2 - gx1) * (gy2 - gy1))

                ix1, iy1 = max(bx1, gx1), max(by1, gy1)
                ix2, iy2 = min(bx2, gx2), min(by2, gy2)
                iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
                inter = iw * ih

                if inter > 0:
                    union = area_i + area_j - inter
                    min_area = min(area_i, area_j)
                    # Merge if high IoU OR one box is mostly contained inside the other across a slice seam
                    if (inter / union) >= iou_thresh or (inter / min_area) >= 0.65:
                        bx1, by1 = min(bx1, gx1), min(by1, gy1)
                        bx2, by2 = max(bx2, gx2), max(by2, gy2)
                        area_i = max(1.0, (bx2 - bx1) * (by2 - by1))
                        used[j] = True
                        merged = True

            used[i] = True
            new_boxes.append([bx1, by1, bx2, by2])

        arr = np.array(new_boxes, dtype=np.float32)

    return arr


def _detect_boxes_on_image(
    model: YOLO,
    full_img: np.ndarray,
    device: str,
    conf: float,
    iou: float
) -> np.ndarray:
    """
    Runs YOLOv8 on a single page.
    - Standard Manga/Comic page (height <= width * 2.2): Single fast inference at imgsz=1024.
    - Tall Webtoon / Manhwa / Manhua strip (height > width * 2.2): Automatically slices
      vertically in RAM with 25% overlap so bubbles are never squashed, then merges seam boxes.
    """
    height, width = full_img.shape[:2]

    # Case 1: Standard Page Aspect Ratio
    if height <= int(width * 2.2):
        results = model(
            full_img,
            verbose=False,
            device=device,
            conf=conf,
            iou=iou,
            imgsz=1024
        )
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                return result.boxes.xyxy.cpu().numpy()
        return np.empty((0, 4), dtype=np.float32)

    # Case 2: Tall Vertical Strip (Manhwa / Manhua Webtoon)
    slice_h = max(1024, int(width * 1.6))
    overlap = int(slice_h * 0.25)
    step = max(256, slice_h - overlap)

    collected_boxes = []
    y_start = 0

    while y_start < height:
        y_end = min(height, y_start + slice_h)
        # Ensure the final slice still has enough height context
        if (y_end - y_start) < int(slice_h * 0.5) and height > slice_h:
            y_start = max(0, height - slice_h)
            y_end = height

        slice_img = full_img[y_start:y_end, 0:width]
        results = model(
            slice_img,
            verbose=False,
            device=device,
            conf=conf,
            iou=iou,
            imgsz=1024
        )

        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                slice_boxes = result.boxes.xyxy.cpu().numpy()
                for b in slice_boxes:
                    collected_boxes.append([
                        float(b[0]),
                        float(b[1] + y_start),
                        float(b[2]),
                        float(b[3] + y_start)
                    ])

        if y_end >= height:
            break
        y_start += step

    return _merge_overlapping_boxes(collected_boxes, iou_thresh=iou)


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
        f"({'Manga RTL' if rtl else 'Manhwa/Manhua LTR'})...",
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
                boxes = _detect_boxes_on_image(
                    model=model,
                    full_img=full_img,
                    device=device,
                    conf=CONFIDENCE,
                    iou=IOU_THRESHOLD
                )

                if boxes is None or len(boxes) == 0:
                    continue

                # Sort detections into Manga RTL / Manhwa LTR order.
                sorted_boxes = sort_boxes_reading_order(
                    boxes,
                    height,
                    rtl=rtl
                )

                for i, box in enumerate(sorted_boxes):
                    x1, y1, x2, y2 = map(int, box)

                    # Skip tiny false-positive detections (minimum 24x24px).
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
    
