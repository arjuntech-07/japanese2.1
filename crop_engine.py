import os
import glob
from ultralytics import YOLO
from PIL import Image


def get_all_images(input_folder):
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(input_folder, '**', ext), recursive=True))
    return files


def run_yolo_batch(input_folder, output_folder, model_path="best.pt"):
    print(f"\n🔄 Phase 1: Scanning images & Saving crops...")

    # Create output folder if not exists
    os.makedirs(output_folder, exist_ok=True)

    image_files = get_all_images(input_folder)
    if not image_files:
        print("❌ No images found in 'input' folder.")
        return {}

    print(f"📦 Loading YOLO ({model_path})...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"❌ Error loading YOLO model: {e}")
        return {}

    PADDING = 10
    # Dictionary: { "full_page_path": [ {crop_path, box}, ... ] }
    page_map = {}

    print(f"🚀 Processing {len(image_files)} pages...")

    for img_path in image_files:
        filename = os.path.basename(img_path)
        name_no_ext = os.path.splitext(filename)[0]
        page_map[img_path] = []

        try:
            results = model(img_path, verbose=False)

            # Open image once to crop multiple times
            with Image.open(img_path) as original_img:
                width, height = original_img.size

                for result in results:
                    # Sort by Y position (top to bottom reading order)
                    # ✅ NEW (Japanese RTL - Right to Left, then Top to Bottom)
                    boxes = result.boxes.xyxy.cpu().numpy()
                    sorted_boxes = sorted(boxes, key=lambda box: (-box[0], box[1]))

                    for i, box in enumerate(sorted_boxes):
                        x1, y1, x2, y2 = map(int, box)

                        # 1. Apply Padding
                        x1 = max(0, x1 - PADDING)
                        y1 = max(0, y1 - PADDING)
                        x2 = min(width, x2 + PADDING)
                        y2 = min(height, y2 + PADDING)

                        cropped_bubble = original_img.crop((x1, y1, x2, y2))

                        # =======================================================
                        # 🔍 SMART ZOOM (High-Quality Upscaling)
                        # This saves Paddle from "Blur Death" and helps Qwen see clearer!
                        # =======================================================
                        crop_w, crop_h = cropped_bubble.size

                        # Only upscale if the bubble is tiny (smaller than 300px)
                        if crop_w < 300 or crop_h < 300:
                            scale_factor = 3  # Make it 3x bigger
                            new_w = int(crop_w * scale_factor)
                            new_h = int(crop_h * scale_factor)

                            # LANCZOS keeps edges sharp.
                            if hasattr(Image, 'Resampling'):
                                cropped_bubble = cropped_bubble.resize((new_w, new_h), Image.Resampling.LANCZOS)
                            else:
                                cropped_bubble = cropped_bubble.resize((new_w, new_h), Image.LANCZOS)
                        # =======================================================

                        # Save Crop
                        save_name = f"{name_no_ext}_bub{i}.png"
                        save_path = os.path.join(output_folder, save_name)
                        cropped_bubble.save(save_path, "PNG")

                        # Store data for later
                        page_map[img_path].append({
                            "crop_path": save_path,
                            "box": (x1, y1, x2, y2)
                        })

        except Exception as e:
            print(f"⚠️ Error processing {filename}: {e}")

    total_bubbles = sum(len(bubbles) for bubbles in page_map.values())
    print(f"✅ Phase 1 Done. Found {total_bubbles} bubbles.")
    return page_map