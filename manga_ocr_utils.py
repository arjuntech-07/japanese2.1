import os
import ollama
import subprocess
import time

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
MODEL_NAME = "local-qwen-vl"

# ⚠️ CHANGED: Added the direct path to your Kaggle dataset GGUF file
# Note: Check if you have a '/models/' subfolder. Based on standard Kaggle layout,
# it's usually directly in the input folder like below.
GGUF_PATH = "/kaggle/input/datasets/rohith7boi/models-for-manga/models/Qwen3-VL-8B-Instruct-abliterated-v2.0.Q8_0.gguf"

# 🧠 STRICT OCR PROMPT
OCR_PROMPT = (
    "You are an expert OCR engine for Manga and Manhua speech bubbles.\n"
    "Task: Extract the text from this cropped bubble image.\n"
    "CRITICAL RULES:\n"
    "1. READING ORDER: Detect if text is Vertical or Horizontal. Read accordingly.\n"
    "2. VERTICAL TEXT: If text is vertical, combine lines correctly (Right-to-Left columns).\n"
    "3. NOISE: Ignore background art, screen tones, or sound effects. Focus ONLY on dialogue text.\n"
    "4. STYLIZED FONTS: Infer characters even if the font is handwritten or brush-style.\n"
    "5. OUTPUT: Output ONLY the raw text. No markdown, no quotes, no explanations.\n"
    "6. LANGUAGE: Keep the original language (Chinese/Japanese/Korean). Do NOT translate."
)


def setup_ollama_model():
    """Registers the Kaggle GGUF file with Ollama so it can be used."""
    print(f"⚙️ Registering {MODEL_NAME} from {GGUF_PATH}...", flush=True)

    if not os.path.exists(GGUF_PATH):
        print(f"❌ ERROR: GGUF file not found at {GGUF_PATH}. Please double-check your Kaggle input path!", flush=True)
        return False

    try:
        # We tell Ollama to create the model directly from the GGUF path
        ollama.create(model=MODEL_NAME, modelfile=f"FROM {GGUF_PATH}")
        print(f"✅ Successfully registered {MODEL_NAME} in Ollama.", flush=True)
        return True
    except Exception as e:
        print(f"❌ Error creating model in Ollama: {e}", flush=True)
        return False


def load_ocr_model():
    """Connects to Ollama and pre-loads the model into VRAM."""

    print(f"📦 Loading {MODEL_NAME} into VRAM via Ollama...", flush=True)
    try:
        ollama.chat(model=MODEL_NAME, keep_alive=-1)
        print("✅ OCR Model Loaded.", flush=True)
        return True
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}", flush=True)
        return False


def unload_ocr_model():
    """Forcefully unloads OCR model from VRAM to make room for Translation."""
    print("🧹 Forcefully unloading OCR model from VRAM...", flush=True)
    try:
        subprocess.run(["ollama", "stop", MODEL_NAME], check=True)
        time.sleep(2)  # Brief pause to let GPU clear memory
        print("✅ OCR Model Unloaded.", flush=True)
    except Exception as e:
        print(f"⚠️ Warning: Could not stop model cleanly: {e}", flush=True)


def run_vl_ocr_batch(crop_paths):
    """Sends crops to Ollama and returns a dictionary of filename to text."""
    if not load_ocr_model():
        return {}

    print(f"📖 Sending {len(crop_paths)} bubbles to Qwen-VL...", flush=True)
    results = {}

    for i, img_path in enumerate(crop_paths):
        if not os.path.exists(img_path):
            continue

        filename = os.path.basename(img_path)
        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[{
                    'role': 'user',
                    'content': OCR_PROMPT,
                    'images': [img_path]
                }],
                keep_alive=-1,
                options={
                    "temperature": 0.0,  # Zero temperature ensures it doesn't get creative
                    "num_predict": 256
                }
            )

            text = response['message']['content'].strip()
            text = text.replace("```", "").strip()
            results[filename] = text

            if (i + 1) % 5 == 0:
                print(f"  Processed {i + 1}/{len(crop_paths)} | {text[:15]}...", flush=True)

        except Exception as e:
            print(f"  ⚠️ Error on {filename}: {e}", flush=True)
            results[filename] = ""

    print("✅ OCR Batch Complete.", flush=True)

    # ALWAYS unload when done so GPU 0 is ready for IOPaint
    unload_ocr_model()

    return results