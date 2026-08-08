from llama_cpp import Llama
import time
import re
import gc

# --- 1. SETUP & GLOBAL DEFINITIONS ---
MODEL_PATH = "/kaggle/input/datasets/rohith7boi/models-for-manga/models/Huihui-Qwen3-14B-abliterated-v2.Q5_K_S.gguf"

llm = None


# --- 2. THE MAIN TRANSLATION FUNCTION ---
def translate_manga(text: str, target_lang: str = "en", source_lang: str = "auto") -> str:
    global llm

    # --- LAZY LOADER (loads ONLY the first time) ---
    if llm is None:
        print("    🚀 Loading 14B Local AI Model into GPU...")
        start_time = time.time()
        try:
            llm = Llama(
                model_path=MODEL_PATH,
                n_gpu_layers=-1,
                n_ctx=2048,  # 🎯 Reduced context window to save VRAM
                verbose=False
            )
            print(f"    ✅ Model Loaded Successfully in {round(time.time() - start_time, 2)} seconds!\n")
        except Exception as e:
            print(f"    ❌ CRITICAL ERROR: Could not load model. Details: {e}")
            return text

    # A. INPUT VALIDATION
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        return ""
    if text.strip() in ["...", "．．．", "Running", "Loading", "呵"] or len(text.strip()) < 2:
        return ""

    # B. SYSTEM PROMPT (Cleaned up rules since we blocked thinking structurally)
    system_prompt = (
        "You are an expert translator for Japanese Manga.\n"
        "Task: Translate the input Japanese text into natural, idiomatic English dialogue.\n"
        "CRITICAL RULES:\n"
        "1. Output ONLY the raw English translation. Do NOT include notes, conversational responses, or explanations.\n"
        "2. OCR ERROR CORRECTION: Silently correct typos using surrounding context.\n"
        "3. NAMES: Use Romaji for Japanese names.\n"
        "4. TONE: Preserve emotion, hesitation (...), slang, and casual narration.\n"
        "5. IDIOMS & NEGATION: Never invert meanings—double-check negations like 'ない' (not)."
    )

    # Strictly capture the ChatML formatting endings
    strict_stops = [
        "<|im_end|>", "<|im_start|>", "User:", "Assistant:", "\n"
    ]

    try:
        # C. PRE-FILL FORMATTING TRICK (Bypasses the thinking block entirely)
        # We build the ChatML prompt structure manually
        formatted_prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{text}<|im_end|>\n"
            f"<|im_start|>assistant\n"  # Stopping right here forces the model to respond immediately without <think>
        )

        # D. GENERATE TRANSLATION (Using recommended Non-Thinking parameters)
        response = llm.create_completion(
            prompt=formatted_prompt,
            temperature=0.5,   # Recommended for natural dialogue style
            top_p=0.8,         # Recommended constraints
            top_k=20,          # Recommended constraints
            min_p=0.0,         # Recommended constraints
            max_tokens=256,
            stop=strict_stops
        )

        translated = response["choices"][0]["text"].strip()

        # E. SAFETY CHECK & RETRY
        if re.search(r'[\u4e00-\u9fff]', translated):
            print(f"      ⚠️ WARNING: Japanese characters detected! Retrying with strict correction...")

            retry_prompt = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\nERROR: Your previous output contained Japanese characters. \nInput: '{text}'\nOutput ONLY English:<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

            response = llm.create_completion(
                prompt=retry_prompt,
                temperature=0.1,  # Lowered slightly for strict error handling
                top_p=0.5,
                top_k=20,
                min_p=0.0,
                max_tokens=256,
                stop=strict_stops
            )
            translated = response["choices"][0]["text"].strip()

        if re.search(r'[\u4e00-\u9fff]', translated):
            print(f"      ⚠️ WARNING: Retry failed. Returning original text to avoid garbage.")
            return text

        # F. POST-PROCESS CLEANING ENGINE 
        # (Kept purely for edge-case text anomalies, but <think> logic is removed since it's obsolete)
        translated = re.sub(r'\s*\([^)]*\)$', '', translated).strip()
        translated = re.sub(r'^\([^)]*\)\s*', '', translated).strip()

        return translated

    except Exception as e:
        print(f"      ⚠️ Local Translation Error: {e}")
        return text