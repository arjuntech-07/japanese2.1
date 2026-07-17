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
                n_ctx=2048,  # 🎯 Reduced context window since we don't have massive few-shots anymore (saves VRAM)
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

    # B. SYSTEM PROMPT (Clean, direct instructions for a 14B model)
    system_prompt = (
        "You are an expert translator for Japanese Manga.\n"
        "Task: Translate the input Japanese text into natural, idiomatic English dialogue.\n"
        "CRITICAL RULES:\n"
        "1. Output ONLY the raw English translation. Do NOT include notes, conversational responses, meta-commentary, or explanations.\n"
        "2. Do NOT use or hallucinate markdown code blocks, think tags, or tool execution structures.\n"
        "3. OCR ERROR CORRECTION: Silently correct typos using surrounding context.\n"
        "4. NAMES: Use Romaji for Japanese names.\n"
        "5. TONE: Preserve emotion, hesitation (...), slang, and casual narration.\n"
        "6. IDIOMS & NEGATION: Never invert meanings—double-check negations like 'ない' (not)."
    )

    # FIX 1: Removed </think>, <tool_response>, and \n\n so the model actually finishes generating!
    strict_stops = [
        "<|im_end|>", "<|im_start|>", "User:", "Assistant:", "(I'm", "(I am"
    ]

    try:
        # C. BUILD THE CHAT LOG (Zero-Shot structure, perfectly suited for 14B)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]

        # D. GENERATE TRANSLATION
        response = llm.create_chat_completion(
            messages=messages,
            temperature=0.05,
            top_p=0.9,
            max_tokens=96,
            stop=strict_stops
        )

        translated = response["choices"][0]["message"]["content"].strip()

        # E. SAFETY CHECK & RETRY
        if re.search(r'[\u4e00-\u9fff]', translated):
            print(f"      ⚠️ WARNING: Japanese characters detected! Retrying with strict correction...")

            retry_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",
                 "content": f"ERROR: Your previous output contained Japanese characters. \nInput: '{text}'\nOutput ONLY English:"}
            ]

            response = llm.create_chat_completion(
                messages=retry_messages,
                temperature=0.01,
                top_p=0.5,
                max_tokens=96,
                stop=strict_stops
            )
            translated = response["choices"][0]["message"]["content"].strip()

        if re.search(r'[\u4e00-\u9fff]', translated):
            print(f"      ⚠️ WARNING: Retry failed. Returning original text to avoid garbage.")
            return text

        # F. POST-PROCESS CLEANING ENGINE (FIX 2: Safe logic implemented here)
        if "</think>" in translated:
            parts = translated.split("</think>")
            text_after = parts[-1].strip()

            if text_after:
                translated = text_after
            else:
                translated = parts[0].replace("<think>", "").replace("<think>\n", "").strip()

        # Clean up any leftover opening tag if the model didn't close it
        translated = translated.replace("<think>", "").strip()
        translated = translated.replace("<tool_response>", "").strip()

        # Clean up accidental leaked descriptions inside brackets
        translated = re.sub(r'\s*\([^)]*\)$', '', translated).strip()
        translated = re.sub(r'^\([^)]*\)\s*', '', translated).strip()

        return translated

    except Exception as e:
        print(f"      ⚠️ Local Translation Error: {e}")
        return text