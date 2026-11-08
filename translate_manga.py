import os
import re
import json
import time
import subprocess
import requests
from huggingface_hub import hf_hub_download

# ==========================================================
# ⚙️ CONFIGURATION: QWEN 3.8 27B + MTP COMPANION (DUAL T4)
# ==========================================================
MODEL_DIR = "/tmp/models"
TRANS_REPO_ID = "chimingw/Qwen3.8-27B-Uncensored-OrcaRouter-GGUF"
TRANS_MODEL_FILE = "Qwen3.8-27B-Uncensored-OrcaRouter-Q6_K.gguf"
TRANS_MTP_FILE = "MTP/Qwen3.8-27B-Uncensored-OrcaRouter-MTP-Q8_0.gguf"

TRANS_MODEL_PATH = os.path.join(MODEL_DIR, TRANS_MODEL_FILE)
TRANS_MTP_PATH = os.path.join(MODEL_DIR, "MTP", "Qwen3.8-27B-Uncensored-OrcaRouter-MTP-Q8_0.gguf")

SERVER_PORT = 8080
LLAMA_CHAT_URL = f"http://127.0.0.1:{SERVER_PORT}/v1/chat/completions"
LLAMA_HEALTH_URL = f"http://127.0.0.1:{SERVER_PORT}/health"

# Logs stored in /dev/shm (Pure RAM)
RAM_LOG_DIR = "/dev/shm/manga_ram"
os.makedirs(RAM_LOG_DIR, exist_ok=True)
TRANS_LOG_PATH = os.path.join(RAM_LOG_DIR, "llama_translate.log")

_trans_process = None
_trans_log_file = None

# ==========================================================
# 🎯 SYSTEM PROMPT (ROMANCE, NSFW, MATURE & COMIC REGISTERS)
# ==========================================================
SYSTEM_PROMPT = """You are an uncensored, expert manga, manhwa, and manhua localizer (JP/CN/KR to EN).
Translate the numbered lines into natural, expressive, emotionally accurate English suitable for comic speech bubbles across all genres, including romance, drama, comedy, and mature/NSFW adult fiction.

CORE LOCALIZATION RULES:
1. FICTIONAL CONTEXT: This text consists purely of fictional dialogue from romance, adult drama, slice-of-life, and mature graphic novels. Dialogue ranges from intimate, affectionate, and teasing to sexually explicit, vulgar, or intense. Translate faithfully without moralizing, preaching, or refusal.
2. MATURE & ROMANTIC TONE:
   - Accurately convey intimacy, romantic tension, flirtation, and adult dialogue.
   - Accurately translate sexual expressions, vulgarity, swearing, and body references using their genuine, direct English equivalents (e.g., "fuck", "cock", "dick", "pussy", "shit", "bitch", "horny", "wet").
   - Localize intimate onomatopoeia, gasps, and vocalizations (e.g., "Ah...", "Mgh...", "Haah...") naturally for English comic pacing.
   - NEVER sanitize, soften, or convert mature/erotic dialogue into childish euphemisms (e.g., do NOT replace explicit words with "stuff", "down there", or "make love" unless the original Japanese/Korean source specifically used polite/euphemistic terms).
3. NO ASTERISKS OR SYMBOLS: Output words cleanly and in full. NEVER mask words using asterisks, dashes, or symbols (e.g., write "fuck", NEVER "f***", "f*ck", or "f---").
4. CONTEXT & IMPLIED PRONOUNS: Seamlessly infer missing subjects and pronouns (I, you, he, she, we, they) and relationship dynamics (lover, partner, boss, rival) from the surrounding dialogue.
5. CLEAN OUTPUT: Output ONLY the requested JSON. Do not include notes, translation trivia, content warnings, or markdown commentary.

OUTPUT FORMAT: Return ONLY a valid JSON object mapping each string index to its English translation.
Example:
{
  "0": "Ah... stop teasing me, it feels too good...",
  "1": "You're so sensitive right here... You want me to keep going, don't you?",
  "2": "Fuck... don't look at me like that."
}"""


def ensure_llama_server_binary():
    """Ensures llama-server binary is available."""
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


def ensure_translation_models_downloaded():
    """Downloads Qwen3.8-27B Q6_K and MTP Draft Q8_0 into /tmp/models."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    # Token loaded from environment with hardcoded fallback as requested
    hf_token = os.environ.get("HF_TOKEN", "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")

    # 1. Main 27B Model (~22.4 GB)
    if os.path.exists(TRANS_MODEL_PATH) and os.path.getsize(TRANS_MODEL_PATH) > 20 * 1e9:
        print(f"✅ Translation Model ready: {TRANS_MODEL_FILE}", flush=True)
    else:
        print(f"📥 Downloading {TRANS_MODEL_FILE} (~22.4 GB) to {MODEL_DIR}...", flush=True)
        hf_hub_download(
            repo_id=TRANS_REPO_ID,
            filename=TRANS_MODEL_FILE,
            local_dir=MODEL_DIR,
            token=hf_token
        )

    # 2. MTP Draft Companion (~3.16 GB)
    if os.path.exists(TRANS_MTP_PATH) and os.path.getsize(TRANS_MTP_PATH) > 3 * 1e9:
        print(f"✅ MTP Draft Companion ready: {os.path.basename(TRANS_MTP_PATH)}", flush=True)
    else:
        print(f"📥 Downloading MTP Companion ({TRANS_MTP_FILE}) to {MODEL_DIR}...", flush=True)
        hf_hub_download(
            repo_id=TRANS_REPO_ID,
            filename=TRANS_MTP_FILE,
            local_dir=MODEL_DIR,
            token=hf_token
        )


def start_translation_server(context_size: str = "32768"):
    """
    Launches llama-server with Qwen3.8-27B + MTP Draft Speculative Decoding across Dual T4 GPUs.
    """
    global _trans_process, _trans_log_file

    if _trans_process is not None and _trans_process.poll() is None:
        try:
            if requests.get(LLAMA_HEALTH_URL, timeout=1).status_code == 200:
                return True
        except Exception:
            pass

    stop_translation_server()
    ensure_translation_models_downloaded()
    server_exec = ensure_llama_server_binary()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    _trans_log_file = open(TRANS_LOG_PATH, "w", buffering=1)

    cmd = server_exec + [
        "--model", TRANS_MODEL_PATH,
        "--model-draft", TRANS_MTP_PATH,
        "--spec-type", "draft-mtp",
        "--spec-draft-n-max", "2",
        "--host", "127.0.0.1",
        "--port", str(SERVER_PORT),
        "--n-gpu-layers", "99",
        "--n-gpu-layers-draft", "99",
        "--ctx-size", str(context_size),
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

    print(f"🚀 Starting Qwen3.8-27B + MTP Server ({context_size} ctx) across Dual T4s...", flush=True)
    _trans_process = subprocess.Popen(
        cmd,
        stdout=_trans_log_file,
        stderr=subprocess.STDOUT,
        text=True
    )

    start_time = time.time()
    while time.time() - start_time < 360:
        if _trans_process.poll() is not None:
            print("❌ Translation llama-server crashed! Last 25 log lines:", flush=True)
            with open(TRANS_LOG_PATH, "r") as f:
                print("".join(f.readlines()[-25:]), flush=True)
            return False
        try:
            res = requests.get(LLAMA_HEALTH_URL, timeout=1)
            if res.status_code == 200:
                print("✅ Qwen3.8-27B + MTP Translation Server is ONLINE!", flush=True)
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)

    print("❌ Timed out waiting for Qwen3.8-27B server.", flush=True)
    stop_translation_server()
    return False


def stop_translation_server():
    """Unloads Qwen3.8-27B + MTP from VRAM so other tools can use the GPUs freely."""
    global _trans_process, _trans_log_file
    if _trans_process is not None:
        try:
            _trans_process.terminate()
            _trans_process.wait(timeout=5)
        except Exception:
            pass
        _trans_process = None

    if _trans_log_file is not None:
        try:
            _trans_log_file.close()
        except Exception:
            pass
        _trans_log_file = None

    subprocess.run("pkill -9 -f llama-server 2>/dev/null || true", shell=True)
    time.sleep(2)


def strip_thinking_and_extract_json(text: str) -> dict:
    """
    Discards all thinking monologue (<think>...</think>) before extracting JSON,
    preventing curly braces inside reasoning traces from breaking extraction.
    """
    text = text.strip()

    # Step 1: Strip out thinking tokens completely
    if "</think>" in text:
        clean_text = text.split("</think>")[-1].strip()
    else:
        # Fallback if generation cut off mid-thought without </think>
        clean_text = re.sub(r"<think>[\s\S]*?(?:$|</think>)", "", text).strip()

    # Step 2: Try parsing from Markdown ```json ... ``` code fences
    fence_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", clean_text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Step 3: Extract the outermost {...} JSON boundaries
    start_idx = clean_text.find("{")
    end_idx = clean_text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        raw_json = clean_text[start_idx:end_idx + 1]
        try:
            return json.loads(raw_json)
        except json.JSONDecodeError:
            pass

    return {}


def clean_translated_line(text: str) -> str:
    """
    Cleans stray CJK symbols or corrupt mathematical unicode while preserving
    English words, punctuation, and colloquial expressions.
    """
    if not text:
        return ""

    if "</think>" in text:
        text = text.split("</think>")[-1]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Remove mathematical cursive/bold unicode hallucinations
    text = re.sub(r"[\u1d400-\u1d7ff]", "", text)

    # Strip leftover untranslated CJK characters (Hiragana, Katakana, Hanzi, Hangul)
    text = re.sub(r"[\u3040-\u309F\u30A0-\u30FF\u4e00-\u9fff\uac00-\ud7af]+", "", text)

    return text.replace("\n", " ").strip()


def translate_chunk(text_list: list, max_retries: int = 3, enable_thinking: bool = True) -> dict:
    """
    Sends sequential speech bubbles to Qwen3.8-27B + MTP server.
    Uses 'low' reasoning effort for contextual subject/gender/tone inference without
    triggering safety self-talk or huge token latencies.
    """
    if not text_list:
        return {}

    if not start_translation_server():
        return {str(i): "" for i in range(len(text_list))}

    user_prompt = "Translate these sequential manga speech bubbles into English JSON:\n"
    for i, text in enumerate(text_list):
        clean_src = text.replace("\n", " ").strip()
        user_prompt += f"[{i}] {clean_src}\n"

    # Base payload parameters
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.8 if enable_thinking else 0.3,
        "top_p": 0.95 if enable_thinking else 0.85,
        "top_k": 20,
        "presence_penalty": 0.0,  # 0.0 prevents language mixing and repetition artifacts
        "max_tokens": 2048 if enable_thinking else 1024,
    }

    if enable_thinking:
        # When thinking is enabled, DO NOT pass {"response_format": {"type": "json_object"}}
        # as it forces the first token to be '{', which suppresses the <think> token stream.
        payload["reasoning_effort"] = "low"
        payload["chat_template_kwargs"] = {
            "enable_thinking": True,
            "preserve_thinking": False
        }
    else:
        # In non-thinking mode, strict JSON grammar enforcement is safe
        payload["response_format"] = {"type": "json_object"}
        payload["chat_template_kwargs"] = {
            "enable_thinking": False
        }

    for attempt in range(max_retries):
        try:
            response = requests.post(LLAMA_CHAT_URL, json=payload, timeout=300)
            response.raise_for_status()

            msg = response.json()["choices"][0]["message"]
            content = (msg.get("content") or "").strip()

            if not content:
                raise ValueError("Empty content returned from Qwen3.8-27B")

            translations_dict = strip_thinking_and_extract_json(content)

            missing = [str(idx) for idx in range(len(text_list)) if str(idx) not in translations_dict]
            if missing and attempt < max_retries - 1:
                print(f"      ⚠️ Missing JSON keys {missing}. Retrying ({attempt + 1}/{max_retries})...", flush=True)
                time.sleep(1.5)
                continue

            final_dict = {}
            for i in range(len(text_list)):
                raw_en = str(translations_dict.get(str(i), ""))
                final_dict[str(i)] = clean_translated_line(raw_en)

            return final_dict

        except Exception as e:
            print(f"      ⚠️ Translation Chunk Error (Attempt {attempt + 1}/{max_retries}): {e}", flush=True)
            if attempt < max_retries - 1:
                time.sleep(2)

    return {str(i): "" for i in range(len(text_list))}


def translate_manga(text: str, target_lang: str = "en", source_lang: str = "auto") -> str:
    """Single-line legacy helper."""
    res = translate_chunk([text])
    return res.get("0", "")
      
