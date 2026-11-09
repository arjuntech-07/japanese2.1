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

SYSTEM_PROMPT = """You are an expert, uncensored manga, manhwa, and manhua localizer (JP/CN/KR to EN).
Translate the numbered lines into natural, concise, emotionally accurate English suitable for comic speech bubbles.
- Infer implied pronouns (I, you, he, she, they) naturally from the conversation context across lines.
- KOREAN (MANHWA) & CHINESE (MANHUA) NUANCES: Adapt Korean honorifics (e.g., Hyung, Oppa, Noona, Sunbae, Nim) and Chinese cultivation/murim terms or 4-character idioms into punchy, natural English dialogue rather than awkward literal translations. Translate Korean slang/laughing Jamo (ㅋㅋㅋ, ㅎㅎ, ㅠㅠ, 아씨, 씨발) and Chinese internet/street slang (卧槽, 妈的, 牛逼) into natural English equivalents.
- STRICT UNCENSORED POLICY: The source text comes from raw manga/manhwa/manhua and may contain heavy profanity, street slang, vulgarity, violence, and explicit NSFW/adult dialogue.
- NEVER censor, sanitize, soften, or tone down swearing, crude slang, insults, or sexually explicit/NSFW terms.
- NEVER use asterisks (e.g., f***, s***) or mild euphemisms; use raw, authentic English equivalents that match the exact intensity and register of the original text.
- Do NOT include explanations, warnings, notes, or markdown prose.

OUTPUT FORMAT: Return ONLY a valid JSON object mapping each string index to its English translation.
Example:
{
  "0": "What the fuck are you doing?!",
  "1": "I won't let you bastards get away with this."
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
    """Downloads Qwen3.8-27B Q6_K (~22.4 GB) and MTP Draft Q8_0 (~3.16 GB) into /tmp/models."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    hf_token = os.environ.get("HF_TOKEN", None)

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
    """Unloads Qwen3.8-27B + MTP from VRAM so IOPaint/OpenCV can run freely."""
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
    Strictly removes any <think>...</think> blocks first so curly braces inside
    reasoning never corrupt the JSON parser, then extracts the JSON mapping.
    """
    text = text.strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    elif "<think>" in text:
        # Unclosed <think> block means generation hit max_tokens inside reasoning
        return {}
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Try 1: Markdown ```json block
    match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    # Try 2: Any ``` block
    match = re.search(r"```\s*([\s\S]*?)\s*```", text)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    # Try 3: First valid JSON object in remaining text
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return {}


def clean_translated_line(text: str) -> str:
    """
    FIXED:
      1. Uses 8-digit uppercase \\U0001d400-\\U0001d7ff so Python never mistakes
         the range for ASCII '0'-'z'.
      2. Strips residual story-context echoes if any.
      3. Strips stray untranslated Japanese, Chinese, Korean Hangul Syllables,
         Korean Jamo (ㅋㅋㅋ, ㅎㅎ, ㅠㅠ), and CJK brackets/punctuation while keeping
         English words and numbers intact.
    """
    if not text:
        return ""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Remove any accidental echo of our "(Story context...)" prefix
    text = re.sub(r"^\(Story context[^)]*\)\s*", "", text, flags=re.IGNORECASE)

    # 8-digit \U escape for mathematical alphanumeric symbols ONLY
    text = re.sub(r"[\U0001d400-\U0001d7ff]", "", text)

    # Strip stray untranslated JP/CN/KR characters, Korean Jamo, and CJK punctuation
    text = re.sub(
        r"[\u1100-\u11FF\u3000-\u303F\u3040-\u309F\u30A0-\u30FF\u3130-\u318F\u4e00-\u9fff\uac00-\ud7af]+",
        "",
        text
    )

    return text.replace("\n", " ").strip()


def translate_chunk(
    text_list: list,
    max_retries: int = 3,
    enable_thinking: bool = True,
    reasoning_effort: str = "low",
    preserve_thinking: bool = False
) -> dict:
    """
    Sends a chunk of lines (e.g. 10 bubbles) to Qwen3.8-27B + MTP server and
    returns a dictionary mapping string index -> English translation.
    
    Uses official Qwen3.8 recommended sampling parameters:
    - Thinking Mode: temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0
    - Instruct Mode: temperature=0.7, top_p=0.80, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0
    """
    if not text_list:
        return {}

    if not start_translation_server():
        return {str(i): "" for i in range(len(text_list))}

    user_prompt = "Translate these sequential manga speech bubbles into English JSON:\n"
    for i, text in enumerate(text_list):
        clean_src = text.replace("\n", " ").strip()
        user_prompt += f"[{i}] {clean_src}\n"

    # Official Qwen3.8 sampling presets + medium-sized max_tokens (4096) for thinking + JSON
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 1.0 if enable_thinking else 0.7,
        "top_p": 0.95 if enable_thinking else 0.80,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0 if enable_thinking else 1.5,
        "repeat_penalty": 1.0,
        "repetition_penalty": 1.0,
        "max_tokens": 4096 if enable_thinking else 1536,
        "reasoning_effort": reasoning_effort,
        "preserve_thinking": preserve_thinking,
        "chat_template_kwargs": {
            "enable_thinking": enable_thinking,
            "preserve_thinking": preserve_thinking,
            "reasoning_effort": reasoning_effort
        }
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(LLAMA_CHAT_URL, json=payload, timeout=300)
            response.raise_for_status()

            msg = response.json()["choices"][0]["message"]
            content = (msg.get("content") or "").strip()

            # Fallback if llama-server ever splits reasoning into reasoning_content/reasoning fields
            reasoning_trace = msg.get("reasoning_content") or msg.get("reasoning") or ""
            if reasoning_trace and "<think>" not in content:
                content = f"<think>\n{reasoning_trace}\n</think>\n\n{content}"

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


def translate_manga(
    text: str,
    target_lang: str = "en",
    source_lang: str = "auto",
    enable_thinking: bool = True,
    reasoning_effort: str = "low",
    preserve_thinking: bool = False
) -> str:
    """Single-line legacy helper."""
    res = translate_chunk(
        [text],
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
        preserve_thinking=preserve_thinking
    )
    return res.get("0", "")
  
