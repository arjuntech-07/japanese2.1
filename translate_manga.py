import requests
import json
import re
import time

# ============================================================
# OPTIMAL SETTINGS FOR QWEN 3.6 27B (OLLAMA)
# ============================================================
OLLAMA_URL = "http://localhost:11434/api/chat"
# ⚠️ CHANGE THIS to your exact Ollama model tag (e.g., "qwen3.6:27b" or "heretic-qwen")
MODEL_NAME = "qwen3.6:27b" 

SYSTEM_PROMPT = """You are an expert manga translator (JP/CN/KR to EN).
Translate the provided lines into natural, concise English for speech bubbles.
Preserve tone and emotion. Do NOT add notes or explanations.

OUTPUT FORMAT: Return ONLY a valid JSON object mapping the input IDs (as strings) to English text.
Example:
{
  "0": "What are you doing?!",
  "1": "I won't let you escape."
}"""

def extract_json(text: str) -> dict:
    """Robustly extract JSON from LLM output."""
    text = text.strip()
    # Try 1: Markdown code block
    match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    if match:
        try: return json.loads(match.group(1))
        except: pass
    
    # Try 2: Any code block
    match = re.search(r"```\s*([\s\S]*?)\s*```", text)
    if match:
        try: return json.loads(match.group(1))
        except: pass

    # Try 3: Raw JSON object
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try: return json.loads(match.group(0))
        except: pass
        
    return {}

def translate_chunk(text_list: list, max_retries: int = 3) -> dict:
    """
    Sends a chunk of text (e.g., 10 lines) to Ollama and returns 
    a dictionary mapping index -> translated text.
    """
    if not text_list:
        return {}

    # Build prompt
    user_prompt = "Translate these manga lines:\n"
    for i, text in enumerate(text_list):
        user_prompt += f"[{i}] {text}\n"
        
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "think": False,  # ✅ CRITICAL: Bypasses thinking, Ollama handles separation
        "stream": False,
        "options": {
            "temperature": 0.3,      # Low for consistency
            "top_p": 0.80,           # Official non-thinking setting
            "top_k": 20,             # Official setting
            "presence_penalty": 1.5, # Official setting (prevents repetition loops)
            "num_ctx": 8192          # Plenty of room for chunks
        }
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=120)
            response.raise_for_status()
            
            # Ollama cleanly separates content from thinking
            content = response.json().get("message", {}).get("content", "").strip()
            
            if not content:
                raise ValueError("Empty content in response")

            # Extract JSON
            translations_dict = extract_json(content)
            
            # Verify we got the expected keys
            missing = [str(idx) for idx in range(len(text_list)) if str(idx) not in translations_dict]
            if missing and attempt < max_retries - 1:
                print(f"      ⚠️ Missing indices {missing} in JSON. Retrying...")
                time.sleep(2)
                continue
                
            # Finalize dict, ensuring all original indices are present
            final_dict = {}
            for i in range(len(text_list)):
                translated_text = translations_dict.get(str(i), "")
                
                # Safety check: Wipe if it failed and returned original JP/CN/KR
                if translated_text and re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4e00-\u9fff\uac00-\ud7af]', translated_text):
                    translated_text = "" 
                    
                final_dict[str(i)] = translated_text.replace('\n', ' ').strip()
                
            return final_dict

        except Exception as e:
            print(f"      ⚠️ Chunk API Error (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(3)

    # Fallback on total failure
    return {str(i): "" for i in range(len(text_list))}

# Legacy wrapper (keeps old code from crashing if called directly per line)
def translate_manga(text: str, target_lang: str = "en", source_lang: str = "auto") -> str:
    res = translate_chunk([text])
    return res.get("0", "")