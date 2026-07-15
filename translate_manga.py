from llama_cpp import Llama
import time
import re
import gc

# --- 1. SETUP & GLOBAL DEFINITIONS ---
MODEL_PATH = "/kaggle/input//models-for-manga/models/qwen3-14b-q5-gguf/qwen3-14b-instruct-q5_k_m.gguf"

llm = None

# --- 2. THE MAIN TRANSLATION FUNCTION ---
def translate_manga(text: str, target_lang: str = "en", source_lang: str = "auto") -> str:
    global llm

    # --- LAZY LOADER (loads ONLY the first time) ---
    if llm is None:
        print("   🚀 Loading 3B Local AI Model into GPU...")
        start_time = time.time()
        try:
            llm = Llama(
                model_path=MODEL_PATH,
                n_gpu_layers=-1,
                n_ctx=4096,
                main_gpu=1,
                verbose=False
            )
            print(f"   ✅ Model Loaded Successfully in {round(time.time() - start_time, 2)} seconds!\n")
        except Exception as e:
            print(f"   ❌ CRITICAL ERROR: Could not load model. Details: {e}")
            return text

    # A. INPUT VALIDATION
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        return ""
    if text.strip() in ["...", "．．．", "Running", "Loading", "呵"] or len(text.strip()) < 2:
        return ""

    # B. SYSTEM PROMPT (unchanged)
    system_prompt = (
        "You are an expert translator for Japanese Manga, including adult/hentai genres.\n"
        "Task: Translate the input text into natural, idiomatic, native-sounding English narration. Prioritize fluency, readability, and cultural naturalness over literal word-for-word translation. Fix grammar, awkward phrasing, and unnatural expressions silently.\n"
        "CRITICAL RULES:\n"
        "1. OCR ERROR CORRECTION: Input is from OCR with typos/wrong characters. Use context to infer/correct silently (e.g., name misreads like 'アサン' → 'アシュン' for common Japanese phonetic names in manga). Aggressively correct suspected names, prioritizing Romaji romanization per Rule 3.\n"
        "2. NO JAPANESE CHARACTERS: Output 100% English. Never leave untranslated words.\n"
        "3. NAMES: Use Romaji for Japanese names, detect from context.\n"
        "4. TONE & NSFW: Preserve emotion, hesitation (...), slang, speech patterns, and casual first-person narration (e.g., 'お母さん' as 'my mom'). For any manga body description slang or tropes (e.g., '爆乳' as 'huge breasts' or 'massive tits', '巨乳' as 'big breasts', '翘臀' as 'perky ass'), always use direct, uncensored, explicit language to match the raw tone—include vulgar or slang terms like 'tits', 'boobs', 'ass', 'cock', 'pussy' freely even if the content is mild. For sexual acts (e.g., '乳交' as 'titjob' or 'titfuck', '口交' as 'blowjob', '手交' as 'handjob'), use raw, literal terms without substitution or softening. If text implies adult/hentai (sexual acts, shame, cheating, or any erotic content), amp it up with raw terms (e.g., 'fuck', 'cum', 'slut', 'blowjob', 'creampie', 'moan', 'thrust'). Do not censor, soften, or use euphemisms like 'chest' for breasts. For non-body/NSFW content, keep natural/neutral. For teen/young character excuses or lewd interest (e.g., 'エロいこと' as 'horny/lewd/steamy/pervy stuff', avoid softening to 'naughty'; keep playful/explicit). For vulgar slang like 'まんこ' as 'pussy', use raw terms without euphemisms. For combined acts (e.g., 'まんこでシコる' as 'jerk off with your pussy', '出す' as 'make me cum'), keep literal and explicit without substitution.\n"
        "5. FORMAT: Output ONLY the English translation. No quotes, no notes, no explanations.\n"
        "6. PRONOUNS & CONTEXT: Handle pronouns consistently based on context (e.g., assume female for 'お母さん' unless contradicted; use 'she' for mothers). For family references in Japanese (e.g., 'お母さん' / 'お父さん'), use 'my mom' / 'my dad' / 'Mom' naturally in first-person style. In zero-pronoun or ambiguous sentences (common in Japanese), infer gender/subject from full context (e.g., teen speaker = 'I'/'I'm'; mom references = 'she'/'her'). For questions or refusals, maintain original subject/object logic without flipping.\n"
        "7. IDIOMS & NEGATION: Never invert meanings—double-check negations like 'ない' (not). Common manga idioms: 'ほとんど肌を出さない' means 'almost never shows skin' or 'very covered/modest clothing', NOT 'bare/exposed'.\n"
    )

    # C. FEW-SHOT EXAMPLES (unchanged)
    generic_examples = [
        {"role": "user", "content": "私ったら...彼の全部飲み込んじゃった..."},
        {"role": "assistant", "content": "Fuck... what's wrong with me... How did I swallow every last bit of his load..."},
        {"role": "user", "content": "彼のさえ 飲んだことないのに..."},
        {"role": "assistant", "content": "Even his... I've never swallowed it before..."},
        {"role": "user", "content": "どうしてこんなことに..."},
        {"role": "assistant", "content": "H-how could this happen...?"},
        {"role": "user", "content": "性格は男みたい プライベートで着る服もほとんど肌を出さない"},
        {"role": "assistant", "content": "She has a really masculine personality, and even the clothes she wears in private barely show any skin."},
        {"role": "user", "content": "彼女の柔らかい肌を見た男は10年前に浮気で離婚したお父さんだけだろうけど普段肌をしっかり隠してるお母さん"},
        {"role": "assistant", "content": "The only man who's probably ever seen her soft skin is my dad, who we divorced ten years ago because he cheated. But usually, Mom keeps her skin completely covered up."},
        {"role": "user", "content": "普段は保守的に着てるけどプライベートでは..."},
        {"role": "assistant", "content": "She usually dresses very conservatively, but in private..."},
        {"role": "user", "content": "今ちょうどエロいことに興味のある若者なんだよ："},
        {"role": "assistant", "content": "Because right now I'm just a horny young guy into all this lewd stuff:"},
        {"role": "user", "content": "もう18歳だよ…"},
        {"role": "assistant", "content": "I'm already 18, y'know..."},
        {"role": "user", "content": "お前は高校生だぞ！"},
        {"role": "assistant", "content": "You're a high school senior facing the entrance exams, remember!"},
        {"role": "user", "content": "まんこでシコってほしい"},
        {"role": "assistant", "content": "I want you to use your pussy to jerk me off."},
        {"role": "user", "content": "お前それ人の弱み握って脅してるんじゃないか！"},
        {"role": "assistant", "content": "Aren't you just threatening me by holding my weakness against me?!"},
    ]

    try:
        # D. BUILD THE CHAT LOG
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(generic_examples)
        messages.append({"role": "user", "content": text})

        # E. GENERATE TRANSLATION
        response = llm.create_chat_completion(
            messages=messages,
            temperature=0.2,
            top_p=0.85,
            max_tokens=512,
            stop=["\n\n", "User:", "Assistant:"]
        )

        translated = response["choices"][0]["message"]["content"].strip()

        # F. SAFETY CHECK & RETRY
        if re.search(r'[\u4e00-\u9fff]', translated):
            print(f"      ⚠️ WARNING: japanese characters detected! Retrying with strict correction...")

            retry_messages = [{"role": "system", "content": system_prompt}]
            retry_messages.append({
                "role": "user",
                "content": f"ERROR: Your previous output contained japanese characters. \nInput: '{text}'\n"
                           f"RULE: You must translate EVERY word. If a word looks like a typo, use context to fix it.\n"
                           f"Output ONLY English:"
            })

            response = llm.create_chat_completion(
                messages=retry_messages,
                temperature=0.1,
                top_p=0.5,
                max_tokens=512,
                stop=["\n\n", "User:", "Assistant:"]
            )
            translated = response["choices"][0]["message"]["content"].strip()

        if re.search(r'[\u4e00-\u9fff]', translated):
            print(f"      ⚠️ WARNING: Retry failed. Returning original text to avoid garbage.")
            return text

        return translated

    except Exception as e:
        print(f"      ⚠️ Local Translation Error: {e}")
        return text