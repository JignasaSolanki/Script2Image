#!/usr/bin/env python3
# scripts/generate_images_hfapi.py
import os
import time
import json
import base64
from pathlib import Path
from io import BytesIO
from PIL import Image
import requests
import nltk
from nltk.tokenize import sent_tokenize

# Ensure punkt is available
nltk.download("punkt", quiet=True)

# ---- Configuration: change if you want other models ----
TEXT_MODEL = "google/flan-t5-large"          # HF model for expanding sentence -> detailed prompt
IMAGE_MODEL = "runwayml/stable-diffusion-v1-5"  # HF image generation model (choose one you have access to)
MAX_PROMPT_SENTENCES = 6
OUTPUT_DIR = Path("generated_images")
OUTPUT_DIR.mkdir(exist_ok=True)

HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
if not HF_TOKEN:
    raise SystemExit("HUGGINGFACE_TOKEN environment variable not set. Set it in GitHub secrets or your env for local testing.")

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# Example article — replace with reading from file if you like
ARTICLE = """
Bitcoin price hits a record high and traders celebrate in Times Square.
Regulators call for clearer rules as adoption grows worldwide.
Miners report increasing node counts, strengthening the network.
Analysts say the rally could continue into next week.
"""

def hf_text_inference(model, prompt, max_length=200, wait_for_model=True, timeout=120):
    url = f"https://api-inference.huggingface.co/models/{model}"
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_length},
        "options": {"wait_for_model": wait_for_model}
    }
    r = requests.post(url, headers=HEADERS, json=payload, timeout=timeout)
    r.raise_for_status()
    res = r.json()
    # common shapes vary; try to extract text safely
    if isinstance(res, list) and res and isinstance(res[0], dict):
        # e.g. [{'generated_text': '...'}]
        gen = res[0].get("generated_text") or res[0].get("text") or None
        if gen:
            return gen
    if isinstance(res, dict):
        if "generated_text" in res:
            return res["generated_text"]
        # some models return {'text': '...'}
        if "text" in res:
            return res["text"]
    # fallback: try common list-of-str responses
    if isinstance(res, list) and res and isinstance(res[0], str):
        return res[0]
    # else return str to avoid crash
    return str(res)

def hf_image_inference(model, prompt, width=768, height=512, steps=28, guidance=7.5, wait_for_model=True, timeout=180):
    url = f"https://api-inference.huggingface.co/models/{model}"
    payload = {
        "inputs": prompt,
        "parameters": {
            "width": width, "height": height,
            "num_inference_steps": steps, "guidance_scale": guidance,
            "negative_prompt": "lowres, bad anatomy, blurry, watermark, text, cropped, deformed"
        },
        "options": {"wait_for_model": wait_for_model}
    }
    r = requests.post(url, headers=HEADERS, json=payload, stream=True, timeout=timeout)
    r.raise_for_status()

    content_type = r.headers.get("content-type", "")
    if content_type.startswith("image"):
        return r.content

    # parse JSON -> search for base64 payload
    j = r.json()
    def find_base64(obj):
        if isinstance(obj, str):
            if len(obj) > 200 and all(c.isalnum() or c in "+/=\n\r" for c in obj[:40]):
                return obj
            return None
        if isinstance(obj, dict):
            for v in obj.values():
                f = find_base64(v)
                if f: return f
        if isinstance(obj, list):
            for v in obj:
                f = find_base64(v)
                if f: return f
        return None

    b64 = find_base64(j)
    if b64:
        return base64.b64decode(b64)

    raise RuntimeError(f"Unexpected HF response (no image). JSON head: {json.dumps(j)[:400]}")

def expand_sentence(sentence):
    instr = (
        "Rewrite the following short news sentence into a concise, vivid image prompt suitable "
        "for a Stable Diffusion style text-to-image model. Include setting, time of day, camera angle, "
        "key objects/people, mood. Do not include instructions for text overlays.\n\n"
        f"Sentence: {sentence}\n\nPrompt:"
    )
    return hf_text_inference(TEXT_MODEL, instr, max_length=120)

def main():
    sentences = [s.strip() for s in sent_tokenize(ARTICLE) if s.strip()]
    sentences = sentences[:MAX_PROMPT_SENTENCES]
    if not sentences:
        print("No sentences found in ARTICLE. Exiting.")
        return

    for idx, s in enumerate(sentences, start=1):
        print(f"\n[{idx}/{len(sentences)}] Original: {s}")
        try:
            detailed = expand_sentence(s)
            prompt = f"{detailed}, ultra realistic, cinematic, 8k, natural lighting"
            print("Prompt:", prompt[:300])

            img_bytes = hf_image_inference(IMAGE_MODEL, prompt)
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            out_path = OUTPUT_DIR / f"image_{idx:02d}.png"
            img.save(out_path)
            print("Saved:", out_path)
        except Exception as e:
            print("Error generating image for sentence:", e)

if __name__ == "__main__":
    main()
