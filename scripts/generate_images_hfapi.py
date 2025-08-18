import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from diffusers import StableDiffusionPipeline
import torch

# ----------------------------
# 1. Prompt expansion model
# ----------------------------
print("Loading text model...")
text_model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(text_model_name)
text_model = AutoModelForSeq2SeqLM.from_pretrained(text_model_name)

def expand_prompt(sentence):
    prompt = f"Expand this into a rich, detailed visual description for AI image generation: {sentence}"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = text_model.generate(**inputs, max_length=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ----------------------------
# 2. Stable Diffusion pipeline
# ----------------------------
print("Loading Stable Diffusion model...")
model_id = "stabilityai/stable-diffusion-2-1-base"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe = pipe.to("cpu")  # CPU mode

# ----------------------------
# 3. Example script input
# ----------------------------
news_script = """
AI-powered robots assist doctors in performing complex surgeries.
Tech companies announce breakthroughs in quantum computing.
Renewable energy adoption accelerates across major cities.
Space agencies prepare for manned missions to Mars by 2030.
"""

# ----------------------------
# 4. Process each sentence
# ----------------------------
os.makedirs("generated_images", exist_ok=True)
sentences = [s.strip() for s in news_script.split('.') if s.strip()]

for idx, sentence in enumerate(sentences, 1):
    print(f"\n[Sentence {idx}] {sentence}")
    detailed_prompt = expand_prompt(sentence)
    print(f"Expanded prompt: {detailed_prompt}")

   image = pipe(
    detailed_prompt,
    num_inference_steps=50,   # more steps = better detail, slower on CPU
    guidance_scale=7.5        # how closely it follows the text
    ).images[0]

    image_path = f"generated_images/image_{idx}.png"
    image.save(image_path)
    print(f"Saved {image_path}")

print("\n✅ Image generation complete. All files saved in 'generated_images/'")
