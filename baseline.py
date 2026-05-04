import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import requests

model_id = "llava-hf/llava-1.5-7b-hf"

print("Loading model into VRAM...")
# We load in float16 to ensure it fits comfortably in the 24GB VRAM
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
).to(0) # .to(0) moves it to your first GPU

processor = AutoProcessor.from_pretrained(model_id)

# Fetching a sample image for the test
print("Fetching sample image...")
url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# LLaVA 1.5 expects a very specific prompt format
prompt = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"

print("Processing inputs...")
inputs = processor(text=prompt, images=image, return_tensors="pt").to(0, torch.float16)

print("Generating response...")
output = model.generate(**inputs, max_new_tokens=100)

print("\n--- Model Output ---")
print(processor.decode(output[0], skip_special_tokens=True))
