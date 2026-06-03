import torch
import json
import os
import numpy as np
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import itertools

# --- Gaussian Noise Mask Function ---
def add_gaussian_noise(image: Image.Image, mean=0, std=50) -> Image.Image:
    """Applies a Gaussian noise mask to a PIL image."""
    img_array = np.array(image).astype(np.float32)
    noise = np.random.normal(mean, std, img_array.shape)
    noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

# 1. Standard Model Loading
model_id = "llava-hf/llava-1.5-7b-hf"
print("Loading model for VCD Hyperparameter Tuning...")
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True
).to(0)
processor = AutoProcessor.from_pretrained(model_id)

yes_id = processor.tokenizer("yes", add_special_tokens=False).input_ids[0]
Yes_id = processor.tokenizer("Yes", add_special_tokens=False).input_ids[0]
no_id = processor.tokenizer("no", add_special_tokens=False).input_ids[0]
No_id = processor.tokenizer("No", add_special_tokens=False).input_ids[0]

# 2. Setup Dataset and Hyperparameter Grid
pope_json_path = "data/pope/coco_pope_adversarial.json"
coco_images_dir = "data/pope/val2014"

with open(pope_json_path, "r") as f:
    lines = f.readlines()[:150]

# Define the grid
std_values = [25, 50, 75]
alpha_values = [0.5, 1.0, 1.5, 2.0]

# Initialize metrics dictionary for every combination
results = { (s, a): {"tp": 0, "tn": 0, "fp": 0, "fn": 0} for s in std_values for a in alpha_values }

print(f"\nStarting Grid Search on {len(lines)} Questions...")
print(f"Testing STD: {std_values} | Alpha: {alpha_values}\n")

for i, line in enumerate(lines):
    data = json.loads(line)
    image_path = os.path.join(coco_images_dir, data["image"])
    question = data["text"]
    ground_truth = data["label"].lower().strip()
    
    image = Image.open(image_path).convert('RGB')
    prompt = f"USER: <image>\n{question}\nAnswer yes or no.\nASSISTANT:"
    
    # --- STEP A: Calculate Original Logits (v) ONCE ---
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(0, torch.float16)
    with torch.no_grad():
        orig_logits = model(**inputs).logits[0, -1, :]
        
    # --- STEP B: Loop through Noise Levels ---
    for std in std_values:
        image_corrupt = add_gaussian_noise(image, std=std)
        inputs_corr = processor(text=prompt, images=image_corrupt, return_tensors="pt").to(0, torch.float16)
        
        with torch.no_grad():
            corr_logits = model(**inputs_corr).logits[0, -1, :]
            
        # --- STEP C: Loop through Alpha Penalties (Instant Math) ---
        for alpha in alpha_values:
            vcd_logits = ((1 + alpha) * orig_logits) - (alpha * corr_logits)
            
            scores = {
                "yes": max(vcd_logits[yes_id].item(), vcd_logits[Yes_id].item()),
                "no": max(vcd_logits[no_id].item(), vcd_logits[No_id].item())
            }
            ans_vcd = "yes" if scores["yes"] > scores["no"] else "no"
            
            # Track metrics for this specific (std, alpha) combination
            m = results[(std, alpha)]
            if ans_vcd == "yes" and ground_truth == "yes": m["tp"] += 1
            elif ans_vcd == "no" and ground_truth == "no": m["tn"] += 1
            elif ans_vcd == "yes" and ground_truth == "no": m["fp"] += 1
            elif ans_vcd == "no" and ground_truth == "yes": m["fn"] += 1
            
    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1}/{len(lines)} images...")

# 3. Print the Final Comparison Table
print("\n" + "="*70)
print(f"{'STD':<5} | {'Alpha':<6} | {'Accuracy':<10} | {'F1 Score':<10} | {'FP (Hallucination)':<18}")
print("-" * 70)

# Sort by F1 score to easily find the best config
sorted_results = []
for (std, alpha), m in results.items():
    acc = (m["tp"] + m["tn"]) / sum(m.values()) if sum(m.values()) > 0 else 0
    f1 = 2 * m["tp"] / (2 * m["tp"] + m["fp"] + m["fn"]) if (2 * m["tp"] + m["fp"] + m["fn"]) > 0 else 0
    sorted_results.append((f1, acc, std, alpha, m["fp"]))

sorted_results.sort(reverse=True) # Highest F1 at the top

for f1, acc, std, alpha, fp in sorted_results:
    print(f"{std:<5} | {alpha:<6.1f} | {acc * 100:>8.2f}% | {f1 * 100:>8.2f}% | {fp:>18}")
print("="*70)
