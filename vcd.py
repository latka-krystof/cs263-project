import torch
import json
import os
import numpy as np
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

# --- NEW: Gaussian Noise Mask Function ---
def add_gaussian_noise(image: Image.Image, mean=0, std=50) -> Image.Image:
    """Applies a Gaussian noise mask to a PIL image, as specified in the VCD paper."""
    img_array = np.array(image).astype(np.float32)
    noise = np.random.normal(mean, std, img_array.shape)
    noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

# 1. Standard Model Loading
model_id = "llava-hf/llava-1.5-7b-hf"
print("Loading model for Visual Contrastive Decoding (VCD)...")
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True
).to(0)
processor = AutoProcessor.from_pretrained(model_id)

# 2. Extract Token IDs for Binary QA
yes_id = processor.tokenizer("yes", add_special_tokens=False).input_ids[0]
Yes_id = processor.tokenizer("Yes", add_special_tokens=False).input_ids[0]
no_id = processor.tokenizer("no", add_special_tokens=False).input_ids[0]
No_id = processor.tokenizer("No", add_special_tokens=False).input_ids[0]

# 3. Setup Dataset and Hyperparameters
pope_json_path = "data/pope/coco_pope_adversarial.json"
coco_images_dir = "data/pope/val2014"

with open(pope_json_path, "r") as f:
    lines = f.readlines()[:150]

vcd_metrics = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
alpha = 1.0  # VCD Contrast Weight Penalty

print(f"\nStarting VCD Evaluation on {len(lines)} Adversarial Questions...\n")

for i, line in enumerate(lines):
    data = json.loads(line)
    image_path = os.path.join(coco_images_dir, data["image"])
    question = data["text"]
    ground_truth = data["label"].lower().strip()
    
    # Generate Original and Noise-Masked Visual Inputs
    image = Image.open(image_path).convert('RGB')
    image_corrupt = add_gaussian_noise(image, std=50) # <--- Updated Distortion
    
    prompt = f"USER: <image>\n{question}\nAnswer yes or no.\nASSISTANT:"
    
    # --- RUN 1: ORIGINAL FORWARD PASS (v) ---
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(0, torch.float16)
    with torch.no_grad():
        outputs = model(**inputs)
        orig_logits = outputs.logits[0, -1, :]
        
    # --- RUN 2: CORRUPTED FORWARD PASS (v') ---
    inputs_corr = processor(text=prompt, images=image_corrupt, return_tensors="pt").to(0, torch.float16)
    with torch.no_grad():
        outputs_corr = model(**inputs_corr)
        corr_logits = outputs_corr.logits[0, -1, :]
        
    # --- VISUAL CONTRASTIVE DECODING (Equation 3) ---
    # logit_vcd = (1 + alpha) * orig_logits - alpha * corr_logits
    vcd_logits = ((1 + alpha) * orig_logits) - (alpha * corr_logits)
    
    # Apply softmax logic (Since we are just comparing max probability between 'yes' and 'no', 
    # taking the max of the raw adjusted logits achieves the mathematically identical result 
    # to applying softmax first, as softmax is monotonic).
    scores = {
        "yes": max(vcd_logits[yes_id].item(), vcd_logits[Yes_id].item()),
        "no": max(vcd_logits[no_id].item(), vcd_logits[No_id].item())
    }
    ans_vcd = "yes" if scores["yes"] > scores["no"] else "no"
    
    print(f"[{i+1:03d}] GT: {ground_truth:<3} | VCD: {ans_vcd:<3} | Q: {question}")
    
    if ans_vcd == "yes" and ground_truth == "yes": vcd_metrics["tp"] += 1
    elif ans_vcd == "no" and ground_truth == "no": vcd_metrics["tn"] += 1
    elif ans_vcd == "yes" and ground_truth == "no": vcd_metrics["fp"] += 1
    elif ans_vcd == "no" and ground_truth == "yes": vcd_metrics["fn"] += 1

# 4. Final Aggregation
acc = (vcd_metrics["tp"] + vcd_metrics["tn"]) / sum(vcd_metrics.values()) if sum(vcd_metrics.values()) > 0 else 0
f1 = 2 * vcd_metrics["tp"] / (2 * vcd_metrics["tp"] + vcd_metrics["fp"] + vcd_metrics["fn"]) if (2 * vcd_metrics["tp"] + vcd_metrics["fp"] + vcd_metrics["fn"]) > 0 else 0

print("\n" + "="*40)
print("=== VCD BASELINE RESULTS (150 QA) ===")
print("="*40)
print(f"Accuracy   | {acc * 100:>15.2f}%")
print(f"F1 Score   | {f1 * 100:>15.2f}%")
print("-" * 40)
print(f"False Pos. | {vcd_metrics['fp']:>15}")
print(f"False Neg. | {vcd_metrics['fn']:>15}")
print("="*40)
