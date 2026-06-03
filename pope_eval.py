import torch
import json
import os
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

# 1. Load Model and Processor
model_id = "llava-hf/llava-1.5-7b-hf"
print("Loading model for Adversarial POPE Baseline...")
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
).to(0)
processor = AutoProcessor.from_pretrained(model_id)

# 2. Setup Evaluation Metrics Trackers
tp, tn, fp, fn = 0, 0, 0, 0

# 3. File Paths
# Updated to the adversarial dataset used in the grid search
pope_json_path = "data/pope/coco_pope_adversarial.json"
coco_images_dir = "data/pope/val2014"

print(f"Loading dataset from {pope_json_path}...")

with open(pope_json_path, "r") as f:
    lines = f.readlines()

# Match the 150-question limit from the other experiments
max_questions = 150 
lines = lines[:max_questions]

print(f"Starting evaluation loop for {len(lines)} adversarial questions...\n")

# 4. Evaluation Loop
for i, line in enumerate(lines):
    data = json.loads(line)
    
    image_name = data["image"]
    image_path = os.path.join(coco_images_dir, image_name)
    
    question = data["text"]
    ground_truth = data["label"].lower().strip()
    
    image = Image.open(image_path).convert('RGB')
        
    prompt = f"USER: <image>\n{question}\nAnswer yes or no.\nASSISTANT:"
    
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(0, torch.float16)
    
    output_ids = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    
    input_len = inputs["input_ids"].shape[1]
    generated_text = processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip().lower()
    
    model_answer = "yes" if "yes" in generated_text else "no"
    
    print(f"[{i+1}/{len(lines)}] Q: {question} | GT: {ground_truth} | Model: {model_answer}")
    
    if model_answer == "yes" and ground_truth == "yes":
        tp += 1
    elif model_answer == "no" and ground_truth == "no":
        tn += 1
    elif model_answer == "yes" and ground_truth == "no":
        fp += 1
    elif model_answer == "no" and ground_truth == "yes":
        fn += 1

# 5. Calculate Final Metrics
accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("\n" + "="*40)
print("=== ADVERSARIAL POPE BASELINE RESULTS ===")
print("="*40)
print(f"Total Questions: {len(lines)}")
print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F1 Score:  {f1 * 100:.2f}%")
print("-" * 40)
print(f"False Pos. (Hallucinations): {fp}")
print(f"False Neg. (Misses):         {fn}")
print("="*40)
