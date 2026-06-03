import torch
import json
import os
import re
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

# 1. Standard Model Loading
model_id = "llava-hf/llava-1.5-7b-hf"
print("Loading model for Pre-Flight Noun Attention...")
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True,
    attn_implementation="eager"
).to(0)
processor = AutoProcessor.from_pretrained(model_id)

yes_id = processor.tokenizer("yes", add_special_tokens=False).input_ids[0]
Yes_id = processor.tokenizer("Yes", add_special_tokens=False).input_ids[0]
no_id = processor.tokenizer("no", add_special_tokens=False).input_ids[0]
No_id = processor.tokenizer("No", add_special_tokens=False).input_ids[0]

# 2. Setup Dataset
pope_json_path = "data/pope/coco_pope_adversarial.json"
coco_images_dir = "data/pope/val2014"

with open(pope_json_path, "r") as f:
    lines = f.readlines()[:150]

metrics = {"vanilla": {"tp": 0, "tn": 0, "fp": 0, "fn": 0}, "adaptive": {"tp": 0, "tn": 0, "fp": 0, "fn": 0}}

# --- TELEMETRY MODE: Set to 0.0 to just log the numbers ---
THRESHOLD = 0.0 

print(f"\nStarting Pre-Flight Evaluation on {len(lines)} Adversarial Questions...\n")

for i, line in enumerate(lines):
    data = json.loads(line)
    image_path = os.path.join(coco_images_dir, data["image"])
    question = data["text"]
    ground_truth = data["label"].lower().strip()
    
    # Extract the target noun from the POPE question using regex
    match = re.search(r"Is there an? (.*?) in the", question, re.IGNORECASE)
    target_noun = match.group(1).strip().lower() if match else ""
    
    image = Image.open(image_path).convert('RGB')
    prompt = f"USER: <image>\n{question}\nAnswer yes or no.\nASSISTANT:"
    
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(0, torch.float16)
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        
    # Get Vanilla Answer
    logits = outputs.logits[0, -1, :]
    y_score = max(logits[yes_id].item(), logits[Yes_id].item())
    n_score = max(logits[no_id].item(), logits[No_id].item())
    ans_vanilla = "yes" if y_score > n_score else "no"
    
    ans_adaptive = ans_vanilla
    noun_attn_mass = 0.0
    
    # If the model wants to say "Yes", we audit its visual evidence for the noun!
    if ans_vanilla == "yes" and target_noun:
        input_ids_list = inputs.input_ids[0].tolist()
        
        # Tokenize the noun directly. We check both with and without a leading space 
        # because SentencePiece uses a specific marker for mid-sentence words.
        noun_tokens = processor.tokenizer(" " + target_noun, add_special_tokens=False).input_ids
        noun_tokens_fallback = processor.tokenizer(target_noun, add_special_tokens=False).input_ids
        
        target_indices = []
        
        # 1. Search for the token sequence in the prompt
        seq_len = len(noun_tokens)
        for j in range(len(input_ids_list) - seq_len + 1):
            if input_ids_list[j:j+seq_len] == noun_tokens:
                target_indices = list(range(j, j+seq_len))
                break
                
        if not target_indices: # Try fallback
            seq_len = len(noun_tokens_fallback)
            for j in range(len(input_ids_list) - seq_len + 1):
                if input_ids_list[j:j+seq_len] == noun_tokens_fallback:
                    target_indices = list(range(j, j+seq_len))
                    break
        
        if target_indices:
            # 2. Extract Anchor Head Attention
            attn_matrix = outputs.attentions[-1][0] # Last layer
            anchor_attn = attn_matrix[[15, 22, 24], :, :] 
            
            # Max across our 3 anchor heads
            max_attn, _ = anchor_attn.max(dim=0) 
            
            # 3. Calculate Visual Mass for the exact noun tokens
            masses = []
            for idx in target_indices:
                # The 576 image tokens begin after 'USER: ' -> indices 2 to 578
                mass = max_attn[idx, 2:578].sum().item() 
                masses.append(mass)
            
            noun_attn_mass = max(masses)
            print(f"  -> [DEBUG] Target: '{target_noun:<12}' | Pre-Flight Attn Mass: {noun_attn_mass:.4f}")
            
            # 4. The Intervention
            if noun_attn_mass < THRESHOLD:
                ans_adaptive = "no"

    print(f"[{i+1:03d}] GT: {ground_truth:<3} | Vanilla: {ans_vanilla:<3} | Adaptive: {ans_adaptive:<3} | Q: {question}")
    
    # Tally Vanilla
    if ans_vanilla == "yes" and ground_truth == "yes": metrics["vanilla"]["tp"] += 1
    elif ans_vanilla == "no" and ground_truth == "no": metrics["vanilla"]["tn"] += 1
    elif ans_vanilla == "yes" and ground_truth == "no": metrics["vanilla"]["fp"] += 1
    elif ans_vanilla == "no" and ground_truth == "yes": metrics["vanilla"]["fn"] += 1

    # Tally Adaptive
    if ans_adaptive == "yes" and ground_truth == "yes": metrics["adaptive"]["tp"] += 1
    elif ans_adaptive == "no" and ground_truth == "no": metrics["adaptive"]["tn"] += 1
    elif ans_adaptive == "yes" and ground_truth == "no": metrics["adaptive"]["fp"] += 1
    elif ans_adaptive == "no" and ground_truth == "yes": metrics["adaptive"]["fn"] += 1

# Final Aggregation
def print_res(m, name):
    acc = (m["tp"] + m["tn"]) / sum(m.values()) if sum(m.values()) > 0 else 0
    f1 = 2 * m["tp"] / (2 * m["tp"] + m["fp"] + m["fn"]) if (2 * m["tp"] + m["fp"] + m["fn"]) > 0 else 0
    print(f"{name:<10} | Acc: {acc*100:.2f}% | F1: {f1*100:.2f}% | FP: {m['fp']} | FN: {m['fn']}")

print("\n=== PRE-FLIGHT ATTENTION RESULTS ===")
print_res(metrics["vanilla"], "Vanilla")
print_res(metrics["adaptive"], "Adaptive")
