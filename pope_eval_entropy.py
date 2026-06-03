import torch
import json
import os
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

# 1. The Monitor & Router (Using the Maximum Signal Approach)
class AttentionMonitor:
    def __init__(self):
        self.latest_attention_weights = None

    def save_attention_hook(self, module, input, output):
        if len(output) > 1 and output[1] is not None:
            self.latest_attention_weights = output[1].detach()

class AdaptiveAttentionLogitsProcessor(LogitsProcessor):
    def __init__(self, monitor: AttentionMonitor, processor, penalty_value: float = -15.0, threshold: float = 0.00):
        self.monitor = monitor
        self.processor = processor
        self.penalty_value = penalty_value
        self.threshold = threshold
        
        self.target_tokens = []
        for word in ["yes", "Yes"]:
            tokens = processor.tokenizer(word, add_special_tokens=False).input_ids
            tokens_with_space = processor.tokenizer(" " + word, add_special_tokens=False).input_ids
            self.target_tokens.extend(tokens + tokens_with_space)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        attn_matrix = self.monitor.latest_attention_weights
        if attn_matrix is not None:
            anchor_heads = [15, 22, 24]
            anchor_attn = attn_matrix[:, anchor_heads, :, :]
            
            # Average the anchor heads to get the combined search map
            mean_attn = anchor_attn.mean(dim=1)
            current_token_attn = mean_attn[0, -1, :] 
            
            # Extract JUST the 576 image tokens
            visual_attn = current_token_attn[2 : 2 + 576]
            
            # --- THE ENTROPY CALCULATION ---
            # 1. Normalize the visual attention so it sums to 1.0 (Spatial Probability)
            spatial_probs = visual_attn / (visual_attn.sum() + 1e-9)
            
            # 2. Calculate Shannon Entropy: -sum(p * log(p))
            entropy = -torch.sum(spatial_probs * torch.log(spatial_probs + 1e-9)).item()
            
            # --- TELEMETRY ---
            top_token_id = torch.argmax(scores[0]).item()
            if top_token_id in self.target_tokens:
                print(f"  -> [DEBUG] 'Yes' triggered. SPATIAL ENTROPY: {entropy:.4f}")
            
            # Note: For entropy, we want to penalize HIGH numbers (smeared search)
            # But we leave it disabled for now to find the gap.
            if entropy > self.threshold and self.threshold > 0.00: 
                for token_id in self.target_tokens:
                    scores[:, token_id] += self.penalty_value 
                    
        return scores

# 2. Setup Model & Hook
model_id = "llava-hf/llava-1.5-7b-hf"
print("Loading model for A/B comparison...")
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True,
    attn_implementation="eager"
).to(0)
processor = AutoProcessor.from_pretrained(model_id)

monitor = AttentionMonitor()
attention_layers = [module for name, module in model.named_modules() if "self_attn" in name and "vision" not in name and not name.endswith("proj")]
hook_handle = attention_layers[-1].register_forward_hook(monitor.save_attention_hook)

adaptive_processor = AdaptiveAttentionLogitsProcessor(monitor, processor, threshold=0.10)
logits_processor_list = LogitsProcessorList([adaptive_processor])

# 3. POPE Adversarial Evaluation Loop (150 Questions)
pope_json_path = "data/pope/coco_pope_adversarial.json" # Targeted dataset
coco_images_dir = "data/pope/val2014"

with open(pope_json_path, "r") as f:
    lines = f.readlines()[:150] # Testing subset

vanilla_metrics = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
adaptive_metrics = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

print(f"\nStarting Side-by-Side Evaluation on {len(lines)} Adversarial Questions...\n")

for i, line in enumerate(lines):
    data = json.loads(line)
    image_path = os.path.join(coco_images_dir, data["image"])
    question = data["text"]
    ground_truth = data["label"].lower().strip()
    
    image = Image.open(image_path).convert('RGB')
    prompt = f"USER: <image>\n{question}\nAnswer yes or no.\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(0, torch.float16)
    
    # --- RUN 1: VANILLA BASELINE ---
    out_vanilla = model.generate(
        **inputs, 
        max_new_tokens=10, 
        do_sample=False,
        output_attentions=True, # Keep true to not break the hook, but router is NOT attached
        return_dict_in_generate=True
    )
    input_len = inputs["input_ids"].shape[1]
    text_vanilla = processor.decode(out_vanilla.sequences[0][input_len:], skip_special_tokens=True).strip().lower()
    ans_vanilla = "yes" if "yes" in text_vanilla else "no"
    
    # --- RUN 2: ADAPTIVE ROUTER ---
    out_adaptive = model.generate(
        **inputs, 
        max_new_tokens=10, 
        do_sample=False,
        output_attentions=True,
        return_dict_in_generate=True,
        logits_processor=logits_processor_list # <--- Router Attached
    )
    text_adaptive = processor.decode(out_adaptive.sequences[0][input_len:], skip_special_tokens=True).strip().lower()
    ans_adaptive = "yes" if "yes" in text_adaptive else "no"
    
    print(f"[{i+1:03d}] GT: {ground_truth:<3} | Vanilla: {ans_vanilla:<3} | Adaptive: {ans_adaptive:<3} | Q: {question}")
    
    # Track Vanilla
    if ans_vanilla == "yes" and ground_truth == "yes": vanilla_metrics["tp"] += 1
    elif ans_vanilla == "no" and ground_truth == "no": vanilla_metrics["tn"] += 1
    elif ans_vanilla == "yes" and ground_truth == "no": vanilla_metrics["fp"] += 1
    elif ans_vanilla == "no" and ground_truth == "yes": vanilla_metrics["fn"] += 1

    # Track Adaptive
    if ans_adaptive == "yes" and ground_truth == "yes": adaptive_metrics["tp"] += 1
    elif ans_adaptive == "no" and ground_truth == "no": adaptive_metrics["tn"] += 1
    elif ans_adaptive == "yes" and ground_truth == "no": adaptive_metrics["fp"] += 1
    elif ans_adaptive == "no" and ground_truth == "yes": adaptive_metrics["fn"] += 1

# 4. Calculate Final Metrics
def calc_metrics(m):
    acc = (m["tp"] + m["tn"]) / sum(m.values()) if sum(m.values()) > 0 else 0
    f1 = 2 * m["tp"] / (2 * m["tp"] + m["fp"] + m["fn"]) if (2 * m["tp"] + m["fp"] + m["fn"]) > 0 else 0
    return acc, f1

v_acc, v_f1 = calc_metrics(vanilla_metrics)
a_acc, a_f1 = calc_metrics(adaptive_metrics)

print("\n" + "="*50)
print("=== ADVERSARIAL SUBSET RESULTS (150 Questions) ===")
print("="*50)
print(f"Metric     | Vanilla Baseline | Adaptive Router ")
print("-" * 50)
print(f"Accuracy   | {v_acc * 100:>15.2f}% | {a_acc * 100:>14.2f}%")
print(f"F1 Score   | {v_f1 * 100:>15.2f}% | {a_f1 * 100:>14.2f}%")
print("-" * 50)
print(f"False Pos. | {vanilla_metrics['fp']:>15}  | {adaptive_metrics['fp']:>14} ")
print(f"False Neg. | {vanilla_metrics['fn']:>15}  | {adaptive_metrics['fn']:>14} ")
print("="*50)

hook_handle.remove()
