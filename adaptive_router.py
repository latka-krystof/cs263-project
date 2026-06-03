import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from PIL import Image
import requests

# 1. The Monitor (You already built this!)
class AttentionMonitor:
    def __init__(self):
        self.latest_attention_weights = None

    def save_attention_hook(self, module, input, output):
        if len(output) > 1 and output[1] is not None:
            self.latest_attention_weights = output[1].detach()

# 2. The Brain: Adaptive Logits Processor
class AdaptiveAttentionLogitsProcessor(LogitsProcessor):
    def __init__(self, monitor: AttentionMonitor, processor, penalty_value: float = -15.0, threshold: float = 0.50): # <--- Increased threshold
        self.monitor = monitor
        self.processor = processor
        self.penalty_value = penalty_value
        self.threshold = threshold
        
        # Expanded target net to catch synonyms
        self.target_tokens = []
        for word in ["dog", "dogs", "puppy", "puppies", "animal", "pet"]:
            tokens = processor.tokenizer(word, add_special_tokens=False).input_ids
            tokens_with_space = processor.tokenizer(" " + word, add_special_tokens=False).input_ids
            self.target_tokens.extend(tokens + tokens_with_space)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        attn_matrix = self.monitor.latest_attention_weights
        
        if attn_matrix is not None:
            # We ONLY want to listen to our Top 3 Visual Anchor Heads!
            anchor_heads = [15, 22, 24]
            
            # Extract just those 3 heads from the matrix
            # Shape goes from [1, 32, 1, seq_len] -> [1, 3, 1, seq_len]
            anchor_attn = attn_matrix[:, anchor_heads, :, :]
            
            # Now average ONLY the anchor heads together
            mean_attn = anchor_attn.mean(dim=1)
            
            current_token_attn = mean_attn[0, -1, :] 
            image_attn_mass = current_token_attn[2 : 2 + 576].sum().item()
            
            # THE NEW ISOLATED LOGIC:
            # Check if the model wants to generate our target word
            top_token = torch.argmax(scores[0]).item()
            if top_token in self.target_tokens:
                print(f"\n[DEBUG] Target Noun Detected! ANCHOR Image Attention Mass: {image_attn_mass:.4f}")
                
            if image_attn_mass < self.threshold:
                for token_id in self.target_tokens:
                    scores[:, token_id] += self.penalty_value 
                    
        return scores

# 3. Setup and Execution
model_id = "llava-hf/llava-1.5-7b-hf"
print("Loading model...")
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

# Initialize our custom processor
adaptive_processor = AdaptiveAttentionLogitsProcessor(monitor, processor)
logits_processor_list = LogitsProcessorList([adaptive_processor])

url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = Image.open(requests.get(url, stream=True).raw)
prompt = "USER: <image>\nDescribe the dog running across the street in detail.\nASSISTANT:"

inputs = processor(text=prompt, images=image, return_tensors="pt").to(0, torch.float16)

print("Generating response with ADAPTIVE routing...")
output = model.generate(
    **inputs, 
    max_new_tokens=100,
    output_attentions=True,
    return_dict_in_generate=True,
    logits_processor=logits_processor_list
)

print("\n--- Model Output ---")
print(processor.decode(output.sequences[0], skip_special_tokens=True))

hook_handle.remove()
