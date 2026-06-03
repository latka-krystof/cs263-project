import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import requests

# 1. Define the Monitor to catch the weights
class AttentionMonitor:
    def __init__(self):
        self.latest_attention_weights = None

    def save_attention_hook(self, module, input, output):
        # LlamaAttention output is a tuple. 
        # output[0] is the projection, output[1] contains the attention weights.
        if len(output) > 1 and output[1] is not None:
            self.latest_attention_weights = output[1].detach().clone()

# 2. Load the Model
model_id = "llava-hf/llava-1.5-7b-hf"
print("Loading model...")
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
    attn_implementation="eager" # <--- Forces the model to compute extractable weights
).to(0)
processor = AutoProcessor.from_pretrained(model_id)

# 3. Wire up the Hook
monitor = AttentionMonitor()

# Dynamically find the last attention layer of the language model
# We look for "self_attn" to avoid accidentally hooking into the vision tower
attention_layers = []
for name, module in model.named_modules():
    # We want the main self_attn block, but NOT the q/k/v/o linear projections inside it
    if "self_attn" in name and "vision" not in name and not name.endswith("proj"):
        attention_layers.append((name, module))

print(f"Found {len(attention_layers)} attention layers.")
last_layer_name, last_layer_attn = attention_layers[-1]
print(f"Hooking into: {last_layer_name}")

hook_handle = last_layer_attn.register_forward_hook(monitor.save_attention_hook)

# 4. Prepare Inputs
url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = Image.open(requests.get(url, stream=True).raw)
prompt = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"

inputs = processor(text=prompt, images=image, return_tensors="pt").to(0, torch.float16)
input_length = inputs["input_ids"].shape[1]
print(f"Total input tokens (text + image): {input_length}")

# 5. Generate exactly ONE token
print("Generating a single token to trigger the hook...")
# We MUST set output_attentions=True, otherwise the model skips calculating the matrix to save memory
_ = model.generate(
    **inputs, 
    max_new_tokens=1, 
    output_attentions=True,
    return_dict_in_generate=True
)

# 6. Inspect the catch
attn = monitor.latest_attention_weights
if attn is not None:
    print("\n--- Success! Intercepted Attention Matrix ---")
    print(f"Shape: {attn.shape}")
    print("Dimension breakdown:")
    print(f"  Batch size: {attn.shape[0]}")
    print(f"  Attention heads: {attn.shape[1]}")
    print(f"  Query sequence length (the new token): {attn.shape[2]}")
    print(f"  Key sequence length (past tokens + new token): {attn.shape[3]}")
else:
    print("\n--- Failed to capture attention ---")

# Clean up
hook_handle.remove()
