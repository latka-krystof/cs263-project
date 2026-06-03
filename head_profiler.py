import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import requests
import numpy as np
import matplotlib.pyplot as plt # Added for visualization

# 1. The Monitor
class AttentionMonitor:
    def __init__(self):
        self.latest_attention_weights = None

    def save_attention_hook(self, module, input, output):
        if len(output) > 1 and output[1] is not None:
            self.latest_attention_weights = output[1].detach()

def get_head_attention(model, processor, monitor, prompt, image):
    """Runs a single token generation and extracts the attention mass for all 32 heads."""
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(0, torch.float16)
    
    # Generate exactly ONE token
    _ = model.generate(
        **inputs, 
        max_new_tokens=1, 
        output_attentions=True,
        return_dict_in_generate=True
    )
    
    attn_matrix = monitor.latest_attention_weights
    
    # Extract the attention of the token currently being generated (last row)
    current_token_attn = attn_matrix[0, :, -1, :] 
    
    # Slice out the 576 image tokens
    image_attn_only = current_token_attn[:, 2 : 2 + 576] 
    
    # Sum the visual attention mass for EACH head independently
    head_masses = image_attn_only.sum(dim=1).cpu().numpy()
    return head_masses

# 2. Setup
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

# 3. The Controlled Experiment
url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Prompt A: The Real Object
prompt_real = "USER: <image>\nDescribe the car in detail.\nASSISTANT:"
print("Profiling Head Attention on REAL object (Car)...")
real_masses = get_head_attention(model, processor, monitor, prompt_real, image)

# Prompt B: The Hallucinated Object
prompt_fake = "USER: <image>\nDescribe the dog running across the street in detail.\nASSISTANT:"
print("Profiling Head Attention on HALLUCINATED object (Dog)...")
fake_masses = get_head_attention(model, processor, monitor, prompt_fake, image)

hook_handle.remove()

# 4. Data Visualization for Slide 2
print("\nGenerating visual asset for presentation...")

# Target heads identified in previous experiments
anchor_heads = [15, 22, 24]

# Prepare labels: Our specific heads + the overall average
labels = [f"Head {h}" for h in anchor_heads] + ["Avg (All 32 Heads)"]

# Extract specific head data
real_vals = [real_masses[h] for h in anchor_heads]
fake_vals = [fake_masses[h] for h in anchor_heads]

# Calculate the mean across all 32 heads to demonstrate the noise problem
real_vals.append(np.mean(real_masses))
fake_vals.append(np.mean(fake_masses))

# Set up the matplotlib plot
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

# Create contrasting bars (Green for Real, Red for Hallucination)
rects1 = ax.bar(x - width/2, real_vals, width, label='Real Object (Car)', color='#2ca02c')
rects2 = ax.bar(x + width/2, fake_vals, width, label='Hallucinated (Dog)', color='#d62728')

# Styling the chart for a presentation
ax.set_ylabel('Visual Attention Mass', fontsize=12, fontweight='bold')
ax.set_title('Visual Anchor Isolation: Real vs. Hallucinated Search Signals', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
ax.legend(fontsize=11)

# Add horizontal grid lines to make values easy to read across the room
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

# Auto-label the bars so the audience doesn't have to guess the exact numbers
ax.bar_label(rects1, padding=3, fmt='%.2f', fontsize=10)
ax.bar_label(rects2, padding=3, fmt='%.2f', fontsize=10)

fig.tight_layout()

# Save locally to your GCP instance
output_filename = "slide2_visual_anchors.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Success! Saved high-resolution graphic to: {output_filename}")
