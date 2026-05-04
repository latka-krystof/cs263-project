import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from PIL import Image
import requests

# 1. Define our Custom Logits Processor
class BanWordLogitsProcessor(LogitsProcessor):
    def __init__(self, processor, banned_words):
        self.processor = processor
        # Convert banned words to their token IDs
        self.banned_token_ids = []
        for word in banned_words:
            # LLaVA/Llama tokenizer usually adds a space before words, so we get both
            tokens = processor.tokenizer(word, add_special_tokens=False).input_ids
            tokens_with_space = processor.tokenizer(" " + word, add_special_tokens=False).input_ids
            self.banned_token_ids.extend(tokens + tokens_with_space)
            
        print(f"Banned token IDs: {self.banned_token_ids}")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 'scores' contains the raw logits for the next token in the vocabulary.
        # We find the columns corresponding to our banned tokens and set them to negative infinity.
        for token_id in self.banned_token_ids:
            scores[:, token_id] = -float('inf')
        return scores

# 2. Load Model and Processor
model_id = "llava-hf/llava-1.5-7b-hf"
print("Loading model...")
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
).to(0)
processor = AutoProcessor.from_pretrained(model_id)

# 3. Setup the custom processor list
# Let's ban the model from saying "car", "cars", or "vehicle"
banned_words = ["car", "cars", "vehicle", "vehicles"]
custom_processor = BanWordLogitsProcessor(processor, banned_words)
logits_processor_list = LogitsProcessorList([custom_processor])

# 4. Fetch image and process inputs
url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = Image.open(requests.get(url, stream=True).raw)
prompt = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"

inputs = processor(text=prompt, images=image, return_tensors="pt").to(0, torch.float16)

# 5. Generate with our custom intervention!
print("Generating response WITH custom decoding...")
output = model.generate(
    **inputs, 
    max_new_tokens=100,
    logits_processor=logits_processor_list # <--- We inject our custom logic here
)

print("\n--- Model Output (Censored) ---")
print(processor.decode(output[0], skip_special_tokens=True))
