# Adaptive Decoding: Reducing Hallucinations of Large Vision Language Models

## Overview
This repository contains the codebase for investigating inference-time interventions to mitigate object hallucination in Large Vision Language Models (LVLMs). Specifically, this project focuses on the open-source **LLaVA-1.5 (7B)** architecture and evaluates performance using the **POPE** (Evaluating Object Hallucination) benchmark. 

The primary goal is to implement and benchmark Visual Contrastive Decoding (VCD) and propose a novel Custom Adaptive Decoding algorithm using a custom Hugging Face `LogitsProcessor` to enforce consistency with visual context during autoregressive generation.

## Environment Setup
This project requires a GPU with at least 24GB of VRAM (e.g., NVIDIA L4 or RTX 3090/4090) to run LLaVA-1.5 in `fp16` precision.

1. **Create a Conda environment:**
   ```bash
   conda create -n cs263 python=3.10
   conda activate cs263
   ```

2. **Install dependencies:**
   ```bash
   pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
   pip install transformers accelerate pillow datasets requests
   ```

3. **Hugging Face Authentication:**
   You must be authenticated to pull the LLaVA weights.
   ```bash
   hf auth login
   ```

## Data Acquisition (POPE Benchmark)
To run the evaluation pipeline, you need the MSCOCO 2014 validation images and the official POPE question files.

Run the following commands from the root of the repository to download and extract the necessary data:
```bash
mkdir -p data/pope
cd data/pope

# Download COCO val2014 images (~6GB)
wget [http://images.cocodataset.org/zips/val2014.zip](http://images.cocodataset.org/zips/val2014.zip)
unzip -q val2014.zip

# Download POPE Random questions
wget [https://raw.githubusercontent.com/AoiDragon/POPE/main/output/coco/coco_pope_random.json](https://raw.githubusercontent.com/AoiDragon/POPE/main/output/coco/coco_pope_random.json)

cd ../..
```

## Usage

### 1. Baseline Model Test
To verify that the model loads correctly into VRAM and can generate coherent descriptions:
```bash
python baseline.py
```

### 2. Custom Decoding Proof-of-Concept
To test the custom `LogitsProcessor` interception (e.g., dynamically banning specific tokens like "car" during generation):
```bash
python custom_decoding.py
```

### 3. POPE Evaluation Pipeline
To run the quantitative evaluation loop against the POPE benchmark and calculate Accuracy, Precision, Recall, and F1 scores:
```bash
python pope_eval.py
```
*(Note: You can adjust the `max_questions` variable inside the script to run a smaller subset for quick testing).*

## Next Steps
- [x] Establish infrastructure and environment
- [x] Implement POPE baseline evaluation loop
- [x] Prove custom `LogitsProcessor` interception mechanics
- [ ] Implement standard Visual Contrastive Decoding (VCD)
- [ ] Architect Custom Adaptive Router using cross-attention weights
