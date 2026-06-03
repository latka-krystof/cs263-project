# Adaptive Decoding: Reducing Hallucinations of Large Vision Language Models

## Overview
This repository contains the codebase for investigating inference-time interventions to mitigate object hallucination in Large Vision Language Models (LVLMs). Specifically, this project focuses on the open-source **LLaVA-1.5 (7B)** architecture and evaluates performance using the **POPE** (Evaluating Object Hallucination) benchmark. 

The primary goal of this project was to evaluate internal adaptive attention-routing mechanisms. After empirical results demonstrated a "forced-scan effect" that caused internal thresholding to mathematically fail in binary VQA tasks, the repository includes a successful pivot to external structural interventions using **Visual Contrastive Decoding (VCD)**.

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
   pip install transformers accelerate pillow datasets requests matplotlib numpy
   ```

3. **Hugging Face Authentication:**
You must be authenticated to pull the LLaVA weights.
   ```bash
   hf auth login
   ```

## Data Acquisition (POPE Benchmark)

To run the evaluation pipeline, you need the MSCOCO 2014 validation images and the adversarial POPE question files.

Run the following commands from the root of the repository to download and extract the necessary data:

```bash
mkdir -p data/pope
cd data/pope

# Download COCO val2014 images (~6GB)
wget [http://images.cocodataset.org/zips/val2014.zip](http://images.cocodataset.org/zips/val2014.zip)
unzip -q val2014.zip

# Download POPE Adversarial questions
wget [https://raw.githubusercontent.com/AoiDragon/POPE/main/output/coco/coco_pope_adversarial.json](https://raw.githubusercontent.com/AoiDragon/POPE/main/output/coco/coco_pope_adversarial.json)

cd ../..

```

## Repository Structure & Execution

The scripts in this repository are designed to be run independently. They map directly to the experimental phases detailed in the final report.

### 1. Baseline Evaluation

Verify the model loads into VRAM and test its vanilla autoregressive performance against the adversarial dataset.

* `baseline.py`: Basic generation test to ensure hardware and library compatibility.
* `pope_eval.py`: Runs the standard LLaVA-1.5 generation loop against the POPE adversarial subset to establish the baseline Accuracy and F1 scores.

### 2. Profiling Visual Attention

Scripts dedicated to intercepting the PyTorch eager execution loop to analyze the 32 attention heads of LLaVA-1.5.

* `head_profiler.py`: Identifies the "Visual Anchor" heads (Heads 15, 22, 24) by contrasting attention mass between real and hallucinated objects.
* `attention_tracer.py`: Utilities for extracting and analyzing raw cross-attention matrices during generation.

### 3. Approach 1: Token-Level Routing

Experiments attempting to apply dynamic logit penalties before the softmax layer by auditing the cross-attention mass of the generated "Yes" token.

* `pope_eval_att_mass_max.py`: Evaluates routing based on the maximum spatial attention mass.
* `pope_eval_att_mass_mean.py`: Evaluates routing based on the mean spatial attention mass.
* `pope_eval_entropy.py`: Evaluates routing based on Spatial Shannon Entropy (distribution sharpness).

### 4. Approach 2: Pre-Flight Target Noun Routing

Experiments attempting to mitigate the Token-Level routing failure by explicitly auditing the target noun found in the input prompt prior to generation.

* `adaptive_router.py`: The core `LogitsProcessor` classes and intervention logic.
* `pope_eval_preflight.py`: Extracts the specific noun from the POPE prompt and audits its visual grounding mass.

### 5. Structural Interventions (Visual Contrastive Decoding)

The final implemented solution utilizing Gaussian noise to isolate and subtract the model's language prior.

* `vcd.py`: Executes the optimized VCD intervention (using Standard Deviation = 25, Alpha = 1.0) against the POPE benchmark.
* `vcd_tuning.py`: A nested grid search script used to determine the optimal noise boundaries and contrastive penalties for the LLaVA-1.5 architecture.
