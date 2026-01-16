# LLM Fine-Tuning Pipeline

End-to-end pipeline for domain-specific LLM fine-tuning: from data collection to production deployment.

---

## Table of Contents

- [Overview](#overview)
- [Methodology & Design Decisions](#methodology--design-decisions)
- [Quick Start](#quick-start)
- [Environment Variables](#environment-variables)
- [Usage](#usage)
  - [run_pipeline.sh](#run_pipelinesh)
  - [pipeline.py](#pipelinepy)
- [Pipeline Stages](#pipeline-stages)
- [Configuration Reference](#configuration-reference)
- [Project Structure](#project-structure)

---

## Overview

This pipeline automates the complete workflow of fine-tuning a small language model for domain-specific question-answering:

| Stage | Name | Description |
|-------|------|-------------|
| 1 | Data Collection | Search and download domain PDFs via SerpAPI + Docling |
| 2 | Data Processing | Extract, clean, and chunk text |
| 3 | Dataset Generation | Generate QA pairs using Gemini API |
| 4 | Fine-tuning | Train LoRA/QLoRA adapters with experiment tracking |
| 4a | -- Baseline Eval | Evaluate base model performance |
| 4b | -- Fine-tuning | Train LoRA adapter |
| 4c | -- Checkpoint Selection | Evaluate all checkpoints, select best |
| 4d | -- Final Eval | Evaluate fine-tuned model |
| 5 | Model Registration | Version and upload to HuggingFace Hub |
| 6 | Deployment | Serve via vLLM + FastAPI (Docker) |
| 7 | Monitoring | Prometheus metrics + structured logging |

---

## Methodology & Design Decisions

Due to time constraints, I prioritized **simplest yet sufficient** solutions. Below are key decisions and trade-offs.

### Stage 1: Data Collection
- **Approach:** SerpAPI + PDF-only mode
- Implemented PDF download with validation (checks `%PDF-` header)
- Stores metadata (URL, title, source) for traceability
- **Future:** Web scraping for HTML, multiple data source types, fixing some auth errors that prevents downloading pdfs

### Stage 2: Data Processing
- **Approach:** Docling for PDF-to-Markdown extraction
- Custom cleaning: removes image tags, page numbers, TOC dot-leaders, photo credits
- Chunking by section headers, filtered by token count (20-500 tokens)
- **Future:** Surya OCR for better accuracy, semantic chunking, MinHash deduplication

### Stage 3: Dataset Generation
- **Approach:** Gemini API with controlled answer style
- Groups small chunks (<=500 tokens) before sending to LLM
- Generates varied QA pairs (what, why, how, compare, explain)

**Critical decision - Answer Style Control:**
```
[ANSWER STYLE]
- Be concise and directly address the question
- Do NOT add explanations unless explicitly required
- Avoid examples, lists, or extra background unless asked
- Keep answers brief and factual
```
**Why?** Early experiments showed mismatch: generated answers were short/precise, but base model produced verbose responses during evaluation. This made comparison unfair. Solution: enforce same answer style in both dataset generation AND evaluation system prompt.

### Stage 4: Fine-tuning
- **Approach:** LoRA/QLoRA with TRL's SFTTrainer
- 8-bit quantization for memory efficiency
- LoRA rank=16, alpha=32, targeting attention layers
- Weights & Biases for experiment tracking

### Stage 5: Evaluation
**Metrics:** ROUGE-1/2/L, BERTScore F1, Token F1

**Why no BLEU?** BLEU expects exact n-gram matches and penalizes shorter outputs. QA answers vary in phrasing - two correct answers can have zero BLEU overlap. In testing, BLEU scores were ~0.02 even for good answers. Removed to avoid misleading metrics.

**Combined score:** Weighted average (BERTScore 40%, ROUGE-L 30%, Token F1 20%, ROUGE-1 10%)

### Stage 6: Model Registration
- Auto-generates repo name: `{domain}-{use_case}-{method}-v{X.Y}`
- Uploads LoRA adapter weights (not full model)
- Generates model card with metrics

### Stage 7 & 8: Deployment & Monitoring
- vLLM (port 8001) + FastAPI gateway (port 8000) in Docker
- Prometheus metrics + structured logging

**CI/CD:**
- CI: GitHub Actions for linting (ruff)
- CD: Not implemented - no deployment server, running locally via Docker

---

## Quick Start

```bash
# 1. Clone and setup
git clone <repo-url>
cd energyai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 4. Run full pipeline + deploy
./scripts/run_pipeline.sh --all
```

---

## Environment Variables

Create a `.env` file in the project root:

```bash
# Required for Stage 1: Data Collection
SERP_API_KEY=your_serpapi_key

# Required for Stage 3: Dataset Generation
GEMINI_API_KEY=your_gemini_key

# Required for Stage 4: Fine-tuning (gated models)
HF_TOKEN=your_huggingface_token

# Optional: Experiment tracking
WANDB_API_KEY=your_wandb_key

# Optional: Model registration (Stage 5)
HF_USERNAME=your_hf_username
```

---

## Usage

### run_pipeline.sh

The main entry point. Handles both training pipeline and serving infrastructure.

```bash
# ---------------------------------------------------------------------
# PIPELINE COMMANDS
# ---------------------------------------------------------------------

# Run full pipeline (stages 1-5)
./scripts/run_pipeline.sh

# Run specific stage
./scripts/run_pipeline.sh --stage 3

# Run sub-stage (Stage 4 has sub-stages: 4a, 4b, 4c, 4d)
./scripts/run_pipeline.sh --stage 4b

# Run range of stages
./scripts/run_pipeline.sh --start 2 --end 4

# List all available stages
./scripts/run_pipeline.sh --list

# Use custom config file
./scripts/run_pipeline.sh --config custom_config.yaml

# ---------------------------------------------------------------------
# SERVING COMMANDS
# ---------------------------------------------------------------------

# Build Docker images
./scripts/run_pipeline.sh --build

# Start servers (Docker) + run integration tests
./scripts/run_pipeline.sh --serve

# Start servers locally (no Docker)
./scripts/run_pipeline.sh --serve-local

# Run integration tests only (servers must be running)
./scripts/run_pipeline.sh --test

# Stop Docker servers
./scripts/run_pipeline.sh --stop

# ---------------------------------------------------------------------
# COMBINED COMMANDS
# ---------------------------------------------------------------------

# Full pipeline + build + serve + test
./scripts/run_pipeline.sh --all
```

### pipeline.py

Python orchestrator for fine-grained control over pipeline stages.

```bash
# Run all stages (1-5)
python pipeline.py

# Run specific stage
python pipeline.py --stage 3

# Run sub-stages
python pipeline.py --stage 4a    # Baseline evaluation only
python pipeline.py --stage 4b    # Fine-tuning only
python pipeline.py --stage 4c    # Checkpoint selection only
python pipeline.py --stage 4d    # Final evaluation only

# Run stage range
python pipeline.py --start 2 --end 4

# Use custom config
python pipeline.py --config custom_config.yaml

# List available stages
python pipeline.py --list
```

---

## Pipeline Stages

### Stage 1: Data Collection

Searches the web for domain-specific PDFs and extracts content.

**Tools:** SerpAPI (search), Docling (PDF extraction)

**Outputs:**
- `data/raw/*.pdf` - Downloaded PDF files
- `data/raw/metadata.json` - Source metadata
- `data/processed/*.md` - Extracted markdown

**Config:**
```yaml
data_collector:
  num_results: 500       # Number of PDFs to download
  mode: "pdf"            # "pdf" or "html"
  query: "pregnancy"     # Search query
```

---

### Stage 2: Data Processing

Cleans and chunks the extracted text.

**Outputs:**
- `data/chunks/chunks.json` - Processed text chunks with metadata

**Config:**
```yaml
data_processor:
  min_tokens: 20         # Skip chunks smaller than this
  target_tokens: 512     # Target chunk size
  max_tokens: 500        # Skip chunks larger than this
```

---

### Stage 3: Dataset Generation

Generates question-answer pairs from chunks using an LLM.

**Tools:** Gemini API

**Outputs:**
- `data/qa_dataset/train.json` - Training QA pairs
- `data/qa_dataset/eval.json` - Evaluation QA pairs

**Config:**
```yaml
dataset_generator:
  max_tokens: 600            # Max tokens per chunk group
  train_ratio: 0.8           # Train/eval split (80/20)
  model: "gemini-2.5-flash"  # LLM for generation
```

---

### Stage 4: Fine-tuning

Trains a LoRA adapter on the base model. Consists of 4 sub-stages:

| Sub-stage | Description |
|-----------|-------------|
| 4a | Evaluate baseline model |
| 4b | Train LoRA adapter |
| 4c | Evaluate checkpoints, select best |
| 4d | Evaluate final fine-tuned model |

**Tools:** TRL, PEFT, BitsAndBytes, Weights & Biases

**Outputs:**
- `models/checkpoints/` - Training checkpoints
- `models/lora_weights/` - Best LoRA adapter
- `reports/eval_baseline.json` - Baseline metrics
- `reports/eval_finetuned.json` - Fine-tuned metrics

**Config:**
```yaml
finetuning:
  base_model: "meta-llama/Llama-3.2-3B-Instruct"
  quantization: "8bit"       # null, "4bit", "8bit"
  
  # LoRA parameters
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  
  # Training
  strategy: "epoch"          # "epoch" or "steps"
  epochs: 20
  batch_size: 8
  learning_rate: 1.0e-4
  
  # Experiment tracking
  wandb_enabled: true
  wandb_project: "pregnancy-qa-finetune"
```

---

### Stage 5: Model Registration

Uploads the fine-tuned adapter to HuggingFace Hub with auto-versioning.

**Outputs:**
- HuggingFace repo: `{username}/{domain}-{use_case}-{method}-v{X.Y}`
- `reports/registration.json` - Registration record

**Config:**
```yaml
registration:
  hf_username: null      # Uses HF_USERNAME env var if null
  repo_name: null        # Auto-generates if null
  private: false         # Public or private repo
```

---

### Serving (--serve)

Deploys the model as an API using Docker.

| Service | Description | Port |
|---------|-------------|------|
| vLLM | High-performance inference server | 8001 |
| FastAPI | API gateway with monitoring | 8000 |

**Endpoints:**
- `GET /health` - Health check
- `GET /ready` - Readiness check
- `GET /models` - List loaded models
- `POST /ask` - Ask a question
- `GET /metrics` - Prometheus metrics

**Config:**
```yaml
serving:
  vllm_host: "vllm"              # "localhost" for local, "vllm" for Docker
  vllm_port: 8001
  api_port: 8000
  timeout: 60
  max_model_len: 1024
  gpu_memory_utilization: 0.2
  warmup_on_start: true
```

---

## Configuration Reference

Complete `config.yaml` structure:

```yaml
# ---------------------------------------------------------------------
# DOMAIN SETTINGS
# ---------------------------------------------------------------------
domain: "pregnancy"              # Target domain (natural language)
use_case: "qa"                   # Task type

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------
paths:
  raw_data: "data/raw"
  processed_data: "data/processed"
  chunks: "data/chunks"
  qa_dataset: "data/qa_dataset"

# ---------------------------------------------------------------------
# STAGE 1: DATA COLLECTION
# ---------------------------------------------------------------------
data_collector:
  num_results: 500
  mode: "pdf"
  query: "pregnancy pdfs"

# ---------------------------------------------------------------------
# STAGE 2: DATA PROCESSING
# ---------------------------------------------------------------------
data_processor:
  min_tokens: 20
  target_tokens: 512
  max_tokens: 500

# ---------------------------------------------------------------------
# STAGE 3: DATASET GENERATION
# ---------------------------------------------------------------------
dataset_generator:
  max_tokens: 600
  train_ratio: 0.8
  model: "gemini-2.5-flash"

# ---------------------------------------------------------------------
# STAGE 4: FINE-TUNING
# ---------------------------------------------------------------------
finetuning:
  base_model: "meta-llama/Llama-3.2-3B-Instruct"
  train_data: "data/qa_dataset/train.json"
  output_dir: "models/checkpoints"
  final_model_dir: "models/lora_weights"
  
  quantization: "8bit"           # null, "4bit", "8bit"
  
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  
  strategy: "epoch"              # "epoch" or "steps"
  epochs: 20
  max_steps: 500
  batch_size: 8
  gradient_accumulation_steps: 4
  learning_rate: 1.0e-4
  lr_scheduler_type: "cosine"
  weight_decay: 0.01
  warmup_ratio: 0.1
  max_grad_norm: 1.0
  max_seq_length: 1024
  optim: "paged_adamw_32bit"
  seed: 42
  
  bf16: true
  fp16: false
  
  save_steps: 100
  cleanup_checkpoints: true
  
  logging_steps: 10
  wandb_enabled: true
  wandb_project: "pregnancy-qa-finetune"
  wandb_run_name: null

# ---------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------
evaluation:
  base_model: "meta-llama/Llama-3.2-3B-Instruct"
  adapter_path: "models/lora_weights"
  quantization: "8bit"
  eval_data: "data/qa_dataset/eval.json"
  batch_size: 16
  max_new_tokens: 256
  metrics:
    - rouge
    - bertscore
    - token_f1
  combined_weights:
    bertscore_f1: 0.40
    rougeL: 0.30
    token_f1: 0.20
    rouge1: 0.10
  output_dir: "reports"

# ---------------------------------------------------------------------
# STAGE 5: MODEL REGISTRATION
# ---------------------------------------------------------------------
registration:
  hf_username: null
  repo_name: null
  private: false

# ---------------------------------------------------------------------
# SERVING
# ---------------------------------------------------------------------
serving:
  vllm_host: "vllm"
  vllm_port: 8001
  api_port: 8000
  timeout: 60
  max_model_len: 1024
  gpu_memory_utilization: 0.2
  warmup_on_start: true
  api_token: null
```

---

## Project Structure

```
energyai/
├── config.yaml                 # Main configuration
├── pipeline.py                 # Python orchestrator
├── .env                        # Environment variables (not in git)
│
├── scripts/
│   ├── run_pipeline.sh         # Main entry point
│   ├── serve_vllm.sh           # vLLM startup script
│   └── test_serving.sh         # Integration tests
│
├── src/
│   ├── data_collector/         # Stage 1
│   ├── data_processor/         # Stage 2
│   ├── dataset_generator/      # Stage 3
│   ├── finetuner/              # Stage 4
│   ├── evaluator/              # Stage 4 (evaluation)
│   ├── registrar/              # Stage 5
│   ├── serving/                # Stage 6-7 (API)
│   └── docker/                 # Docker configs
│
├── data/                       # Generated data (not in git)
│   ├── raw/
│   ├── processed/
│   ├── chunks/
│   └── qa_dataset/
│
├── models/                     # Trained models (not in git)
│   ├── checkpoints/
│   └── lora_weights/
│
├── reports/                    # Evaluation reports
│   ├── eval_baseline.json
│   └── eval_finetuned.json
│
└── logs/                       # API logs
    ├── api.log
    └── requests.log
```

---