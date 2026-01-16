#!/bin/bash
# =============================================================================
# vLLM Server Startup Script
# =============================================================================
# Usage:
#   ./scripts/serve_vllm.sh              # With LoRA adapter
#   ./scripts/serve_vllm.sh --no-lora    # Base model only
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${PROJECT_ROOT}/config.yaml"

# Check config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config not found: $CONFIG_FILE"
    exit 1
fi

# Load config using Python
eval $(python3 << EOF
import yaml

with open("$CONFIG_FILE", "r") as f:
    config = yaml.safe_load(f)

# From finetuning section
print(f'MODEL_NAME="{config["finetuning"]["base_model"]}"')
print(f'LORA_ADAPTER_PATH="{config["finetuning"]["final_model_dir"]}"')

# From serving section
print(f'VLLM_PORT="{config["serving"]["vllm_port"]}"')
print(f'MAX_MODEL_LEN="{config["serving"]["max_model_len"]}"')
print(f'GPU_MEMORY_UTILIZATION="{config["serving"]["gpu_memory_utilization"]}"')
EOF
)

# Adapter name = folder name
LORA_ADAPTER_NAME=$(basename "$LORA_ADAPTER_PATH")

# Check for --no-lora flag
NO_LORA=false
if [[ "$1" == "--no-lora" ]]; then
    NO_LORA=true
fi

# Export HF token if available
if [ -n "$HF_TOKEN" ]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

# Print config
echo "=============================================="
echo "vLLM Server Configuration"
echo "=============================================="
echo "Model:         $MODEL_NAME"
echo "Port:          $VLLM_PORT"
echo "Max Length:    $MAX_MODEL_LEN"
echo "GPU Memory:    $GPU_MEMORY_UTILIZATION"
echo "LoRA:          $([ "$NO_LORA" = true ] && echo 'disabled' || echo $LORA_ADAPTER_NAME)"
echo "LoRA Path:     $LORA_ADAPTER_PATH"
echo "=============================================="

# Check if LoRA path exists
if [ "$NO_LORA" = false ] && [ ! -d "$LORA_ADAPTER_PATH" ]; then
    echo "ERROR: LoRA adapter path not found: $LORA_ADAPTER_PATH"
    echo "Use --no-lora to start without adapter."
    exit 1
fi

# Run vLLM
if [ "$NO_LORA" = true ]; then
    exec python -m vllm.entrypoints.openai.api_server \
        --host 0.0.0.0 \
        --port $VLLM_PORT \
        --model $MODEL_NAME \
        --dtype float16 \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $GPU_MEMORY_UTILIZATION
else
    exec python -m vllm.entrypoints.openai.api_server \
        --host 0.0.0.0 \
        --port $VLLM_PORT \
        --model $MODEL_NAME \
        --dtype float16 \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
        --enable-lora \
        --lora-modules $LORA_ADAPTER_NAME=$LORA_ADAPTER_PATH \
        --max-lora-rank 64
fi