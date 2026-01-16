#!/bin/bash
# =============================================================================
# vLLM Docker Startup Script
# Reads config.yaml and starts vLLM with those values
# =============================================================================

set -e

CONFIG_FILE="${CONFIG_FILE:-/app/config.yaml}"

# Check config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config not found: $CONFIG_FILE"
    exit 1
fi

# Read config using Python
eval $(python3 << EOF
import yaml

with open("$CONFIG_FILE") as f:
    config = yaml.safe_load(f)

ft = config["finetuning"]
sv = config["serving"]

print(f'MODEL="{ft["base_model"]}"')
print(f'LORA_PATH="/app/{ft["final_model_dir"]}"')
print(f'LORA_NAME="{ft["final_model_dir"].split("/")[-1]}"')
print(f'PORT="{sv["vllm_port"]}"')
print(f'MAX_LEN="{sv["max_model_len"]}"')
print(f'GPU_MEM="{sv["gpu_memory_utilization"]}"')
EOF
)

echo "=============================================="
echo "vLLM Docker Configuration"
echo "=============================================="
echo "Model:      $MODEL"
echo "LoRA:       $LORA_NAME"
echo "LoRA Path:  $LORA_PATH"
echo "Port:       $PORT"
echo "Max Length: $MAX_LEN"
echo "GPU Memory: $GPU_MEM"
echo "=============================================="

exec python3 -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port $PORT \
    --model $MODEL \
    --dtype float16 \
    --max-model-len $MAX_LEN \
    --gpu-memory-utilization $GPU_MEM \
    --enable-lora \
    --lora-modules $LORA_NAME=$LORA_PATH \
    --max-lora-rank 64