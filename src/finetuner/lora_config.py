import torch
from peft import LoraConfig, TaskType
from transformers import BitsAndBytesConfig


def get_quantization_config(quantization: str) -> BitsAndBytesConfig | None:
    """Get BitsAndBytes config for quantization."""
    if quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif quantization == "8bit":
        return BitsAndBytesConfig(
            load_in_8bit=True,
        )
    return None


def get_lora_config(config: dict) -> LoraConfig:
    """Get LoRA config from training config."""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
        bias="none",
    )