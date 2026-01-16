---
library_name: peft
base_model: meta-llama/Llama-3.2-3B-Instruct
tags:
- pregnancy
- qa
- lora
- fine-tuned
language:
- en
---

# pregnancy-qa-qlora-v1.1

Fine-tuned LoRA adapter for **pregnancy** domain, optimized for **qa** tasks.

**Version**: v1.1

## Model Details

- **Base Model**: [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- **Method**: QLoRA
- **Quantization**: 8bit
- **LoRA Rank (r)**: 16
- **LoRA Alpha**: 32

## Evaluation Metrics

- **rouge1**: 0.4368
- **rouge2**: 0.2375
- **rougeL**: 0.3860
- **bertscore_precision**: 0.9127
- **bertscore_recall**: 0.9041
- **bertscore_f1**: 0.9081
- **token_f1**: 0.3778
- **combined_score**: 0.5983


## Training Details

- **Domain**: pregnancy
- **Training Data**: Synthetic QA pairs generated from domain PDFs
- **Epochs**: 20
- **Learning Rate**: 0.0001
- **Batch Size**: 8

## Last Updated

2026-01-12 15:14:12
