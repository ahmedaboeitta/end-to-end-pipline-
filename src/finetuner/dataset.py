# src/finetuner/dataset.py

import json
from datasets import Dataset

SYSTEM_PROMPT = """You are a helpful assistant specializing in answering questions.

[ANSWER STYLE]
- Be concise and directly address the question
- Do NOT add explanations unless explicitly required
- Avoid examples, lists, or extra background unless asked
- Keep answers brief and factual"""

def load_qa_data(filepath: str) -> list[dict]:
    """Load QA pairs from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def format_chat(example: dict, tokenizer) -> dict:
    """Format QA pair to chat template."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]
    
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    
    return {"text": text}


def prepare_dataset(filepath: str, tokenizer) -> Dataset:
    """Load and format dataset for training."""
    data = load_qa_data(filepath)
    dataset = Dataset.from_list(data)
    
    dataset = dataset.map(
        lambda x: format_chat(x, tokenizer),
        remove_columns=dataset.column_names,
    )
    
    return dataset