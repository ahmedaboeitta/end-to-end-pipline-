import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
import os
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are a helpful assistant specializing in answering questions.

[ANSWER STYLE]
- Be concise and directly address the question
- Do NOT add explanations unless explicitly required
- Avoid examples, lists, or extra background unless asked
- Keep answers brief and factual"""

class InferenceEngine:
    def __init__(
        self,
        base_model: str,
        adapter_path: str = None,
        quantization: str = None,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, token=os.getenv("HF_TOKEN"))
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        # Quantization config
        bnb_config = None
        if quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif quantization == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            dtype=torch.bfloat16 if not bnb_config else None,
            token=os.getenv("HF_TOKEN")
        )
        
        # Load adapter if provided
        if adapter_path:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        
        self.model.eval()
    
    def generate_batch(
        self,
        questions: list[str],
        max_new_tokens: int = 256,
    ) -> list[str]:
        """Generate answers for a batch of questions."""
        # Format as chat
        prompts = []
        for q in questions:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt)
        
        # Tokenize
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only new tokens
        answers = []
        for i, output in enumerate(outputs):
            input_len = inputs["input_ids"][i].shape[0]
            answer = self.tokenizer.decode(output[input_len:], skip_special_tokens=True)
            answers.append(answer.strip())
        
        return answers
    
    def run_inference(
        self,
        questions: list[str],
        batch_size: int = 4,
        max_new_tokens: int = 256,
    ) -> list[str]:
        """Run batched inference on all questions."""
        all_answers = []
        
        for i in tqdm(range(0, len(questions), batch_size), desc="Inference"):
            batch = questions[i : i + batch_size]
            answers = self.generate_batch(batch, max_new_tokens)
            all_answers.extend(answers)
        
        return all_answers