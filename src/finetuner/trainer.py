import os
import torch
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from dotenv import load_dotenv

from dataset import prepare_dataset
from lora_config import get_quantization_config, get_lora_config

load_dotenv()


class Finetuner:
    def __init__(self, config: dict):
        self.config = config
        self.ft_config = config["finetuning"]
    
    def _generate_run_name(self) -> str:
        """Generate descriptive run name from config."""
        model_short = self.ft_config["base_model"].split("/")[-1]
        quant = self.ft_config.get("quantization") or "fp16"
        lora_r = self.ft_config["lora_r"]
        lr = self.ft_config["learning_rate"]
        timestamp = datetime.now().strftime("%m%d_%H%M")
        
        return f"{model_short}_{quant}_r{lora_r}_lr{lr}_{timestamp}"
    
    def _setup_wandb(self):
        """Initialize W&B if enabled."""
        if not self.ft_config.get("wandb_enabled", True):
            return False
        
        import wandb
        
        api_key = os.getenv("WANDB_API_KEY")
        if not api_key:
            print("Warning: WANDB_API_KEY not set, disabling W&B")
            return False
        
        run_name = self.ft_config.get("wandb_run_name") or self._generate_run_name()
        
        wandb.login(key=api_key)
        wandb.init(
            project=self.ft_config["wandb_project"],
            name=run_name,
            config=self.ft_config,
        )
        return True
    
    def run(self):
        """Run full fine-tuning pipeline."""
        # Setup W&B
        wandb_enabled = self._setup_wandb()
        
        # Set seed
        seed = self.ft_config.get("seed", 42)
        torch.manual_seed(seed)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.ft_config["base_model"],
            token=os.getenv("HF_TOKEN"),
        )
        
        # Set pad token (prefer dedicated pad token if available)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
        tokenizer.padding_side = "right"
        
        # Load model with quantization
        bnb_config = get_quantization_config(self.ft_config.get("quantization"))
        
        model = AutoModelForCausalLM.from_pretrained(
            self.ft_config["base_model"],
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16 if not bnb_config else None,
            token=os.getenv("HF_TOKEN"),
        )
        
        # Prepare for k-bit training if quantized
        if bnb_config:
            model = prepare_model_for_kbit_training(model)
        
        # Apply LoRA
        lora_config = get_lora_config(self.ft_config)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # Load dataset
        dataset = prepare_dataset(self.ft_config["train_data"], tokenizer)
        print(f"Training samples: {len(dataset)}")
        
        # Determine precision
        bf16 = self.ft_config.get("bf16", True)
        fp16 = self.ft_config.get("fp16", False)
        
        # Training strategy
        strategy = self.ft_config.get("strategy", "epoch")

        if strategy == "epoch":
            sft_config = SFTConfig(
                output_dir=self.ft_config["output_dir"],
                num_train_epochs=self.ft_config["epochs"],
                max_steps=-1,
                per_device_train_batch_size=self.ft_config["batch_size"],
                gradient_accumulation_steps=self.ft_config["gradient_accumulation_steps"],
                
                # Learning rate & scheduler
                learning_rate=self.ft_config["learning_rate"],
                lr_scheduler_type=self.ft_config.get("lr_scheduler_type", "cosine"),
                warmup_ratio=self.ft_config.get("warmup_ratio", 0.1),
                weight_decay=self.ft_config.get("weight_decay", 0.01),
                max_grad_norm=self.ft_config.get("max_grad_norm", 1.0),
                
                # Precision
                bf16=bf16,
                fp16=fp16,
                
                # Checkpoints - save each epoch
                save_strategy="epoch",
                
                # Logging
                logging_steps=self.ft_config["logging_steps"],
                report_to="wandb" if wandb_enabled else "none",
                
                # Optimizer
                optim=self.ft_config.get("optim", "paged_adamw_32bit"),
                
                # SFT specific
                max_length=self.ft_config.get("max_seq_length", 1024),
                dataset_text_field="text",
                packing=False,
                
                # Misc
                seed=seed,
                gradient_checkpointing=False,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        else:  # steps
            sft_config = SFTConfig(
                output_dir=self.ft_config["output_dir"],
                max_steps=self.ft_config["max_steps"],
                per_device_train_batch_size=self.ft_config["batch_size"],
                gradient_accumulation_steps=self.ft_config["gradient_accumulation_steps"],
                
                # Learning rate & scheduler
                learning_rate=self.ft_config["learning_rate"],
                lr_scheduler_type=self.ft_config.get("lr_scheduler_type", "cosine"),
                warmup_ratio=self.ft_config.get("warmup_ratio", 0.1),
                weight_decay=self.ft_config.get("weight_decay", 0.01),
                max_grad_norm=self.ft_config.get("max_grad_norm", 1.0),
                
                # Precision
                bf16=bf16,
                fp16=fp16,
                
                # Checkpoints - save every N steps
                save_strategy="steps",
                save_steps=self.ft_config["save_steps"],
                
                # Logging
                logging_steps=self.ft_config["logging_steps"],
                report_to="wandb" if wandb_enabled else "none",
                
                # Optimizer
                optim=self.ft_config.get("optim", "paged_adamw_32bit"),
                
                # SFT specific
                max_length=self.ft_config.get("max_seq_length", 1024),
                dataset_text_field="text",
                packing=False,
                
                # Misc
                seed=seed,
                gradient_checkpointing=False,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        
        # Trainer
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            processing_class=tokenizer,
            args=sft_config,
        )
        
        # Train
        trainer.train()
        
        # Save final adapter
        final_path = os.path.join(self.ft_config["output_dir"], "final")
        model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        
        # Close W&B
        if wandb_enabled:
            import wandb
            wandb.finish()
        
        # Clear GPU memory
        del model
        del trainer
        torch.cuda.empty_cache()
        
        print(f"Training complete. Checkpoints saved to {self.ft_config['output_dir']}")