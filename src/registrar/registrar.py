import os
import json
from pathlib import Path
from datetime import datetime
from huggingface_hub import HfApi, create_repo, upload_folder, repo_exists
from dotenv import load_dotenv

load_dotenv()


class ModelRegistrar:
    def __init__(self, config: dict):
        self.config = config
        self.ft_config = config["finetuning"]
        self.reg_config = config["registration"]
        self.eval_config = config["evaluation"]
        
        # Resolve HF username
        self.hf_username = self.reg_config.get("hf_username") or os.getenv("HF_USERNAME")
        if not self.hf_username:
            raise ValueError("HF username required. Set in config or HF_USERNAME env var")
        
        # HF API
        self.api = HfApi(token=os.getenv("HF_TOKEN"))
        
        # Resolve version and repo name
        self.version = self._resolve_version()
        self.repo_name = self._generate_repo_name()
        self.repo_id = f"{self.hf_username}/{self.repo_name}"
        
        # Paths
        self.model_dir = Path(self.ft_config["final_model_dir"])
        self.reports_dir = Path(self.eval_config["output_dir"])
    
    def _get_base_repo_name(self) -> str:
        """Generate base repo name without version."""
        domain = self.config["domain"].lower().replace(" ", "-")
        use_case = self.config["use_case"].lower()
        
        quantization = self.ft_config.get("quantization")
        if quantization in ["4bit", "8bit"]:
            method = "qlora"
        else:
            method = "lora"
        
        return f"{domain}-{use_case}-{method}"
    
    def _repo_exists(self, repo_name: str) -> bool:
        """Check if repo exists on HF Hub."""
        repo_id = f"{self.hf_username}/{repo_name}"
        try:
            return repo_exists(repo_id, token=os.getenv("HF_TOKEN"))
        except Exception:
            return False
    
    def _get_next_version(self) -> str:
        """Find next available version by checking existing repos."""
        base_name = self._get_base_repo_name()
        
        # Start from 1.0 and find next available
        major = 1
        minor = 0
        
        while True:
            version = f"v{major}.{minor}"
            repo_name = f"{base_name}-{version}"
            
            if not self._repo_exists(repo_name):
                return version
            
            # Increment
            minor += 1
            if minor > 9:
                major += 1
                minor = 0
            
            # Safety limit
            if major > 99:
                raise ValueError("Too many versions")
    
    def _resolve_version(self) -> str:
        """Resolve version from config or auto-increment."""
        configured_version = self.reg_config.get("version")
        
        if configured_version:
            # Normalize format
            v = str(configured_version)
            if v.startswith("v"):
                v = v[1:]
            
            # Ensure X.Y format
            if "." not in v:
                v = f"{v}.0"
            
            return f"v{v}"
        else:
            return self._get_next_version()
    
    def _generate_repo_name(self) -> str:
        """Generate full repo name with version."""
        base_name = self._get_base_repo_name()
        return f"{base_name}-{self.version}"
    
    def _is_update(self) -> bool:
        """Check if this is an update to existing repo."""
        return self._repo_exists(self.repo_name)
    
    def _load_metrics(self) -> dict:
        """Load evaluation metrics."""
        eval_path = self.reports_dir / "eval_finetuned.json"
        
        if eval_path.exists():
            with open(eval_path, "r") as f:
                report = json.load(f)
                return report.get("metrics", {})
        
        # Try checkpoint selection report
        checkpoint_path = self.reports_dir / "checkpoint_selection.json"
        if checkpoint_path.exists():
            with open(checkpoint_path, "r") as f:
                report = json.load(f)
                return {"combined_score": report.get("best_score", 0)}
        
        return {}
    
    def _generate_model_card(self, metrics: dict, is_update: bool) -> str:
        """Generate README model card."""
        
        base_model = self.ft_config["base_model"]
        quantization = self.ft_config.get("quantization") or "none"
        lora_r = self.ft_config["lora_r"]
        lora_alpha = self.ft_config["lora_alpha"]
        
        # Format metrics
        metrics_str = ""
        for key, value in metrics.items():
            if isinstance(value, float):
                metrics_str += f"- **{key}**: {value:.4f}\n"
        
        update_note = " (Updated)" if is_update else ""
        
        card = f"""---
library_name: peft
base_model: {base_model}
tags:
- {self.config["domain"]}
- {self.config["use_case"]}
- lora
- fine-tuned
language:
- en
---

# {self.repo_name}

Fine-tuned LoRA adapter for **{self.config["domain"]}** domain, optimized for **{self.config["use_case"]}** tasks.

**Version**: {self.version}{update_note}

## Model Details

- **Base Model**: [{base_model}](https://huggingface.co/{base_model})
- **Method**: {"QLoRA" if quantization in ["4bit", "8bit"] else "LoRA"}
- **Quantization**: {quantization}
- **LoRA Rank (r)**: {lora_r}
- **LoRA Alpha**: {lora_alpha}

## Evaluation Metrics

{metrics_str if metrics_str else "No metrics available."}

## Training Details

- **Domain**: {self.config["domain"]}
- **Training Data**: Synthetic QA pairs generated from domain PDFs
- **Epochs**: {self.ft_config.get("epochs", "N/A")}
- **Learning Rate**: {self.ft_config.get("learning_rate", "N/A")}
- **Batch Size**: {self.ft_config.get("batch_size", "N/A")}

## Last Updated

{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
        return card
    
    def run(self) -> dict:
        """Run full registration pipeline."""
        is_update = self._is_update()
        action = "Updating" if is_update else "Creating"
        
        print(f"{action} model: {self.repo_id}")
        print(f"Version: {self.version}")
        
        # Check model exists locally
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model not found: {self.model_dir}")
        
        # Load metrics
        metrics = self._load_metrics()
        print(f"Loaded metrics: {list(metrics.keys())}")
        
        # Create repo (or use existing)
        private = self.reg_config.get("private", False)
        create_repo(
            repo_id=self.repo_id,
            token=os.getenv("HF_TOKEN"),
            private=private,
            exist_ok=True,
        )
        
        # Generate and save model card
        model_card = self._generate_model_card(metrics, is_update)
        readme_path = self.model_dir / "README.md"
        readme_path.write_text(model_card)
        print("Generated model card")
        
        # Upload
        commit_msg = f"Update {self.version}" if is_update else f"Initial upload {self.version}"
        upload_folder(
            folder_path=str(self.model_dir),
            repo_id=self.repo_id,
            token=os.getenv("HF_TOKEN"),
            commit_message=commit_msg,
        )
        
        print(f"{action} complete: https://huggingface.co/{self.repo_id}")
        
        # Save registration record locally
        record = {
            "repo_id": self.repo_id,
            "repo_url": f"https://huggingface.co/{self.repo_id}",
            "version": self.version,
            "is_update": is_update,
            "base_model": self.ft_config["base_model"],
            "metrics": metrics,
            "registered_at": datetime.now().isoformat(),
            "private": private,
        }
        
        record_path = self.reports_dir / "registration.json"
        with open(record_path, "w") as f:
            json.dump(record, f, indent=2)
        print(f"Registration record saved: {record_path}")
        
        return record