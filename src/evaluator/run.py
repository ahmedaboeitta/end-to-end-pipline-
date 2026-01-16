import argparse
import yaml
from evaluator import Evaluator


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_baseline(config: dict):
    """Run evaluation on baseline model (no adapter, no quantization)."""
    eval_config = config["evaluation"]
    
    evaluator = Evaluator(eval_config)
    report = evaluator.run(
        base_model=eval_config["base_model"],
        adapter_path=None,
        quantization=None,
        output_name="eval_baseline.json",
    )
    
    return report


def run_finetuned(config: dict):
    """Run evaluation on fine-tuned model (from config)."""
    eval_config = config["evaluation"]
    
    evaluator = Evaluator(eval_config)
    report = evaluator.run(
        base_model=eval_config["base_model"],
        adapter_path=eval_config["adapter_path"],
        quantization=eval_config["quantization"],
        output_name="eval_finetuned.json",
    )
    
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--mode", type=str, choices=["baseline", "finetuned"], required=True)
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.mode == "baseline":
        run_baseline(config)
    else:
        run_finetuned(config)


if __name__ == "__main__":
    main()