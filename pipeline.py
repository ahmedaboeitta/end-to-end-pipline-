#!/usr/bin/env python3
"""
LLM Fine-tuning Pipeline Orchestrator

Usage:
    python pipeline.py                     # Run all stages (1-5)
    python pipeline.py --stage 3           # Run specific stage
    python pipeline.py --start 2 --end 4   # Run stages 2, 3, 4
    python pipeline.py --stage 4b          # Run sub-stage (finetune only)
"""

import argparse
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# Configuration
# =============================================================================

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# =============================================================================
# Prerequisites Check
# =============================================================================

PREREQUISITES = {
    "1": [],
    "2": ["data/raw/metadata.json"],
    "3": ["data/chunks/chunks.json"],
    "4a": ["data/qa_dataset/eval.json"],
    "4b": ["data/qa_dataset/train.json"],
    "4c": ["models/checkpoints"],
    "4d": ["models/lora_weights/adapter_config.json"],
    "5": ["models/lora_weights/adapter_config.json"],
}


def check_prerequisites(stage: str, config: dict):
    """Check if required files exist before running a stage."""
    if stage not in PREREQUISITES:
        return
    
    missing = []
    for path in PREREQUISITES[stage]:
        full_path = Path(path)
        if not full_path.exists():
            missing.append(path)
    
    if missing:
        print(f"\n❌ Stage {stage} prerequisites not met:")
        for path in missing:
            print(f"   Missing: {path}")
        print(f"\n   Run previous stages first.")
        sys.exit(1)


# =============================================================================
# Stage Functions
# =============================================================================

def run_stage_1(config: dict):
    """Stage 1: Data Collection - Search, download, extract PDFs"""
    print("📥 Stage 1: Data Collection")
    
    sys.path.insert(0, "src/data_collector")
    from collector import Collector
    
    stage_config = {
        "paths": config["paths"],
        "num_results": config["data_collector"]["num_results"],
        "mode": config["data_collector"]["mode"],
    }
    
    collector = Collector(stage_config)
    docs = collector.run(config["data_collector"]["query"])
    
    print(f"✅ Stage 1 complete: {len(docs)} documents collected")
    return docs


def run_stage_2(config: dict):
    """Stage 2: Data Processing - Clean, chunk, filter"""
    print("🔧 Stage 2: Data Processing")
    
    sys.path.insert(0, "src/data_processor")
    from processor import Processor
    
    stage_config = {
        "paths": {
            "raw_data": config["paths"]["raw_data"],
            "processed_data": config["paths"]["processed_data"],
            "chunks_data": config["paths"]["chunks"],
        },
        "min_tokens": config["data_processor"]["min_tokens"],
        "target_tokens": config["data_processor"]["target_tokens"],
        "max_tokens": config["data_processor"]["max_tokens"],
    }
    
    processor = Processor(stage_config)
    chunks = processor.run()
    
    print(f"✅ Stage 2 complete: {len(chunks)} chunks created")
    return chunks


def run_stage_3(config: dict):
    """Stage 3: Dataset Generation - Generate QA pairs"""
    print("📝 Stage 3: Dataset Generation")
    
    sys.path.insert(0, "src/dataset_generator")
    from run import run
    
    run(
        chunks_path=os.path.join(config["paths"]["chunks"], "chunks.json"),
        output_dir=config["paths"]["qa_dataset"],
        domain=config["domain"],
        use_case=config["use_case"],
        api_key=os.getenv("GEMINI_API_KEY"),
        max_tokens=config["dataset_generator"]["max_tokens"],
        train_ratio=config["dataset_generator"]["train_ratio"],
        model=config["dataset_generator"]["model"],
    )
    
    print("✅ Stage 3 complete")


def run_stage_4a(config: dict):
    """Stage 4a: Baseline Evaluation"""
    print("📊 Stage 4a: Baseline Evaluation")
    
    sys.path.insert(0, "src/evaluator")
    from evaluator import Evaluator
    
    eval_config = config["evaluation"]
    evaluator = Evaluator(eval_config)
    
    report = evaluator.run(
        base_model=eval_config["base_model"],
        adapter_path=None,
        quantization=config["finetuning"]["quantization"],
        output_name="eval_baseline.json",
    )
    
    print(f"✅ Stage 4a complete: baseline score = {report['metrics']['combined_score']:.4f}")
    return report


def run_stage_4b(config: dict):
    """Stage 4b: Fine-tuning"""
    print("🎯 Stage 4b: Fine-tuning")
    
    sys.path.insert(0, "src/finetuner")
    from trainer import Finetuner
    
    finetuner = Finetuner(config)
    finetuner.run()
    
    print("✅ Stage 4b complete")


def run_stage_4c(config: dict):
    """Stage 4c: Checkpoint Selection"""
    print("🏆 Stage 4c: Checkpoint Selection")
    
    sys.path.insert(0, "src/finetuner")
    sys.path.insert(0, "src/evaluator")
    from checkpoint_selector import CheckpointSelector
    
    selector = CheckpointSelector(config)
    report = selector.run()
    
    if report:
        print(f"✅ Stage 4c complete: best score = {report['best_score']:.4f}")
    return report


def run_stage_4d(config: dict):
    """Stage 4d: Final Model Evaluation"""
    print("📊 Stage 4d: Final Model Evaluation")
    
    sys.path.insert(0, "src/evaluator")
    from evaluator import Evaluator
    
    eval_config = config["evaluation"]
    evaluator = Evaluator(eval_config)
    
    report = evaluator.run(
        base_model=eval_config["base_model"],
        adapter_path=config["finetuning"]["final_model_dir"],
        quantization=config["finetuning"]["quantization"],
        output_name="eval_finetuned.json",
    )
    
    print(f"✅ Stage 4d complete: finetuned score = {report['metrics']['combined_score']:.4f}")
    return report


def run_stage_5(config: dict):
    """Stage 5: Model Registration"""
    print("📦 Stage 5: Model Registration")
    
    sys.path.insert(0, "src/registrar")
    from registrar import ModelRegistrar
    
    registrar = ModelRegistrar(config)
    record = registrar.run()
    
    print(f"✅ Stage 5 complete: {record['repo_url']}")
    return record


# =============================================================================
# Stage 4 Combined Runner
# =============================================================================

def run_stage_4(config: dict):
    """Stage 4: Full Training Pipeline (4a → 4b → 4c → 4d)"""
    print("=" * 50)
    print("🚀 Stage 4: Training Pipeline")
    print("=" * 50)
    
    # 4a: Baseline (optional but useful for comparison)
    run_stage_4a(config)
    print()
    
    # 4b: Fine-tune
    run_stage_4b(config)
    print()
    
    # 4c: Select best checkpoint
    run_stage_4c(config)
    print()
    
    # 4d: Evaluate final model
    run_stage_4d(config)
    
    print("=" * 50)
    print("✅ Stage 4 complete")
    print("=" * 50)


# =============================================================================
# Stage Registry
# =============================================================================

STAGES = {
    "1": run_stage_1,
    "2": run_stage_2,
    "3": run_stage_3,
    "4": run_stage_4,
    "4a": run_stage_4a,
    "4b": run_stage_4b,
    "4c": run_stage_4c,
    "4d": run_stage_4d,
    "5": run_stage_5,
}

STAGE_NAMES = {
    "1": "Data Collection",
    "2": "Data Processing",
    "3": "Dataset Generation",
    "4": "Training (full)",
    "4a": "Baseline Evaluation",
    "4b": "Fine-tuning",
    "4c": "Checkpoint Selection",
    "4d": "Final Evaluation",
    "5": "Model Registration",
}


# =============================================================================
# Main
# =============================================================================

def print_stages():
    """Print available stages."""
    print("\nAvailable stages:")
    for stage, name in STAGE_NAMES.items():
        print(f"  {stage}: {name}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="LLM Fine-tuning Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py                     # Run all stages (1-5)
  python pipeline.py --stage 3           # Run stage 3 only
  python pipeline.py --start 2 --end 4   # Run stages 2, 3, 4
  python pipeline.py --stage 4b          # Run fine-tuning only
  python pipeline.py --list              # Show all stages
        """
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument("--stage", type=str, help="Run specific stage (1, 2, 3, 4, 4a, 4b, 4c, 4d, 5)")
    parser.add_argument("--start", type=int, help="Start from stage N")
    parser.add_argument("--end", type=int, help="End at stage N")
    parser.add_argument("--list", action="store_true", help="List available stages")
    args = parser.parse_args()
    
    # List stages
    if args.list:
        print_stages()
        return
    
    # Load config
    config = load_config(args.config)
    
    # Determine which stages to run
    if args.stage:
        # Single stage (including sub-stages like 4a, 4b)
        if args.stage not in STAGES:
            print(f"❌ Unknown stage: {args.stage}")
            print_stages()
            sys.exit(1)
        
        stages_to_run = [args.stage]
    
    elif args.start or args.end:
        # Range of stages
        start = args.start or 1
        end = args.end or 5
        stages_to_run = [str(i) for i in range(start, end + 1)]
    
    else:
        # Default: run all main stages (1-5)
        stages_to_run = ["1", "2", "3", "4", "5"]
    
    # Run stages
    print("=" * 60)
    print("🚀 LLM Fine-tuning Pipeline")
    print(f"   Config: {args.config}")
    print(f"   Stages: {', '.join(stages_to_run)}")
    print("=" * 60)
    print()
    
    for stage in stages_to_run:
        check_prerequisites(stage, config)
        
        print(f"\n{'─' * 40}")
        STAGES[stage](config)
        print()
    
    print("=" * 60)
    print("🎉 Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()