import sys
import json
import shutil
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "evaluator"))
from evaluator import Evaluator


class CheckpointSelector:
    def __init__(self, config: dict):
        self.config = config
        self.ft_config = config["finetuning"]
        self.eval_config = config["evaluation"]
        self.checkpoint_dir = Path(self.ft_config["output_dir"])
        self.final_dir = Path(self.ft_config["final_model_dir"])
    
    def find_checkpoints(self) -> list[Path]:
        """Find all checkpoint directories."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint-*"))
        
        # Also include final if exists
        final = self.checkpoint_dir / "final"
        if final.exists():
            checkpoints.append(final)
        
        return sorted(checkpoints)
    
    def clear_memory(self):
        """Clear GPU memory."""
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    def evaluate_checkpoint(self, checkpoint_path: Path) -> dict:
        """Run evaluation on a single checkpoint."""
        print(f"\nEvaluating: {checkpoint_path.name}")
        
        evaluator = Evaluator(self.eval_config)
        report = evaluator.run(
            base_model=self.ft_config["base_model"],
            adapter_path=str(checkpoint_path),
            quantization=self.ft_config.get("quantization"),
            output_name=f"eval_{checkpoint_path.name}.json",
        )
        
        # Clear memory after evaluation
        self.clear_memory()
        
        return {
            "checkpoint": str(checkpoint_path),
            "combined_score": report["metrics"]["combined_score"],
            "metrics": report["metrics"],
        }
    
    def run(self) -> dict:
        """Evaluate all checkpoints, select best, cleanup."""
        checkpoints = self.find_checkpoints()
        
        if not checkpoints:
            print("No checkpoints found")
            return None
        
        print(f"Found {len(checkpoints)} checkpoints")
        
        # Evaluate each
        results = []
        for ckpt in checkpoints:
            result = self.evaluate_checkpoint(ckpt)
            results.append(result)
            print(f"  {ckpt.name}: {result['combined_score']:.4f}")
        
        # Find best
        best = max(results, key=lambda x: x["combined_score"])
        best_path = Path(best["checkpoint"])
        print(f"\nBest checkpoint: {best_path.name} (score: {best['combined_score']:.4f})")
        
        # Copy to final dir
        if self.final_dir.exists():
            shutil.rmtree(self.final_dir)
        shutil.copytree(best_path, self.final_dir)
        print(f"Copied to: {self.final_dir}")
        
        # Cleanup if enabled
        if self.ft_config.get("cleanup_checkpoints", False):
            for ckpt in checkpoints:
                if ckpt != best_path:
                    shutil.rmtree(ckpt)
                    print(f"Deleted: {ckpt.name}")
        
        # Save selection report
        report = {
            "best_checkpoint": str(best_path),
            "best_score": best["combined_score"],
            "all_results": results,
        }
        report_path = Path(self.eval_config["output_dir"]) / "checkpoint_selection.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        return report