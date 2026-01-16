import json
from pathlib import Path
from datetime import datetime
from inference import InferenceEngine
from metrics import calculate_all_metrics, calculate_combined_score
import torch 

class Evaluator:
    def __init__(self, config: dict):
        self.config = config
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_eval_data(self) -> tuple[list[str], list[str]]:
        """Load questions and reference answers from eval.json."""
        with open(self.config["eval_data"], "r") as f:
            data = json.load(f)
        
        questions = [item["question"] for item in data]
        references = [item["answer"] for item in data]
        
        return questions, references
    
    def run(
        self,
        base_model: str,
        adapter_path: str = None,
        quantization: str = None,
        output_name: str = "eval_results.json",
    ) -> dict:
        """Run full evaluation pipeline."""
        # Load data
        questions, references = self.load_eval_data()
        print(f"Loaded {len(questions)} eval samples")
        
        # Initialize inference engine
        print(f"Loading model: {base_model}")
        if adapter_path:
            print(f"Loading adapter: {adapter_path}")
        if quantization:
            print(f"Quantization: {quantization}")
        
        engine = InferenceEngine(
            base_model=base_model,
            adapter_path=adapter_path,
            quantization=quantization,
        )
        
        # Run inference
        print("Running inference...")
        predictions = engine.run_inference(
            questions=questions,
            batch_size=self.config["batch_size"],
            max_new_tokens=self.config["max_new_tokens"],
        )
        
        # Calculate metrics
        print("Calculating metrics...")
        metrics = calculate_all_metrics(predictions, references)
        
        # Combined score
        combined = calculate_combined_score(metrics, self.config["combined_weights"])
        metrics["combined_score"] = combined
        
        # Build report
        report = {
            "model": base_model,
            "adapter": adapter_path,
            "quantization": quantization,
            "timestamp": datetime.now().isoformat(),
            "num_samples": len(questions),
            "metrics": metrics,
            "predictions": [
                {
                    "question": q,
                    "reference": r,
                    "predicted": p,
                }
                for q, r, p in zip(questions, references, predictions)
            ],
        }
        
        # Save report
        output_path = self.output_dir / output_name
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        
        # Clear GPU memory
        del engine
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        print(f"Report saved: {output_path}")
        print(f"Combined score: {combined:.4f}")
        
        return report