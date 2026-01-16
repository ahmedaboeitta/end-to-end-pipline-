import argparse
import yaml
from trainer import Finetuner


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    finetuner = Finetuner(config)
    finetuner.run()
    
    print("Stage 4 complete")


if __name__ == "__main__":
    main()