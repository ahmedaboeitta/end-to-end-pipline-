import argparse
import yaml
from registrar import ModelRegistrar


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    registrar = ModelRegistrar(config)
    record = registrar.run()
    
    action = "Updated" if record["is_update"] else "Created"
    print(f"\nStage 6 complete: {action} {record['repo_url']}")


if __name__ == "__main__":
    main()