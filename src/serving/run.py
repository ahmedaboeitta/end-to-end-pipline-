import argparse
import uvicorn
import yaml
from pathlib import Path


def load_config(path: str = "config.yaml") -> dict:
    """Load config, return empty dict if not found."""
    config_path = Path(path)
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def main():
    config = load_config()
    default_port = config.get("serving", {}).get("api_port", 8000)
    
    parser = argparse.ArgumentParser(description="Run the serving API")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=default_port, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    args = parser.parse_args()
    
    uvicorn.run(
        "src.serving.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()