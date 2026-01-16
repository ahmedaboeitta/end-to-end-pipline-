import os
from collector import Collector

# Set API key
os.environ["SERP_API_KEY"] = os.getenv("SERP_API_KEY")

# Config
config = {
    "paths": {
        "raw_data": "/mnt/nfs/ahmed/base/data/raw",
        "processed_data": "/mnt/nfs/ahmed/base/data/processed"
    },
    "num_results": 20,
    "mode": "pdf"
}

def main():
    query = "pregnancy"  # Can be changed or passed as argument
    
    collector = Collector(config)
    docs = collector.run(query)
    
    print("\n=== Stage 1 Complete ===")
    print(f"Documents collected: {len(docs)}")
    print(f"Raw files: {config['paths']['raw_data']}")
    print(f"Processed files: {config['paths']['processed_data']}")

if __name__ == "__main__":
    main()