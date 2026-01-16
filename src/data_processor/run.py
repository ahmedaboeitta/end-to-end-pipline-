from processor import Processor

# Config
config = {
    "paths": {
        "raw_data": "/mnt/nfs/ahmed/base/data/raw",
        "processed_data": "/mnt/nfs/ahmed/base/data/processed",
        "chunks_data": "/mnt/nfs/ahmed/base/data/chunks"
    },
    "min_tokens": 20,
    "target_tokens": 512,
    "max_tokens": 500
}


def main():
    processor = Processor(config)
    chunks = processor.run()
    
    print("\n=== Stage 2 Complete ===")
    print(f"Total chunks: {len(chunks)}")


if __name__ == "__main__":
    main()