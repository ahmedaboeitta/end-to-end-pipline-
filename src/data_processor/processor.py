import json
from pathlib import Path
from chunker import split_into_sections, process_sections


class Processor:
    def __init__(self, config: dict):
        self.config = config
        self.processed_dir = Path(config["paths"]["processed_data"])
        self.raw_dir = Path(config["paths"]["raw_data"])
        self.chunks_dir = Path(config["paths"]["chunks_data"])
        self.metadata_path = self.raw_dir / "metadata.json"
        
        # Token thresholds
        self.min_tokens = config.get("min_tokens", 20)
        self.target_tokens = config.get("target_tokens", 512)
        self.max_tokens = config.get("max_tokens", 500)
        
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
    
    def load_metadata(self) -> list[dict]:
        """Load source metadata."""
        with open(self.metadata_path, 'r') as f:
            return json.load(f)
    
    def process_document(self, doc_id: str, metadata: dict) -> list[dict]:
        """Process a single document into chunks."""
        md_path = self.processed_dir / f"{doc_id}.md"
        
        if not md_path.exists():
            print(f"Warning: {md_path} not found, skipping")
            return []
        
        # Read markdown
        content = md_path.read_text(encoding='utf-8')
        
        # Split into sections
        sections = split_into_sections(content)
        
        # Clean and count tokens
        sections = process_sections(sections)
        
        # Build chunks with metadata
        chunks = []
        for i, section in enumerate(sections):
            chunk = {
                "id": f"{doc_id}_chunk_{i:02d}",
                "source_id": doc_id,
                "source_url": metadata.get("url", ""),
                "source_title": metadata.get("title", ""),
                "section_title": section["section_title"],
                "content": section["content"],
                "token_count": section["token_count"]
            }
            chunks.append(chunk)
        
        return chunks
    
    def analyze_distribution(self, all_chunks: list[dict]) -> dict:
        """Analyze token distribution across all chunks."""
        token_counts = [c["token_count"] for c in all_chunks]
        
        if not token_counts:
            return {}
        
        token_counts_sorted = sorted(token_counts)
        n = len(token_counts)
        
        stats = {
            "total_chunks": n,
            "min": min(token_counts),
            "max": max(token_counts),
            "mean": sum(token_counts) / n,
            "median": token_counts_sorted[n // 2],
            "p10": token_counts_sorted[int(n * 0.1)] if n >= 10 else token_counts_sorted[0],
            "p90": token_counts_sorted[int(n * 0.9)] if n >= 10 else token_counts_sorted[-1],
        }
        
        # Count by buckets
        stats["under_50"] = sum(1 for t in token_counts if t < 50)
        stats["50_to_200"] = sum(1 for t in token_counts if 50 <= t < 200)
        stats["200_to_500"] = sum(1 for t in token_counts if 200 <= t < 500)
        stats["500_to_1000"] = sum(1 for t in token_counts if 500 <= t < 1000)
        stats["over_1000"] = sum(1 for t in token_counts if t >= 1000)
        
        return stats
    
    def run(self) -> list[dict]:
        """Run full processing pipeline."""
        # Load metadata
        metadata_list = self.load_metadata()
        metadata_map = {m["id"]: m for m in metadata_list} # noqa: F841
        
        print(f"Processing {len(metadata_list)} documents...")
        
        # Process all documents
        all_chunks = []
        for meta in metadata_list:
            doc_id = meta["id"]
            chunks = self.process_document(doc_id, meta)
            all_chunks.extend(chunks)
            print(f"  {doc_id}: {len(chunks)} sections")
        
        # Analyze distribution
        print("\n=== Token Distribution (before filtering) ===")
        stats = self.analyze_distribution(all_chunks)
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Filter outliers
        filtered_chunks = []
        for c in all_chunks:
            # Skip single-letter section titles (index sections)
            if len(c["section_title"].strip()) <= 1:
                continue
            # Skip very short sections
            if c["token_count"] < self.min_tokens:
                continue
            # Skip very long sections
            if c["token_count"] > self.max_tokens:
                continue
            filtered_chunks.append(c)
        
        print(f"\nFiltered: {len(all_chunks)} → {len(filtered_chunks)} chunks (min_tokens={self.min_tokens}, max_tokens={self.max_tokens}, excluding single-letter titles)")
        
        # Save chunks
        output_path = self.chunks_dir / "chunks.json"
        with open(output_path, 'w') as f:
            json.dump(filtered_chunks, f, indent=2)
        
        print(f"\nSaved to {output_path}")
        
        return filtered_chunks