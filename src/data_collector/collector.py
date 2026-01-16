import json
from pathlib import Path
from web_extractor import WebExtractor
from document_extractor import DocumentExtractor

class Collector:
    def __init__(self, config: dict):
        self.config = config
        self.web_extractor = WebExtractor(config)
        self.document_extractor = DocumentExtractor(config)
        self.metadata_path = Path(config["paths"]["raw_data"]) / "metadata.json"
    
    def run(self, query: str) -> list[dict]:
        """Run full collection pipeline."""
        # Step 1: Search and download
        print(f"Searching for: {query}")
        results, downloaded = self.web_extractor.extract(query)
        print(f"Downloaded {len(downloaded)} files")
        
        # Step 2: Save metadata
        self._save_metadata(downloaded)
        
        # Step 3: Extract content to markdown
        print("Extracting content...")
        documents = self.document_extractor.extract_all()
        print(f"Extracted {len(documents)} documents")
        
        return documents
    
    def _save_metadata(self, downloaded: list[dict]) -> None:
        """Save download metadata to JSON."""
        metadata = []
        for item in downloaded:
            metadata.append({
                "id": item["path"].stem,
                "url": item["url"],
                "title": item["title"],
                "source": item["source"],
                "path": str(item["path"])
            })
        
        self.metadata_path.write_text(json.dumps(metadata, indent=2))