import os
import requests
from pathlib import Path
from serpapi import GoogleSearch

class WebExtractor:
    def __init__(self, config: dict):
        self.api_key = os.getenv("SERP_API_KEY")
        self.output_dir = Path(config["paths"]["raw_data"])
        self.num_results = config.get("num_results", 20)
        self.mode = config.get("mode", "pdf")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def search(self, query: str, start: int = 0) -> list[dict]:
        """Search Google, return list of results."""
        params = {
            "engine": "google",
            "q": query,
            "api_key": self.api_key,
            "num": 10,
            "start": start,
            "hl": "en",
            "gl": "us"
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        return results.get("organic_results", [])
    
    def download(self, url: str, filename: str) -> Path:
        """Download URL to file."""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
        response.raise_for_status()
        
        # Verify it's actually a PDF
        if response.content[:5] != b"%PDF-":
            raise ValueError("Not a valid PDF")
        
        filepath = self.output_dir / filename
        filepath.write_bytes(response.content)
        return filepath

    def extract(self, query: str) -> tuple[list[dict], list[dict]]:
        """Search and download until num_results successful downloads."""
        if self.mode == "pdf":
            query = f"{query} filetype:pdf"
        
        all_results = []
        downloaded = []
        start = 0
        doc_index = 0
        
        while len(downloaded) < self.num_results:
            results = self.search(query, start)
            
            if not results:  # No more results
                break
            
            all_results.extend(results)
            
            for result in results:
                if len(downloaded) >= self.num_results:
                    break
                
                url = result.get("link")
                is_pdf = ".pdf" in url
                
                if self.mode == "pdf" and not is_pdf:
                    continue
                if self.mode == "html" and is_pdf:
                    continue
                
                filename = f"doc_{doc_index:03d}.pdf" if is_pdf else f"doc_{doc_index:03d}.html"
                
                try:
                    path = self.download(url, filename)
                    downloaded.append({
                        "path": path,
                        "url": url,
                        "title": result.get("title"),
                        "source": result.get("source")
                    })
                    doc_index += 1
                except Exception as e:
                    print(f"Failed: {url}: {e}")
            
            start += 10
        
        return all_results, downloaded