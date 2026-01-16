from pathlib import Path
from docling.document_converter import DocumentConverter

class DocumentExtractor:
    def __init__(self, config: dict):
        self.converter = DocumentConverter()
        self.input_dir = Path(config["paths"]["raw_data"])
        self.output_dir = Path(config["paths"]["processed_data"])
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def extract(self, file_path: Path) -> dict:
        """Extract content from single file (PDF or HTML)."""
        suffix = file_path.suffix.lower()
        
        if suffix == ".pdf":
            return self._extract_pdf(file_path)
        elif suffix in [".html", ".htm"]:
            # TODO: implement later
            raise NotImplementedError("HTML extraction not implemented yet")
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    
    def _extract_pdf(self, pdf_path: Path) -> dict:
        """Extract content from PDF using Docling."""
        result = self.converter.convert(str(pdf_path))
        content = result.document.export_to_markdown()
        
        # Save markdown file
        output_path = self.output_dir / f"{pdf_path.stem}.md"
        output_path.write_text(content)
        
        return {
            "id": pdf_path.stem,
            "source": str(pdf_path),
            "output": str(output_path),
            "content": content
        }
    
    def extract_all(self) -> list[dict]:
        """Extract from all files in input directory."""
        documents = []
        
        # Get both PDFs and HTMLs
        files = list(self.input_dir.glob("*.pdf")) + list(self.input_dir.glob("*.html"))
        
        for file_path in files:
            try:
                doc = self.extract(file_path)
                documents.append(doc)
            except Exception as e:
                print(f"Failed to extract {file_path}: {e}")
        
        return documents
    
# pipeline_options.ocr_options.force_full_page_ocr = True