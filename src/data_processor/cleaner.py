import re


def clean_section(text: str) -> str:
    """Apply all cleaning steps to a section."""
    text = remove_image_tags(text)
    text = remove_blank_form_fields(text)
    text = fix_html_entities(text)
    text = remove_page_number_lines(text)
    text = remove_dot_leader_lines(text)
    text = remove_photo_credits(text)
    text = clean_toc_table(text)
    text = clean_table_whitespace(text)
    text = normalize_whitespace(text)
    return text.strip()


def remove_image_tags(text: str) -> str:
    """Remove <!-- image --> tags."""
    return re.sub(r'<!--\s*image\s*-->', '', text)


def remove_blank_form_fields(text: str) -> str:
    """Remove lines with only underscores (blank form fields)."""
    return re.sub(r'^[_\s]+$', '', text, flags=re.MULTILINE)


def fix_html_entities(text: str) -> str:
    """Convert HTML entities to characters."""
    replacements = {
        '&amp;': '&',
        '&lt;': '<',
        '&gt;': '>',
        '&quot;': '"',
        '&#39;': "'",
    }
    for entity, char in replacements.items():
        text = text.replace(entity, char)
    return text


def remove_page_number_lines(text: str) -> str:
    """Remove lines that are just page numbers."""
    return re.sub(r'^\s*\d{1,3}\s*$', '', text, flags=re.MULTILINE)


def remove_dot_leader_lines(text: str) -> str:
    """Remove TOC-style lines with dot leaders (e.g., 'Chapter one ...........')."""
    return re.sub(r'^.*\.{4,}.*$', '', text, flags=re.MULTILINE)


def remove_photo_credits(text: str) -> str:
    """Remove photo credits blocks (lines with multiple image source references)."""
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        # Detect photo credits by pattern: multiple occurrences of source - page refs
        if re.search(r'(Alamy|Getty|Corbis|Shutterstock|istock|Image Source)', line, re.IGNORECASE):
            if line.count(' - ') > 2 or line.count(',') > 10:
                continue
        cleaned.append(line)
    return '\n'.join(cleaned)


def clean_toc_table(text: str) -> str:
    """Remove table of contents tables (detected by dot spacers + Page references)."""
    lines = text.split('\n')
    cleaned_lines = []
    in_toc_table = False
    
    for line in lines:
        # Detect TOC table by dot patterns and "Page" references
        if re.search(r'\.(\s+\.){5,}', line) and re.search(r'Page\s*\d+', line, re.IGNORECASE):
            in_toc_table = True
            continue
        # Table separator line while in TOC
        if in_toc_table and re.match(r'^\|[-|]+\|$', line.strip()):
            continue
        # Exit TOC table on non-table line
        if in_toc_table and not line.strip().startswith('|'):
            in_toc_table = False
        
        if not in_toc_table:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def clean_table_whitespace(text: str) -> str:
    """Normalize whitespace within table cells."""
    def clean_table_line(match):
        line = match.group(0)
        # Split by |, clean each cell, rejoin
        cells = line.split('|')
        cleaned_cells = [' '.join(cell.split()) for cell in cells]
        return '|'.join(cleaned_cells)
    
    return re.sub(r'^\|.*\|$', clean_table_line, text, flags=re.MULTILINE)


def normalize_whitespace(text: str) -> str:
    """Reduce multiple blank lines to single blank line."""
    return re.sub(r'\n{3,}', '\n\n', text)


def count_word_tokens(text: str) -> int:
    """Count only actual words (alphabetic sequences only, no numbers)."""
    words = re.findall(r'[a-zA-Z]+', text)
    return len(words)