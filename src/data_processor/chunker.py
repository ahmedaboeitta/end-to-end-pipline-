import re
from cleaner import clean_section, count_word_tokens


def split_into_sections(markdown_text: str) -> list[dict]:
    """Split markdown by ## headers into sections."""
    # Split by ## headers, keeping the header
    parts = re.split(r'(^## .+$)', markdown_text, flags=re.MULTILINE)
    
    sections = []
    current_title = None
    current_content = []
    
    for part in parts:
        if part.startswith('## '):
            # Save previous section if exists
            if current_title is not None:
                content = '\n'.join(current_content).strip()
                if content:
                    sections.append({
                        'section_title': current_title,
                        'content': content
                    })
            # Start new section
            current_title = part[3:].strip()
            current_content = []
        else:
            if current_title is not None:
                current_content.append(part)
            # Content before first header (if any) is ignored or could be intro
    
    # Don't forget last section
    if current_title is not None:
        content = '\n'.join(current_content).strip()
        if content:
            sections.append({
                'section_title': current_title,
                'content': content
            })
    
    return sections


def process_sections(sections: list[dict]) -> list[dict]:
    """Clean sections and add token counts."""
    processed = []
    for section in sections:
        cleaned_content = clean_section(section['content'])
        token_count = count_word_tokens(cleaned_content)
        
        processed.append({
            'section_title': section['section_title'],
            'content': cleaned_content,
            'token_count': token_count
        })
    
    return processed


def merge_small_sections(sections: list[dict], min_tokens: int, target_tokens: int) -> list[dict]:
    """Merge consecutive small sections into chunks."""
    if not sections:
        return []
    
    chunks = []
    current_chunk = {
        'section_titles': [sections[0]['section_title']],
        'contents': [sections[0]['content']],
        'token_count': sections[0]['token_count']
    }
    
    for section in sections[1:]:
        # If current chunk is small and adding this section keeps us under target
        if (current_chunk['token_count'] < min_tokens and 
            current_chunk['token_count'] + section['token_count'] <= target_tokens):
            # Merge
            current_chunk['section_titles'].append(section['section_title'])
            current_chunk['contents'].append(section['content'])
            current_chunk['token_count'] += section['token_count']
        else:
            # Save current chunk and start new one
            chunks.append(finalize_chunk(current_chunk))
            current_chunk = {
                'section_titles': [section['section_title']],
                'contents': [section['content']],
                'token_count': section['token_count']
            }
    
    # Don't forget last chunk
    chunks.append(finalize_chunk(current_chunk))
    
    return chunks


def finalize_chunk(chunk: dict) -> dict:
    """Convert chunk to final format."""
    # Combine contents with headers preserved
    combined_content = ''
    for title, content in zip(chunk['section_titles'], chunk['contents']):
        combined_content += f"## {title}\n\n{content}\n\n"
    
    return {
        'section_title': ' | '.join(chunk['section_titles']),
        'content': combined_content.strip(),
        'token_count': chunk['token_count']
    }


def filter_sections(sections: list[dict], min_tokens: int, max_tokens: int) -> list[dict]:
    """Filter out sections that are too short or flag too long ones."""
    filtered = []
    for section in sections:
        if section['token_count'] < min_tokens:
            continue  # Skip very short sections
        if section['token_count'] > max_tokens:
            print(f"Warning: Section '{section['section_title']}' has {section['token_count']} tokens (exceeds {max_tokens})")
        filtered.append(section)
    return filtered
