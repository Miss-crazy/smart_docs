import re

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Splits text into chunks respecting paragraph and sentence boundaries.
    chunk_size and overlap are approximate character lengths.
    """
    if not text or not text.strip():
        return []
    
    # Normalize whitespace and clean text
    text = re.sub(r'\r\n|\r', '\n', text)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = []
    current_len = 0
    
    for para in paragraphs:
        # Split paragraph into sentences if paragraph is large
        sentences = re.split(r'(?<=[.!?])\s+', para)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_len = len(sentence)
            if current_len + sentence_len > chunk_size and current_chunk:
                chunk_str = " ".join(current_chunk)
                chunks.append(chunk_str)
                
                # Keep last few sentences for overlap
                overlap_chunk = []
                overlap_len = 0
                for s in reversed(current_chunk):
                    if overlap_len + len(s) <= overlap:
                        overlap_chunk.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break
                
                current_chunk = overlap_chunk
                current_len = overlap_len
            
            current_chunk.append(sentence)
            current_len += sentence_len
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks