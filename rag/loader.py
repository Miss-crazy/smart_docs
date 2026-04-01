import fitz

def load_pdf(file_path:str)-> str:
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text.strip()

def load_gtgxt(file_path:str)-> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()
    
def load_file(file_path:str)-> str:
    if file_path.endswith('.pdf'):
        return load_pdf(file_path)
    elif file_path.endswith('.txt'):
        return load_gtgxt(file_path)
    else:
        raise ValueError("Unsupported file type. Only .pdf and .txt are supported.")