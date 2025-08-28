import re
from langchain.schema import Document

def clean_text(text: str) -> str:
    text = re.sub(r'\bPage\s*\d+\b','',text,flags=re.IGNORECASE)
    text = re.sub(r'^\s*\d+\s*\$','',text,flags=re.MULTILINE)
    text = re.sub(r"(?i)\b(references|bibliography)\b.*", "", text, flags=re.DOTALL)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def clean_documents(docs):
    cleaned_docs = []
    for d in docs:
        cleaned_text = clean_text(d.page_content)
        cleaned_docs.append(Document(page_content=cleaned_text, metadata=d.metadata))
    print(f"Cleaned {len(cleaned_docs)} documents")
    return cleaned_docs
