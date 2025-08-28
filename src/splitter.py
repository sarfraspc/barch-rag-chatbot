from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def split_docs(docs, chunk_size=250, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n","\n"," ",""]
    )
    all_chunks = []
    for doc in docs:
        chunks = splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            all_chunks.append(Document(page_content=chunk, metadata={**doc.metadata, 'chunk': i}))
    print(f"Created {len(all_chunks)} chunks")
    return all_chunks
