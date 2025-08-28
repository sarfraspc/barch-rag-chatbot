from langchain.vectorstores import Chroma

def build_vectorstore(documents, embeddings, persist_dir="chroma_db"):
    vector_db = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=persist_dir)
    vector_db.persist()
    print(f"Vector DB persisted at {persist_dir}")
    return vector_db
