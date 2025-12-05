from src.embeddings import get_embeddings
from langchain_chroma import Chroma
from src.qa import create_qa_chain

DB_DIR = "chroma_db/"

def load_vectorstore(persist_dir=DB_DIR):
    embeddings = get_embeddings()
    vector_db = Chroma(persist_directory=persist_dir,embedding_function=embeddings)
    return vector_db

def qa_chain(vectorestore):
    chain = create_qa_chain(vectorestore)
    return chain
