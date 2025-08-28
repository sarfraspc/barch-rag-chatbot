import os
from dotenv import load_dotenv
from src.loader import load_pdfs
from src.cleaner import clean_documents
from src.splitter import split_docs
from src.embeddings import get_embeddings
from src.vectorstore import build_vectorstore

load_dotenv()

DATA_DIR = "data/"
DB_DIR = "chroma_db/"

def main():
    print("Loading PDFs")
    docs = load_pdfs(DATA_DIR)

    print("Cleaning texts")
    cleaned = clean_documents(docs)

    print("Splitting into chunks")
    splits = split_docs(cleaned)

    print("Creating embeddings")
    embeddings = get_embeddings()

    print("Building and saving vectorstore")
    build_vectorstore(splits, embeddings)

    print("Training complete. Vectorstore persisted at:", DB_DIR)

if __name__ == "__main__":
    main()