from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader

def load_pdfs(data_path="data/"):
    loader = DirectoryLoader(
        data_path,
        glob="*pdf",
        loader_cls=PyMuPDFLoader
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} documents")
    return docs
