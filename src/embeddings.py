from langchain.embeddings import HuggingFaceEmbeddings

def get_embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu"):
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": device})
    print(f"Using {model_name} on {device}")
    return embeddings
