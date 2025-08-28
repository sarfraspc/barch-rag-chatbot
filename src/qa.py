import os
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks.tracers import LangChainTracer

def create_qa_chain(vector_db, langsmith_project="b-arch-chatbot", model_name="llama3-70b-8192"):
    tracer = LangChainTracer(project_name=langsmith_project)

    llm = ChatGroq(
        model=model_name,
        api_key=os.getenv("GROQ_API_KEY"),
        verbose=True,
        callbacks=[tracer]
    )

    retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}
    )

    template = """
    You are an Expert Academic Research Assistant.

    Use only the provided context to answer the question. 
    Cite sections, pages, or PDF names whenever possible.

    If the answer is not in the context, respond: "I don't have enough information."

    Context:
    {context}

    Question:
    {question}

    Answer in a formal, academic style with step-by-step reasoning.
    """
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
        callbacks=[tracer]
    )

    print(f"RetrievalQA chain ready using {model_name} and LangSmith project '{langsmith_project}'")
    return chain