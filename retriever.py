from langchain_core.tools.retriever import create_retriever_tool
from ingest import load_vector_store

def get_retriever_tool():
    vector_store = load_vector_store()

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    return create_retriever_tool(
        retriever=retriever,
        name="brightmind_knowledge",
        description="Search knowledge base. Use this tool for any question about the app's features, study material, or user notes."
    )