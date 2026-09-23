from typing import List
import pymupdf
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_vector_store(documents: List[Document]) -> Chroma:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name="course_docs",
        persist_directory="chroma_db"
    )
    print(f"Vectors in Chroma: {vector_store._collection.count()}")


def load_vector_store() -> Chroma:
    return Chroma(
        collection_name="course_docs",
        embedding_function=get_embeddings(),
        persist_directory="chroma_db",
    )


if __name__ == "__main__":
    documents = []
    document = pymupdf.open("test.pdf")
    for i, page in enumerate(document):
        text = page.get_text()
        documents.append(Document(
            page_content=text,
            metadata={
                "source": "test.pdf",
                "page": i
            }
        ))
    build_vector_store(documents)