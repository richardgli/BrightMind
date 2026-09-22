from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_vector_store(texts: list[str]) -> Chroma:
    documents = [Document(page_content=t) for t in texts]
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_text(documents)

    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name="temp_collection",
        persist_directory="chroma_db"
    )

def load_vector_store() -> Chroma:
    return Chroma(
        collection_name="temp_collection",
        embedding_function=get_embeddings(),
        persist_directory="chroma_db",
    )

if __name__ == "__main__":
    sample = [
        "Twinkle, twinkle, little star, how I wonder what you are. Up above the world so high,"
        "like a diamond in the sky. Twinkle, twinkle, little star, how I wonder what you are."
        "When the blazing sun is set, and the grass with dew is wet. Then you show your little"
        "light, twinkle, twinkle all the night. Twinkle, twinkle little star, how I wonder what you are."
        "Then the traveler in the dark thanks you for your tiny spark. How could he see where to"
        "go if you did not twinkle so? Twinkle, twinkle little star, how I wonder what you are."
        "As your bright and tiny spark lights the traveler in the dark, though I know not what you"
        "are, twinkle, twinkle, little star. Twinkle, twinkle, little star, how I wonder what you are."
    ]

    build_vector_store(sample)
    print(f"Ingested {len(sample)} docs into 'chroma_db'")