from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

embeddings_model = OllamaEmbeddings(
    model="hf.co/Qwen/Qwen3-Embedding-8B-GGUF:Q4_K_M"
)

vector_store = Chroma(
    collection_name="testCollection",
    embedding_function=embeddings_model,
    persist_directory="./basicLangchain/testing_chroma_db"
)

docs_to_delete = vector_store.get(
    where={"product_id": 10}
)

vector_store.delete(ids=docs_to_delete["ids"])
