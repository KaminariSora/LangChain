import os
from langchain_google_vertexai.embeddings import VertexAIEmbeddings
from langchain_core.documents import Document
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS  # ใช้ FAISS vector store

# กำหนด path ไปยัง service account ของ Google
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "D:/Working/C_work/Coding/LangChain/.env/service_account.json"

# โหลดเอกสารจากไฟล์
loader = TextLoader('data.txt', encoding='utf-8')
documents = loader.load()

# แบ่งข้อความเป็น chunk
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0
)
chunks = text_splitter.split_documents(documents)

# สร้าง embeddings ด้วย Gemini
embedding_model = VertexAIEmbeddings(
    model_name="gemini-embedding-001",
    location="us-central1"
)

# สร้าง FAISS vector store
# vector_store = FAISS.from_documents(chunks, embedding_model)
# vector_store.save_local("faiss_index/")
vector_store = FAISS.load_local(
    "faiss_index/", 
    embedding_model, 
    allow_dangerous_deserialization=True)

# คำค้นหา
query = "โซระ"
print(f"\nSearching for: '{query}'")

# ค้นหา similarity
results = vector_store.similarity_search(query, k=2)
print("\nTop 2 results:")
for i, result in enumerate(results, 1):
    print(f"{i}. {result.page_content}")

# สร้าง retriever จาก FAISS store
retriever = vector_store.as_retriever()
