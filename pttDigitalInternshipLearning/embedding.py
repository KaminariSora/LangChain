import os
from langchain_google_vertexai.embeddings import VertexAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

# ตั้งค่าคีย์ Google Cloud
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "D:/Working/C_work/Coding/LangChain/.env/service_account.json"

documents = [
    "ชื่อบริษัท : KamiSora",
    "ข้อมูลธุรกิจ : จัดจำหน่ายของเล่นและหนังสือการ์ตูน",
    "ผู้ก่อตั้ง : โซระ",
    "ปีที่ก่อตั้ง : 2568",
]

# สร้าง embedding model จาก Gemini
embedding_model = VertexAIEmbeddings(
    model_name="gemini-embedding-001",
    location="us-central1"
)

embeddings = []
for doc in documents:
    embedding_vector = embedding_model.embed_query(doc)
    embeddings.append(embedding_vector)

print(f"Number of embeddings: {len(embeddings)}")
print(f"Shape of one embedding vector: {len(embeddings[0])}")

