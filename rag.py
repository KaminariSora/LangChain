import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "D:/Working/C_work/Coding/LangChain/.env/service_account.json"



# 1. โหลดเอกสาร
loader = TextLoader("data.txt", encoding='utf-8')
documents = loader.load()
print(documents)

# 2. แบ่งชิ้นส่วนย่อยๆ
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
chunks = text_splitter.split_documents(documents)

# 3. แปลงข้อมูลเป็นเวกเตอร์์
from langchain_google_vertexai.embeddings import VertexAIEmbeddings

embedding = VertexAIEmbeddings(
    model_name="gemini-embedding-001",
    location="us-central1"
)

vertorStore = FAISS.from_documents(chunks,embedding)

retrievers = vertorStore.as_retriever()